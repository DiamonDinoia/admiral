#pragma once

// ============================================================================
// Real-input FFT (r2c) and its inverse (c2r), built on the c2c engine.
//
// 1D (even N): the classic half-length trick. Pack the N real samples into
// N/2 complex numbers z[j] = x[2j] + i*x[2j+1], run an existing length-(N/2)
// complex plan_impl, then one recombination pass with the length-N twiddles
// splits the packed spectrum into the N/2+1 half-spectrum. c2r is the mirror
// (inverse recombination -> length-(N/2) inverse c2c -> unpack). No new kernels
// and no new twiddle machinery — just plan_impl + portable_trig.
//
//   ponytail: odd innermost N falls back to a full length-N c2c + slice
//   (correct, not optimal). The half-length trick needs an even split; the
//   benchmark r2c sizes are all even, so the odd path is a correctness fallback,
//   not a performance target. Upgrade path: a split-radix real kernel for odd N.
//
// N-D r2c = 1D r2c on the innermost (real) axis -> the half-spectrum, then the
// existing c2c column passes (nd_apply_axis) over the remaining axes on the
// complex tensor whose innermost extent is Nh = N/2+1. c2r reverses the order.
// This is the same row-column decomposition proven for c2c — no new N-D
// machinery. Output layout: innermost axis N/2+1 complex (Hermitian half-spectrum).
//
// Normalization matches the c2c API: forward (r2c) is unscaled; inverse (c2r)
// fully inverts, so r2c -> c2r is the identity. The inner c2r's length-(N/2)
// (or length-N) inverse already carries the 1/N factor; each outer inverse axis
// carries its own 1/len — the product is 1/Ntot.
// ============================================================================

#include <complex>
#include <concepts>   // std::invocable (run_tiles body constraint)
#include <cstddef>
#include <span>
#include <vector>

#include "nd_plan.hpp"        // nd_axis_state, make_nd_axis_state, nd_apply_axis
#include "plan.hpp"           // plan_impl
#include "portable_trig.hpp"  // sincos_turns
#include "twiddles.hpp"       // build_dif_factor_plan, dif_candidate_radices
#include "vecpass.hpp"        // vp::multipass_tables, vp::multipass_run (WS-B)
#include <memory>             // make_unique_for_overwrite
#include <poet/poet.hpp>      // poet::static_for (compile-time unroll of the tile loops)
#include <xsimd/xsimd.hpp>    // xsimd::batch

namespace admiral {
namespace detail {

// 1D real<->half-complex transform. Reusable: inner c2c plans (both directions)
// and the recombination twiddle ring are built once. Batch methods transform
// `rows` contiguous lines at once (the N-D outer axes call this with rows > 1),
// reusing a single scratch buffer.
template<typename T>
class real_adm_plan {
public:
    explicit real_adm_plan(std::size_t N)
        : N_(N), even_(N % 2 == 0), M_(even_ ? N / 2 : N),
          Nh_(N / 2 + 1),
          fwd_(M_, /*is_forward=*/true), inv_(M_, /*is_forward=*/false) {
        if (even_) {
            // W_N^k = e^{-2*pi*i k/N}, k = 0..N/2, for the recombination pass.
            const std::size_t half = N_ / 2;
            tw_.resize(half + 1);
            for (std::size_t k = 0; k <= half; ++k) {
                const auto [sn, cs] = portable_trig::sincos_turns<true>(k, N_);
                tw_[k] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
            }
            // WS-B: the row-batched fast path packs W real rows into SIMD lanes and
            // runs the inner size-M complex transform once per W-row tile via the
            // lane-batched DIF multipass (vecpass.hpp). Only usable when M>=2 and M
            // factors entirely into the dispatchable radix set (else the scalar
            // per-row plan_impl path below handles it). Tables are direction-aware.
            batched_ = M_ >= 2 && multipass_supported(M_);
            if (batched_) {
                tab_fwd_.template build<true>(static_cast<unsigned>(M_));
                tab_inv_.template build<false>(static_cast<unsigned>(M_));
                // One over-aligned ping-pong block reused across all tiles/executes
                // (single-threaded plan).
                tile_scratch_ = std::make_unique_for_overwrite<V[]>(4 * M_);
            }
        }
    }

    [[nodiscard]] std::size_t real_len() const noexcept { return N_; }
    [[nodiscard]] std::size_t cplx_len() const noexcept { return Nh_; }

    // Forward r2c: `rows` real rows of length N -> `rows` half-spectra of length Nh.
    // pool (may be null) threads the batched tile loop over rows (even path only).
    void r2c(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool = nullptr) const {
        if (even_) r2c_even(in, out, rows, pool);
        else       r2c_odd(in, out, rows, pool);
    }

    // Inverse c2r: `rows` half-spectra of length Nh -> `rows` real rows of
    // length N. Fully inverts (carries 1/N), so r2c -> c2r is the identity.
    void c2r(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool = nullptr) const {
        if (even_) c2r_even(in, out, rows, pool);
        else       c2r_odd(in, out, rows, pool);
    }

private:
    using V = xsimd::batch<T>;
    static constexpr std::size_t W = V::size;

    // True iff M factors entirely into the dispatchable radix set — the multipass
    // (vpass_dispatch over dif_radix_set) has no Bluestein/Rader escape, so an
    // out-of-set radix (a prime > 11, e.g. M=13) is not batchable.
    static bool multipass_supported(std::size_t M) {
        const auto fp = admiral::detail::build_dif_factor_plan<T>(M);
        if (fp.count == 0) return false;
        for (std::size_t p = 0; p < fp.count; ++p) {
            bool ok = false;
            for (unsigned c : admiral::detail::dif_candidate_radices)
                if (c == fp.radices[p]) ok = true;
            if (!ok) return false;
        }
        return true;
    }

    void r2c_even(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool) const {
        std::size_t done = 0;
        if (batched_) {
            const std::size_t ntiles = rows / W;
            run_tiles(pool, ntiles, rows * N_, [&](std::size_t t, V* sc) {
                r2c_even_tile(in + t * W * N_, out + t * W * Nh_, sc);
            });
            done = ntiles * W;
        }
        r2c_even_scalar(in + done * N_, out + done * Nh_, rows - done);  // <W tail (and non-batched)
    }

    void c2r_even(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool) const {
        std::size_t done = 0;
        if (batched_) {
            const std::size_t ntiles = rows / W;
            run_tiles(pool, ntiles, rows * N_, [&](std::size_t t, V* sc) {
                c2r_even_tile(in + t * W * Nh_, out + t * W * N_, sc);
            });
            done = ntiles * W;
        }
        c2r_even_scalar(in + done * Nh_, out + done * N_, rows - done);
    }

    // Run `ntiles` W-row tiles through `body(tile, scratch)`. Serial reuses the
    // single plan-owned tile_scratch_. Threaded gives each chunk its own 4*M
    // ping-pong block, allocated ONCE per chunk.
    template<typename Body>
        requires std::invocable<Body&, std::size_t, V*>
    void run_tiles(thread_pool* pool, std::size_t ntiles, std::size_t total_elems, Body&& body) const {
        const bool par = pool && ntiles >= 2 * pool->size() && total_elems >= kThreadMinElems;
        if (par) {
            pool->parallel_for(ntiles, [&](std::size_t b, std::size_t e, std::size_t) {
                auto sc = std::make_unique_for_overwrite<V[]>(4 * M_);
                for (std::size_t t = b; t < e; ++t) body(t, sc.get());
            });
        } else {
            for (std::size_t t = 0; t < ntiles; ++t) body(t, tile_scratch_.get());
        }
    }

    // --- Batched: W real rows -> W half-spectra, packed into SIMD lanes. The inner
    // size-M complex DFT runs once per tile (vp::multipass_run); the recombination
    // butterfly is V-wide across the W rows (twiddle scalar-broadcast per k). ---
    void r2c_even_tile(const T* in, std::complex<T>* out, V* scratch) const {
        const std::size_t M = M_, half = N_ / 2;
        V* cur_re = scratch;         V* cur_im = scratch + M;
        V* nxt_re = scratch + 2 * M; V* nxt_im = scratch + 3 * M;

        // Pack z[j]=x[2j]+i x[2j+1] for W rows into cur_re/cur_im (lane = row).
        // Transpose-load: one WxW in-register transpose of contiguous reals (W rows x
        // W reals) yields H=W/2 consecutive j — vunpck/vperm, ~2-4x the scalar
        // vinsertps gather (see four_step.hpp). M % H != 0 -> a scalar < H tail.
        constexpr std::size_t H = W / 2;
        std::size_t j = 0;
        for (; j + H <= M; j += H) {
            V t[W];
            poet::static_for<0, W>([&](auto l) { t[l] = V::load_unaligned(in + l * N_ + 2 * j); });
            xsimd::transpose(t, t + W);
            poet::static_for<0, H>([&](auto m) { cur_re[j + m] = t[2 * m]; cur_im[j + m] = t[2 * m + 1]; });
        }
        for (; j < M; ++j) {                                   // < H scalar tail
            alignas(V) T zr[W], zi[W];
            for (std::size_t l = 0; l < W; ++l) {
                const T* xr = in + l * N_;
                zr[l] = xr[2 * j]; zi[l] = xr[2 * j + 1];
            }
            cur_re[j] = V::load_aligned(zr);
            cur_im[j] = V::load_aligned(zi);
        }
        auto [Gre, Gim] = vp::multipass_run<T, true, V>(tab_fwd_, cur_re, cur_im, nxt_re, nxt_im);

        // Recombine to the half-spectrum X[k], k=0..M (=half). The butterfly is per-k
        // (Gre[ka]/Gre[kb] are already row-vectors); only the STORE is transpose-batched
        // over H consecutive k (mirror of pack). For a full block all k<M, so ka=k and
        // kb=M-k need no modulo (this also kills the per-k scalar `k%M` div); the k=M
        // Nyquist bin plus the M % H remainder go through the scalar tail.
        const V hv(T(0.5));
        std::size_t k = 0;
        for (; k + H <= M; k += H) {
            V t[W];
            poet::static_for<0, H>([&](auto m) {
                const std::size_t kk = k + m, kb = (kk == 0) ? 0 : (M - kk);
                const V zkr = Gre[kk], zki = Gim[kk];
                const V zcr = Gre[kb], zci = -Gim[kb];         // conj(z[M-k])
                const V Zer = (zkr + zcr) * hv, Zei = (zki + zci) * hv;
                const V Zor = (zki - zci) * hv, Zoi = (zcr - zkr) * hv;
                const V twr(tw_[kk].real()), twi(tw_[kk].imag());
                t[2 * m]     = Zer + (twr * Zor - twi * Zoi);  // Xr = Ze + tw*Zo (re)
                t[2 * m + 1] = Zei + (twr * Zoi + twi * Zor);  // Xi
            });
            xsimd::transpose(t, t + W);
            poet::static_for<0, W>([&](auto l) {
                t[l].store_unaligned(reinterpret_cast<T*>(out + l * Nh_ + k));
            });
        }
        for (; k <= half; ++k) {                               // k=M Nyquist + < H tail
            const std::size_t ka = (k == M) ? 0 : k, kb = (ka == 0) ? 0 : (M - ka);
            const V zkr = Gre[ka], zki = Gim[ka];
            const V zcr = Gre[kb], zci = -Gim[kb];             // conj(z[M-k])
            const V Zer = (zkr + zcr) * hv, Zei = (zki + zci) * hv;
            const V Zor = (zki - zci) * hv, Zoi = (zcr - zkr) * hv;
            const V twr(tw_[k].real()), twi(tw_[k].imag());
            const V Xr = Zer + (twr * Zor - twi * Zoi);        // X = Ze + tw*Zo
            const V Xi = Zei + (twr * Zoi + twi * Zor);
            alignas(V) T xr[W], xi[W];
            Xr.store_aligned(xr); Xi.store_aligned(xi);
            for (std::size_t l = 0; l < W; ++l)
                out[l * Nh_ + k] = std::complex<T>(xr[l], xi[l]);
        }
    }

    void c2r_even_tile(const std::complex<T>* in, T* out, V* scratch) const {
        const std::size_t M = M_;
        V* cur_re = scratch;         V* cur_im = scratch + M;
        V* nxt_re = scratch + 2 * M; V* nxt_im = scratch + 3 * M;

        // Build Z[k], k=0..M-1, from the pair (in[k], in[M-k]). Transpose-load both
        // the ascending in[k0..k0+H-1] block and the in[M-k] block, which for k in a
        // block is the *descending* run in[M-k0..M-k0-H+1] = the ascending contiguous
        // block starting at mb=M-k0-H+1, consumed in reverse (p=H-1-m -> mb+p=M-kk).
        // The butterfly stays per-k; only the strided complex I/O is vectorized.
        const V hv(T(0.5));
        constexpr std::size_t H = W / 2;
        std::size_t k = 0;
        for (; k + H <= M; k += H) {
            V tk[W], tm[W];
            const std::size_t mb = M - k - H + 1;              // in[M-k] block base (ascending)
            poet::static_for<0, W>([&](auto l) {
                tk[l] = V::load_unaligned(reinterpret_cast<const T*>(in + l * Nh_ + k));
                tm[l] = V::load_unaligned(reinterpret_cast<const T*>(in + l * Nh_ + mb));
            });
            xsimd::transpose(tk, tk + W);                      // tk[2m]=re(in[k+m]), tk[2m+1]=im
            xsimd::transpose(tm, tm + W);                      // tm[2p]=re(in[mb+p]), tm[2p+1]=im
            poet::static_for<0, H>([&](auto m) {
                const std::size_t kk = k + m, p = H - 1 - m;   // in[M-kk] = in[mb+p]
                const V Xkr = tk[2 * m], Xki = tk[2 * m + 1];
                const V Xmr = tm[2 * p], Xmi = tm[2 * p + 1];
                const V Zer = (Xkr + Xmr) * hv, Zei = (Xki - Xmi) * hv;
                const V Vor = (Xkr - Xmr) * hv, Voi = (Xki + Xmi) * hv;
                const V twr(tw_[kk].real()), twi(tw_[kk].imag());
                const V Zor = twr * Vor + twi * Voi;
                const V Zoi = twr * Voi - twi * Vor;
                cur_re[kk] = Zer - Zoi;
                cur_im[kk] = Zei + Zor;
            });
        }
        for (; k < M; ++k) {                                   // < H scalar tail
            alignas(V) T xkr[W], xki[W], xmr[W], xmi[W];
            for (std::size_t l = 0; l < W; ++l) {
                const std::complex<T>* Xr = in + l * Nh_;
                xkr[l] = Xr[k].real();     xki[l] = Xr[k].imag();
                xmr[l] = Xr[M - k].real(); xmi[l] = Xr[M - k].imag();  // k==0 -> Xr[M]
            }
            const V Xkr = V::load_aligned(xkr), Xki = V::load_aligned(xki);
            const V Xmr = V::load_aligned(xmr), Xmi = V::load_aligned(xmi);
            const V Zer = (Xkr + Xmr) * hv, Zei = (Xki - Xmi) * hv;   // Ze, Xmk=conj(in[M-k])
            const V Vor = (Xkr - Xmr) * hv, Voi = (Xki + Xmi) * hv;   // Vo=(Xk-Xmk)*0.5
            const V twr(tw_[k].real()), twi(tw_[k].imag());
            const V Zor = twr * Vor + twi * Voi;               // Zo = conj(tw)*Vo
            const V Zoi = twr * Voi - twi * Vor;
            cur_re[k] = Zer - Zoi;                             // Z = Ze + i*Zo
            cur_im[k] = Zei + Zor;
        }
        auto [Gre, Gim] = vp::multipass_run<T, false, V>(tab_inv_, cur_re, cur_im, nxt_re, nxt_im);

        // Unpack z[j] -> x[2j],x[2j+1] for W rows (mirror of the r2c pack): H=W/2
        // consecutive j per WxW transpose-store, scalar < H tail.
        const V invM(T(1) / static_cast<T>(M));                // multipass is UN-normalized
        std::size_t j = 0;
        for (; j + H <= M; j += H) {
            V t[W];
            poet::static_for<0, H>([&](auto m) { t[2 * m] = Gre[j + m] * invM; t[2 * m + 1] = Gim[j + m] * invM; });
            xsimd::transpose(t, t + W);
            poet::static_for<0, W>([&](auto l) { t[l].store_unaligned(out + l * N_ + 2 * j); });
        }
        for (; j < M; ++j) {                                   // < H scalar tail
            alignas(V) T zr[W], zi[W];
            (Gre[j] * invM).store_aligned(zr);
            (Gim[j] * invM).store_aligned(zi);
            for (std::size_t l = 0; l < W; ++l) {
                T* xr = out + l * N_;
                xr[2 * j] = zr[l]; xr[2 * j + 1] = zi[l];
            }
        }
    }

    // Scalar per-row path: the <W row tail, non-batchable M, and (via r2c_even) the
    // whole transform when the multipass can't take M.
    void r2c_even_scalar(const T* in, std::complex<T>* out, std::size_t rows) const {
        const std::size_t M = M_, half = N_ / 2;
        std::vector<std::complex<T>> z(M == 0 ? 1 : M);
        for (std::size_t r = 0; r < rows; ++r) {
            const T* xr = in + r * N_;
            for (std::size_t j = 0; j < M; ++j) z[j] = std::complex<T>(xr[2 * j], xr[2 * j + 1]);
            fwd_.execute(std::span<std::complex<T>>(z.data(), M));  // Z = DFT_M(z)
            std::complex<T>* Xr = out + r * Nh_;
            for (std::size_t k = 0; k <= half; ++k) {
                const std::complex<T> zk = z[k % M];
                const std::complex<T> zc = std::conj(z[(M - k % M) % M]);
                const std::complex<T> Ze = (zk + zc) * T(0.5);
                // Zo = (zk - zc) / (2i) = (zk - zc) * (-i/2)
                const std::complex<T> Zo = (zk - zc) * std::complex<T>(T(0), T(-0.5));
                Xr[k] = Ze + tw_[k] * Zo;
            }
        }
    }

    void c2r_even_scalar(const std::complex<T>* in, T* out, std::size_t rows) const {
        const std::size_t M = M_;
        std::vector<std::complex<T>> Z(M == 0 ? 1 : M);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::complex<T>* Xr = in + r * Nh_;
            for (std::size_t k = 0; k < M; ++k) {
                const std::complex<T> Xk  = Xr[k];
                const std::complex<T> Xmk = std::conj(Xr[M - k]);  // k==0 -> Xr[M]
                const std::complex<T> Ze = (Xk + Xmk) * T(0.5);
                const std::complex<T> Vo = (Xk - Xmk) * T(0.5);    // = W_N^k Zo[k]
                const std::complex<T> Zo = std::conj(tw_[k]) * Vo; // Zo[k] = W_N^{-k} Vo
                Z[k] = Ze + std::complex<T>(T(0), T(1)) * Zo;
            }
            inv_.execute(std::span<std::complex<T>>(Z.data(), M));  // z = IDFT_M(Z), *1/M
            T* xr = out + r * N_;
            for (std::size_t j = 0; j < M; ++j) { xr[2 * j] = Z[j].real(); xr[2 * j + 1] = Z[j].imag(); }
        }
    }

    // Odd-N fallback: full length-N c2c, slice the first Nh outputs (r2c) /
    // rebuild the conjugate-symmetric full spectrum then inverse c2c (c2r).
    // Rows are independent (each is a gather -> c2c -> slice), so the loop is a
    // drop-in parallel_for over rows with a per-chunk scratch vector.
    void r2c_odd(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            std::vector<std::complex<T>> c(N_);
            for (std::size_t r = b; r < e; ++r) {
                const T* xr = in + r * N_;
                for (std::size_t i = 0; i < N_; ++i) c[i] = std::complex<T>(xr[i], T(0));
                fwd_.execute(std::span<std::complex<T>>(c.data(), N_));
                std::complex<T>* Xr = out + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) Xr[k] = c[k];
            }
        });
    }

    void c2r_odd(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            std::vector<std::complex<T>> c(N_);
            for (std::size_t r = b; r < e; ++r) {
                const std::complex<T>* Xr = in + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) c[k] = Xr[k];
                for (std::size_t k = Nh_; k < N_; ++k) c[k] = std::conj(Xr[N_ - k]);
                inv_.execute(std::span<std::complex<T>>(c.data(), N_));  // *1/N
                T* xr = out + r * N_;
                for (std::size_t i = 0; i < N_; ++i) xr[i] = c[i].real();
            }
        });
    }

    // Declaration order matters: even_ before M_ (M_ is computed from even_),
    // and both before fwd_/inv_ (built from M_) — members initialise in
    // declaration order, not mem-init-list order.
    std::size_t N_;
    bool even_;
    std::size_t M_, Nh_;
    plan_impl<T> fwd_, inv_;
    std::vector<std::complex<T>> tw_;  // recombination twiddles (even path only)
    bool batched_ = false;             // WS-B: row-batched multipass fast path usable
    vp::multipass_tables<T> tab_fwd_, tab_inv_;  // inner size-M DIF tables (even+batched)
    mutable std::unique_ptr<V[]> tile_scratch_;  // 4*M ping-pong, reused across tiles
};

// N-D real transform (bidirectional): forward = r2c, inverse = c2r. The real
// tensor is shape[0]*...*shape[n-1] contiguous row-major (last axis fastest, and
// real); the complex half-spectrum tensor replaces the innermost extent with
// Nh = shape[n-1]/2 + 1. Outer axes reuse the c2c column-pass machinery.
template<typename T>
class nd_real_plan {
public:
    // NB: kept inline (not extern-templated like plan_impl / nd_runtime_plan).
    // Out-of-lining forward/inverse loses the call-site context the optimizer
    // needs to make the serial (pool==null) and threaded row/column passes
    // contract FMAs identically — under -ffast-math that surfaced as a 1-ULP
    // serial-vs-threaded divergence (test_fft_threads r2c bit-identity). The
    // r2c engine lives in far fewer TUs than the c2c engine, so leaving it
    // inline costs little compile memory and preserves the determinism guarantee.
    explicit nd_real_plan(std::span<const std::size_t> shape) {
        m.shape.assign(shape.begin(), shape.end());
        const std::size_t n = m.shape.size();
        for (auto e : m.shape) {
            if (e == 0) throw std::invalid_argument("Plan size must be greater than 0");
        }
        m.inner_len = m.shape.back();
        m.Nh = m.inner_len / 2 + 1;
        m.rp.emplace(m.inner_len);
        // Product of the outer extents (== number of innermost rows).
        m.rows = 1;
        for (std::size_t d = 0; d + 1 < n; ++d) m.rows *= m.shape[d];
        m.total_c = m.rows * m.Nh;
        // Per-axis c2c states for the outer axes (never innermost), both dirs. The
        // per-axis `inner` in the complex tensor (innermost extent Nh) drives the
        // small_inner predicate; compute it innermost-first, matching run_outer().
        const std::size_t nouter = (n == 0) ? 0 : n - 1;
        m.fwd_axes.resize(nouter);
        m.inv_axes.resize(nouter);
        std::size_t inner = m.Nh;
        for (std::size_t di = 0; di < nouter; ++di) {
            const std::size_t d = nouter - 1 - di;
            m.fwd_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, /*is_forward=*/true,  /*innermost=*/false);
            m.inv_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, /*is_forward=*/false, /*innermost=*/false);
            inner *= m.shape[d];
        }
    }

    [[nodiscard]] std::size_t cplx_size() const noexcept { return m.total_c; }
    [[nodiscard]] std::size_t real_size() const noexcept { return m.rows * m.inner_len; }
    [[nodiscard]] std::size_t inner_cplx() const noexcept { return m.Nh; }

    // Forward r2c: real tensor `in` -> complex half-spectrum `out`. opts.pool (may
    // be null) threads the innermost real tile loop and the outer c2c column
    // passes. opts.fct scales the output (default r2c is unscaled); a custom
    // factor is applied as one post-scale — the woven default path is untouched.
    void forward(const T* in, std::complex<T>* out, exec_options<T> opts = {}) const {
        m.rp->r2c(in, out, m.rows, opts.pool);                       // innermost real axis
        run_outer(out, m.fwd_axes, /*is_forward=*/true, opts.pool);  // remaining axes (c2c)
        if (opts.fct && *opts.fct != T(1))
            for (std::size_t i = 0; i < m.total_c; ++i) out[i] *= *opts.fct;
    }

    // Inverse c2r: complex half-spectrum `spec` (consumed) -> real tensor `out`.
    // Default divides by the real element count; a custom opts.fct is applied by
    // rescaling that normalized result (one post-scale; default path untouched).
    void inverse(std::complex<T>* spec, T* out, exec_options<T> opts = {}) const {
        run_outer(spec, m.inv_axes, /*is_forward=*/false, opts.pool);  // outer axes first
        m.rp->c2r(spec, out, m.rows, opts.pool);                       // then innermost c2r
        if (opts.fct) {
            const std::size_t Nr = real_size();
            const T def = T(1) / static_cast<T>(Nr);
            if (*opts.fct != def) {
                const T s = *opts.fct * static_cast<T>(Nr);
                for (std::size_t i = 0; i < Nr; ++i) out[i] *= s;
            }
        }
    }

private:
    // Apply the c2c column passes over every outer axis of the complex tensor
    // (innermost extent Nh, never transformed). inner starts at Nh.
    void run_outer(std::complex<T>* data, const std::vector<nd_axis_state<T>>& axes,
                   bool is_forward, thread_pool* pool) const {
        const std::size_t nouter = axes.size();
        std::size_t inner = m.Nh;
        for (std::size_t di = 0; di < nouter; ++di) {
            const std::size_t d = nouter - 1 - di;  // innermost outer axis first
            nd_apply_axis<T>(data, m.total_c, m.shape[d], inner,
                             /*innermost=*/false, is_forward, axes[d], std::nullopt, pool);
            inner *= m.shape[d];
        }
    }

    struct M {
        std::vector<std::size_t> shape;
        std::size_t inner_len;   // shape.back() (real innermost extent)
        std::size_t Nh;          // inner_len/2 + 1
        std::size_t rows;        // product of outer extents
        std::size_t total_c;     // rows * Nh
        std::optional<real_adm_plan<T>> rp;
        std::vector<nd_axis_state<T>> fwd_axes, inv_axes;
    } m;
};

} // namespace detail
} // namespace admiral
