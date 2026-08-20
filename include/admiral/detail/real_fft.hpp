#pragma once

// ============================================================================
// Real-to-complex (r2c) and inverse (c2r), built on the c2c engine.
//
// 1D even-N: half-length trick. Pack N reals into N/2 complex z[j]=x[2j]+i*x[2j+1],
// run length-(N/2) plan_impl, then one recombination pass (length-N twiddles) yields
// the N/2+1 half-spectrum. c2r: inverse recombination -> IDFT_{N/2} -> unpack.
// No new kernels or twiddle machinery; just plan_impl + portable_trig.
//
//   Odd N: full length-N c2c + slice (correctness fallback, never a performance
//   target; benchmark r2c sizes are all even).
//
// N-D: 1D r2c on the innermost axis -> complex tensor (Nh=N/2+1 innermost extent);
// c2c column passes (nd_apply_axis) over remaining axes. c2r reverses the order.
//
// Normalization: r2c unscaled; c2r carries 1/N in the inner IDFT and 1/len per outer
// axis (product = 1/Ntot). r2c -> c2r is the identity.
// ============================================================================

#include <complex>
#include <concepts>   // std::invocable (run_tiles body constraint)
#include <cstddef>
#include <memory>      // make_unique_for_overwrite
#include <span>
#include <utility>     // std::pair
#include <vector>

#include "nd_plan.hpp"        // nd_axis_state, make_nd_axis_state, nd_apply_axis
#include "plan.hpp"           // plan_impl
#include "portable_trig.hpp"  // sincos_turns
#include "twiddles.hpp"       // build_dif_factor_plan, dif_radix_set
#include "vecpass.hpp"        // vp::multipass_tables, vp::multipass_run (WS-B)
#include <poet/poet.hpp>      // poet::static_for (tile loop unroll)
#include "simd.hpp"    // xsimd::batch
// Last of the includes, paired with undef_macros.hpp at the end of the file: every
// sibling header above re-includes macros.hpp itself, which is an error while ours
// is still defined.
#include "macros.hpp"         // ADM_NOINLINE, ADM_COLD (trace())

namespace admiral {
namespace detail {

// 1D real<->half-complex: inner c2c plans (both dirs) and twiddle ring built once.
// `rows` contiguous lines per call; N-D outer axes pass rows>1.
template<typename T>
class real_adm_plan {
public:
    // Defined out-of-line below so `extern template` actually suppresses
    // instantiation in consumer TUs: [temp.explicit]/12 exempts inline functions,
    // and a member defined in the class body is implicitly inline.
    // eff flows to the inner 1-D engine (fwd_/inv_).
    explicit real_adm_plan(std::size_t N, admiral::effort eff = admiral::effort::estimate);

    // Forward r2c: `rows` real rows of length N -> `rows` half-spectra of length Nh.
    // pool (may be null) threads the batched tile loop over rows (even path only).
    void r2c(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool = nullptr) const;

    // Inverse c2r: `rows` half-spectra of length Nh -> `rows` real rows of
    // length N. Fully inverts (carries 1/N), so r2c -> c2r is the identity.
    void c2r(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool = nullptr) const;

private:
    using V = xsimd::batch<T>;
    static constexpr std::size_t W = V::size;

    // True iff M factors into dif_radix_set (no Bluestein/Rader escape).
    // A prime > 11 (e.g. M=13) is not batchable.
    static bool multipass_supported(std::size_t M) {
        const auto fp = admiral::detail::build_dif_factor_plan<T>(M);
        if (fp.count == 0) return false;
        for (std::size_t p = 0; p < fp.count; ++p)
            if (!in_seq(admiral::detail::dif_radix_set{}, fp.radices[p])) return false;
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

    // Run `ntiles` tiles through body(tile, scratch). Serial: reuses tile_scratch_.
    // Threaded: per-chunk 4*M scratch, allocated once per chunk.
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

    // --- Batched W-row tile: W rows -> W half-spectra. Inner size-M DFT via
    // vp::multipass_run; recombination butterfly V-wide (scalar twiddle broadcast per k). ---
    // Half-spectrum recombine per frequency: Ze = even part of the pair, Zo = odd part.
    // r2c: X = Ze + tw*Zo; c2r: Z = Ze + i*Zo with conj(tw) applied to Vo.
    static ADM_ALWAYS_INLINE std::pair<V, V> r2c_recombine(V zkr, V zki, V zcr, V zci, V hv, V twr,
                                          V twi) {
        const V Zer = (zkr + zcr) * hv, Zei = (zki + zci) * hv;
        const V Zor = (zki - zci) * hv, Zoi = (zcr - zkr) * hv;
        return {Zer + (twr * Zor - twi * Zoi), Zei + (twr * Zoi + twi * Zor)};
    }
    static ADM_ALWAYS_INLINE std::pair<V, V> c2r_recombine(V Xkr, V Xki, V Xmr, V Xmi, V hv, V twr,
                                          V twi) {
        const V Zer = (Xkr + Xmr) * hv, Zei = (Xki - Xmi) * hv;
        const V Vor = (Xkr - Xmr) * hv, Voi = (Xki + Xmi) * hv;
        const V Zor = twr * Vor + twi * Voi, Zoi = twr * Voi - twi * Vor;
        return {Zer - Zoi, Zei + Zor};
    }

    void r2c_even_tile(const T* in, std::complex<T>* out, V* scratch) const {
        const std::size_t M = M_, half = N_ / 2;
        V* cur_re = scratch;         V* cur_im = scratch + M;
        V* nxt_re = scratch + 2 * M; V* nxt_im = scratch + 3 * M;

        // Pack z[j]=x[2j]+i*x[2j+1] for W rows. WxW transpose of W rows×W reals
        // yields H=W/2 consecutive j (vunpck/vperm beats scalar vinsertps;
        // see four_step.hpp). M%H != 0 -> scalar < H tail.
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
        // Plain pointers, not a structured binding: the static_for lambdas below capture
        // these, and AppleClang rejects capturing a binding (C++20 P1091).
        const auto G = vp::multipass_run<T, true, V>(tab_, cur_re, cur_im, nxt_re, nxt_im);
        const V* Gre = G.first;
        const V* Gim = G.second;

        // Recombine to half-spectrum X[k], k=0..M. Butterfly per-k; store
        // transpose-batched over H consecutive k (mirror of pack). Full block: k<M,
        // ka=k, kb=M-k, no modulo. k=M Nyquist + M%H remainder go scalar.
        const V hv(T(0.5));
        std::size_t k = 0;
        for (; k + H <= M; k += H) {
            V t[W];
            poet::static_for<0, H>([&](auto m) {
                const std::size_t kk = k + m, kb = (kk == 0) ? 0 : (M - kk);
                const V zkr = Gre[kk], zki = Gim[kk];
                const V zcr = Gre[kb], zci = -Gim[kb];         // conj(z[M-k])
                const V twr(tw_[kk].real()), twi(tw_[kk].imag());
                const auto [Xr, Xi] = r2c_recombine(zkr, zki, zcr, zci, hv, twr, twi);
                t[2 * m] = Xr;
                t[2 * m + 1] = Xi;
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
            const V twr(tw_[k].real()), twi(tw_[k].imag());
            const auto [Xr, Xi] = r2c_recombine(zkr, zki, zcr, zci, hv, twr, twi);
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

        // Build Z[k] from (in[k], in[M-k]) for W rows. Ascending k block + descending
        // conjugate block (mb=M-k0-H+1, reverse: p=H-1-m -> mb+p=M-kk), both loaded
        // via WxW transpose. Butterfly per-k; strided I/O vectorized.
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
                const V twr(tw_[kk].real()), twi(tw_[kk].imag());
                const auto [Zr, Zi] = c2r_recombine(Xkr, Xki, Xmr, Xmi, hv, twr, twi);
                cur_re[kk] = Zr;
                cur_im[kk] = Zi;
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
            const V twr(tw_[k].real()), twi(tw_[k].imag());
            const auto [Zr, Zi] = c2r_recombine(Xkr, Xki, Xmr, Xmi, hv, twr, twi);
            cur_re[k] = Zr;                                    // Z = Ze + i*Zo
            cur_im[k] = Zi;
        }
        const auto G = vp::multipass_run<T, false, V>(tab_, cur_re, cur_im, nxt_re, nxt_im);
        const V* Gre = G.first;   // plain pointers: the lambdas capture them, see r2c_even_tile
        const V* Gim = G.second;

        // Unpack z[j] -> x[2j],x[2j+1] for W rows: H=W/2 j per WxW transpose-store, scalar < H tail.
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

    // Scalar per-row: <W row tail, non-batchable M, or multipass fallback.
    void r2c_even_scalar(const T* in, std::complex<T>* out, std::size_t rows) const {
        const std::size_t M = M_, half = N_ / 2;
        // Uninitialized: the deinterleave below fills all M entries every row.
        const auto z = std::make_unique_for_overwrite<std::complex<T>[]>(M == 0 ? 1 : M);
        for (std::size_t r = 0; r < rows; ++r) {
            const T* xr = in + r * N_;
            for (std::size_t j = 0; j < M; ++j) z[j] = std::complex<T>(xr[2 * j], xr[2 * j + 1]);
            fwd_.execute(std::span<std::complex<T>>(z.get(), M));  // Z = DFT_M(z)
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
        // Uninitialized: the recombination below fills all M entries every row.
        const auto Z = std::make_unique_for_overwrite<std::complex<T>[]>(M == 0 ? 1 : M);
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
            inv_.execute(std::span<std::complex<T>>(Z.get(), M));  // z = IDFT_M(Z), *1/M
            T* xr = out + r * N_;
            for (std::size_t j = 0; j < M; ++j) { xr[2 * j] = Z[j].real(); xr[2 * j + 1] = Z[j].imag(); }
        }
    }

    // Odd-N fallback: full length-N c2c + slice (r2c); rebuild conjugate-symmetric
    // spectrum + IDFT (c2r). Per-row parallel_for with per-chunk scratch.
    void r2c_odd(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            // Uninitialized: the loops below fill all N_ entries every row.
            const auto c = std::make_unique_for_overwrite<std::complex<T>[]>(N_);
            for (std::size_t r = b; r < e; ++r) {
                const T* xr = in + r * N_;
                for (std::size_t i = 0; i < N_; ++i) c[i] = std::complex<T>(xr[i], T(0));
                fwd_.execute(std::span<std::complex<T>>(c.get(), N_));
                std::complex<T>* Xr = out + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) Xr[k] = c[k];
            }
        });
    }

    void c2r_odd(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            // Uninitialized: the loops below fill all N_ entries every row.
            const auto c = std::make_unique_for_overwrite<std::complex<T>[]>(N_);
            for (std::size_t r = b; r < e; ++r) {
                const std::complex<T>* Xr = in + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) c[k] = Xr[k];
                for (std::size_t k = Nh_; k < N_; ++k) c[k] = std::conj(Xr[N_ - k]);
                inv_.execute(std::span<std::complex<T>>(c.get(), N_));  // *1/N
                T* xr = out + r * N_;
                for (std::size_t i = 0; i < N_; ++i) xr[i] = c[i].real();
            }
        });
    }

    // Init order = declaration order: even_ before M_ (M_ derived from even_),
    // both before fwd_/inv_ (built from M_).
    std::size_t N_;
    bool even_;
    std::size_t M_, Nh_;
    plan_impl<T> fwd_, inv_;
    std::vector<std::complex<T>> tw_;  // recombination twiddles (even path only)
    bool batched_ = false;             // WS-B row-batched fast path enabled
    vp::multipass_tables<T> tab_;  // inner size-M DIF tables (even+batched); direction-free
    // 4*M ping-pong, reused across tiles by the SERIAL run_tiles branch only (the
    // threaded branch allocates per chunk). Unlike four_step_large_plan, this makes
    // r2c/c2r NON-re-entrant: nd_real_plan hands the pool down rather than calling
    // r2c from inside its own parallel_for, so nothing shares one plan across threads.
    std::unique_ptr<V[]> tile_scratch_;
};

template<typename T>
real_adm_plan<T>::real_adm_plan(std::size_t N, admiral::effort eff)
    : N_(N), even_(N % 2 == 0), M_(even_ ? N / 2 : N),
      Nh_(N / 2 + 1),
      fwd_(M_, /*is_forward=*/true, 1, nullptr, eff),
      inv_(M_, /*is_forward=*/false, 1, nullptr, eff) {
    if (even_) {
        // W_N^k = e^{-2*pi*i k/N}, k = 0..N/2, for the recombination pass.
        const std::size_t half = N_ / 2;
        tw_.resize(half + 1);
        for (std::size_t k = 0; k <= half; ++k) {
            const auto [sn, cs] = portable_trig::sincos_turns<true>(k, N_);
            tw_[k] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        // WS-B fast path: W real rows per SIMD tile, inner size-M DIF multipass
        // (vecpass.hpp). Active when M>=2 and M factors into dif_radix_set;
        // else scalar plan_impl handles it. One direction-free table pair
        // serves both r2c and c2r (swapped-domain conjugation, vecpass.hpp).
        batched_ = M_ >= 2 && multipass_supported(M_);
        if (batched_) {
            tab_.build(M_);
            // One over-aligned ping-pong block reused across all tiles/executes
            // (single-threaded plan).
            tile_scratch_ = std::make_unique_for_overwrite<V[]>(4 * M_);
        }
    }
}

template<typename T>
void real_adm_plan<T>::r2c(const T* in, std::complex<T>* out, std::size_t rows,
                           thread_pool* pool) const {
    if (even_) r2c_even(in, out, rows, pool);
    else       r2c_odd(in, out, rows, pool);
}

template<typename T>
void real_adm_plan<T>::c2r(const std::complex<T>* in, T* out, std::size_t rows,
                           thread_pool* pool) const {
    if (even_) c2r_even(in, out, rows, pool);
    else       c2r_odd(in, out, rows, pool);
}

// N-D real transform (r2c/c2r). Row-major real tensor (last axis fastest);
// half-spectrum innermost extent Nh = shape[n-1]/2+1. Outer axes via c2c column passes.
template<typename T>
class nd_real_plan {
public:
    // Out-of-line (see real_adm_plan above). The WHOLE tree has to sit behind one
    // instantiation: move only part of it and the serial path inlines and contracts
    // where the threaded path calls, breaking the 1-vs-N-thread bit identity that
    // test_fft_threads checks for r2c.
    // nthreads > 1 builds the plan-owned pool (tile + column loops thread on it
    // at execute; no per-call threading knob exists).
    explicit nd_real_plan(std::span<const std::size_t> shape, std::size_t nthreads = 1,
                          admiral::effort eff = admiral::effort::estimate);

    [[nodiscard]] std::size_t cplx_size() const noexcept { return m.total_c; }
    [[nodiscard]] std::size_t real_size() const noexcept { return m.rows * m.inner_len; }

    // r2c: real `in` -> half-spectrum `out`. opts.fct scales output (default:
    // unscaled), applied as one post-pass.
    void forward(const T* in, std::complex<T>* out, const exec_options<T>& opts = {}) const;

    // c2r: half-spectrum `spec` (consumed) -> real `out`. Default: 1/Ntot;
    // custom fct rescales (one post-pass).
    void inverse(std::complex<T>* spec, T* out, const exec_options<T>& opts = {}) const;

private:
    // c2c column passes over outer axes (innermost Nh not transformed).
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
        std::unique_ptr<thread_pool> pool;   // null means serial
    } m;

    // exec_options::debug >= dbg_route. One line per transform, not per line, for the
    // same reason as nd_runtime_plan::trace. `axes` is the direction's outer-axis set, so
    // the shape a route is reported for is the one that actually ran.
    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how,
                                     const std::vector<nd_axis_state<T>>& axes) const {
        dbg_print("r2c rank=", m.shape.size(), " ", how, " real=", real_size(), " cplx=",
                  cplx_size(), m.pool ? " threaded" : " serial");
        if (level < dbg_shape) return;
        dbg_print_seq("  shape", m.shape);
        for (std::size_t d = 0; d + 1 < m.shape.size(); ++d)
            dbg_print("  axis ", d, " len=", m.shape[d], " ",
                      axes[d].dif ? "col_dif" : axes[d].plan->route_name());
    }
};

template<typename T>
nd_real_plan<T>::nd_real_plan(std::span<const std::size_t> shape, std::size_t nthreads,
                              admiral::effort eff) {
    m.shape.assign(shape.begin(), shape.end());
    const std::size_t n = m.shape.size();
    // Rank 0 has no innermost axis, and a wrapped product would become a buffer
    // size below. Same rejection as nd_runtime_plan.
    if (n == 0 || !extent_product(m.shape))
        throw std::invalid_argument("Plan size must be greater than 0");
    m.inner_len = m.shape.back();
    m.Nh = m.inner_len / 2 + 1;
    m.rp.emplace(m.inner_len, eff);
    // Product of the outer extents (== number of innermost rows).
    m.rows = 1;
    for (std::size_t d = 0; d + 1 < n; ++d) m.rows *= m.shape[d];
    m.total_c = m.rows * m.Nh;
    // Per-axis c2c states (outer axes only, both dirs). `inner` starts at Nh,
    // built innermost-first to match run_outer().
    const std::size_t nouter = n - 1;
    m.fwd_axes.resize(nouter);
    m.inv_axes.resize(nouter);
    std::size_t inner = m.Nh;
    for (std::size_t di = 0; di < nouter; ++di) {
        const std::size_t d = nouter - 1 - di;
        // Outer-axis sub-plans stay 1-thread: the batch loops above them are
        // threadable (units >= 2 for any real shape that executes them), and
        // nesting parallel_for is forbidden.
        m.fwd_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, /*is_forward=*/true,
                                              /*innermost=*/false, 1, eff);
        m.inv_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, /*is_forward=*/false,
                                              /*innermost=*/false, 1, eff);
        inner *= m.shape[d];
    }
    if (nthreads > 1) m.pool = std::make_unique<thread_pool>(nthreads);
}

template<typename T>
void nd_real_plan<T>::forward(const T* in, std::complex<T>* out, const exec_options<T>& opts) const {
    if (opts.debug >= dbg_route) [[unlikely]] trace(opts.debug, "fwd", m.fwd_axes);
    m.rp->r2c(in, out, m.rows, m.pool.get());                      // innermost real axis
    run_outer(out, m.fwd_axes, /*is_forward=*/true, m.pool.get()); // remaining axes (c2c)
    if (opts.fct && *opts.fct != T(1))
        for (std::size_t i = 0; i < m.total_c; ++i) out[i] *= *opts.fct;
}

template<typename T>
void nd_real_plan<T>::inverse(std::complex<T>* spec, T* out, const exec_options<T>& opts) const {
    if (opts.debug >= dbg_route) [[unlikely]] trace(opts.debug, "inv", m.inv_axes);
    run_outer(spec, m.inv_axes, /*is_forward=*/false, m.pool.get()); // outer axes first
    m.rp->c2r(spec, out, m.rows, m.pool.get());                      // then innermost c2r
    if (opts.fct) {
        const std::size_t Nr = real_size();
        const T def = T(1) / static_cast<T>(Nr);
        if (*opts.fct != def) {
            const T s = *opts.fct * static_cast<T>(Nr);
            for (std::size_t i = 0; i < Nr; ++i) out[i] *= s;
        }
    }
}

// Instantiated once in src/inst_real_{f,d}.cpp. Every consumer TU (c_api,
// fftw_api, 4 tests, 2 benchmarks) otherwise rebuilds this whole tree.
extern template class real_adm_plan<float>;
extern template class real_adm_plan<double>;
extern template class nd_real_plan<float>;
extern template class nd_real_plan<double>;

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
