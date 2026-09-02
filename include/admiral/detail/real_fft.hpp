#pragma once

// Real input of length N through one complex FFT of length N/2: pack even and odd samples as
// real and imaginary, then untangle. Sorensen et al., IEEE Trans. ASSP 35 (1987) 849.

#include <complex>
#include <concepts>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <admiral/errors.hpp>

#include "cxx_compat.hpp"
#include "nd_plan.hpp"
#include "plan.hpp"
#include "portable_trig.hpp"
#include "real_recombine.hpp"
#include "twiddles.hpp"
#include "vecpass.hpp"
#include <poet/poet.hpp>
#include "simd.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

template<typename T>
class real_adm_plan {
public:
    explicit real_adm_plan(std::size_t N, admiral::effort eff = admiral::effort::estimate);

    void r2c(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool = nullptr) const;

    void c2r(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool = nullptr) const;

private:
    using V = xsimd::batch<T>;
    static constexpr std::size_t W = V::size;

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
        r2c_even_scalar(in + done * N_, out + done * Nh_, rows - done);
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

#if ADM_CXX20
    template<typename Body>
        requires std::invocable<Body&, std::size_t, V*>
#else
    template<typename Body,
             std::enable_if_t<std::is_invocable_v<Body&, std::size_t, V*>, int> = 0>
#endif
    void run_tiles(thread_pool* pool, std::size_t ntiles, std::size_t total_elems, Body&& body) const {
        const bool par = pool && ntiles >= 2 * pool->size() && total_elems >= kThreadMinElems;
        if (par) {
            pool->parallel_for(ntiles, [&](std::size_t b, std::size_t e, std::size_t) {
                auto sc = detail::make_unique_for_overwrite<V[]>(4 * M_);
                for (std::size_t t = b; t < e; ++t) body(t, sc.get());
            });
        } else {
            auto sc = detail::make_unique_for_overwrite<V[]>(4 * M_);
            for (std::size_t t = 0; t < ntiles; ++t) body(t, sc.get());
        }
    }

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

        constexpr std::size_t H = W / 2;
        std::size_t j = 0;
        for (; j + H <= M; j += H) {
            V t[W];
            poet::static_for<0, W>([&](auto l) { t[l] = V::load_unaligned(in + l * N_ + 2 * j); });
            xsimd::transpose(t, t + W);
            poet::static_for<0, H>([&](auto m) { cur_re[j + m] = t[2 * m]; cur_im[j + m] = t[2 * m + 1]; });
        }
        for (; j < M; ++j) {
            alignas(V) T zr[W], zi[W];
            for (std::size_t l = 0; l < W; ++l) {
                const T* xr = in + l * N_;
                zr[l] = xr[2 * j]; zi[l] = xr[2 * j + 1];
            }
            cur_re[j] = V::load_aligned(zr);
            cur_im[j] = V::load_aligned(zi);
        }
        const auto G = vp::multipass_run<T, true, V>(tab_, cur_re, cur_im, nxt_re, nxt_im);
        const V* Gre = G.first;
        const V* Gim = G.second;

        const V hv(T(0.5));
        std::size_t k = 0;
        for (; k + H <= M; k += H) {
            V t[W];
            poet::static_for<0, H>([&](auto m) {
                const std::size_t kk = k + m, kb = (kk == 0) ? 0 : (M - kk);
                const V zkr = Gre[kk], zki = Gim[kk];
                const V zcr = Gre[kb], zci = -Gim[kb];
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
        for (; k <= half; ++k) {
            const std::size_t ka = (k == M) ? 0 : k, kb = (ka == 0) ? 0 : (M - ka);
            const V zkr = Gre[ka], zki = Gim[ka];
            const V zcr = Gre[kb], zci = -Gim[kb];
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

        const V hv(T(0.5));
        constexpr std::size_t H = W / 2;
        std::size_t k = 0;
        for (; k + H <= M; k += H) {
            V tk[W], tm[W];
            const std::size_t mb = M - k - H + 1;
            poet::static_for<0, W>([&](auto l) {
                tk[l] = V::load_unaligned(reinterpret_cast<const T*>(in + l * Nh_ + k));
                tm[l] = V::load_unaligned(reinterpret_cast<const T*>(in + l * Nh_ + mb));
            });
            xsimd::transpose(tk, tk + W);
            xsimd::transpose(tm, tm + W);
            poet::static_for<0, H>([&](auto m) {
                const std::size_t kk = k + m, p = H - 1 - m;
                const V Xkr = tk[2 * m], Xki = tk[2 * m + 1];
                const V Xmr = tm[2 * p], Xmi = tm[2 * p + 1];
                const V twr(tw_[kk].real()), twi(tw_[kk].imag());
                const auto [Zr, Zi] = c2r_recombine(Xkr, Xki, Xmr, Xmi, hv, twr, twi);
                cur_re[kk] = Zr;
                cur_im[kk] = Zi;
            });
        }
        for (; k < M; ++k) {
            alignas(V) T xkr[W], xki[W], xmr[W], xmi[W];
            for (std::size_t l = 0; l < W; ++l) {
                const std::complex<T>* Xr = in + l * Nh_;
                xkr[l] = Xr[k].real();     xki[l] = Xr[k].imag();
                xmr[l] = Xr[M - k].real(); xmi[l] = Xr[M - k].imag();
            }
            const V Xkr = V::load_aligned(xkr), Xki = V::load_aligned(xki);
            const V Xmr = V::load_aligned(xmr), Xmi = V::load_aligned(xmi);
            const V twr(tw_[k].real()), twi(tw_[k].imag());
            const auto [Zr, Zi] = c2r_recombine(Xkr, Xki, Xmr, Xmi, hv, twr, twi);
            cur_re[k] = Zr;
            cur_im[k] = Zi;
        }
        const auto G = vp::multipass_run<T, false, V>(tab_, cur_re, cur_im, nxt_re, nxt_im);
        const V* Gre = G.first;
        const V* Gim = G.second;

        const V invM(T(1) / static_cast<T>(M));
        std::size_t j = 0;
        for (; j + H <= M; j += H) {
            V t[W];
            poet::static_for<0, H>([&](auto m) { t[2 * m] = Gre[j + m] * invM; t[2 * m + 1] = Gim[j + m] * invM; });
            xsimd::transpose(t, t + W);
            poet::static_for<0, W>([&](auto l) { t[l].store_unaligned(out + l * N_ + 2 * j); });
        }
        for (; j < M; ++j) {
            alignas(V) T zr[W], zi[W];
            (Gre[j] * invM).store_aligned(zr);
            (Gim[j] * invM).store_aligned(zi);
            for (std::size_t l = 0; l < W; ++l) {
                T* xr = out + l * N_;
                xr[2 * j] = zr[l]; xr[2 * j + 1] = zi[l];
            }
        }
    }

    void r2c_even_scalar(const T* in, std::complex<T>* out, std::size_t rows) const {
        const std::size_t M = M_, half = N_ / 2;
        const auto z = detail::make_unique_for_overwrite<std::complex<T>[]>(M == 0 ? 1 : M);
        for (std::size_t r = 0; r < rows; ++r) {
            const T* xr = in + r * N_;
            for (std::size_t j = 0; j < M; ++j) z[j] = std::complex<T>(xr[2 * j], xr[2 * j + 1]);
            fwd_.execute(span<std::complex<T>>(z.get(), M));
            std::complex<T>* Xr = out + r * Nh_;
            for (std::size_t k = 0; k <= half; ++k) Xr[k] = r2c_even_bin(z.get(), tw_[k], M, k);
        }
    }

    void c2r_even_scalar(const std::complex<T>* in, T* out, std::size_t rows) const {
        const std::size_t M = M_;
        const auto Z = detail::make_unique_for_overwrite<std::complex<T>[]>(M == 0 ? 1 : M);
        for (std::size_t r = 0; r < rows; ++r) {
            const std::complex<T>* Xr = in + r * Nh_;
            for (std::size_t k = 0; k < M; ++k) Z[k] = c2r_even_bin(Xr, tw_[k], M, k);
            inv_.execute(span<std::complex<T>>(Z.get(), M));
            T* xr = out + r * N_;
            for (std::size_t j = 0; j < M; ++j) { xr[2 * j] = Z[j].real(); xr[2 * j + 1] = Z[j].imag(); }
        }
    }

    void r2c_odd(const T* in, std::complex<T>* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            const auto c = detail::make_unique_for_overwrite<std::complex<T>[]>(N_);
            for (std::size_t r = b; r < e; ++r) {
                const T* xr = in + r * N_;
                for (std::size_t i = 0; i < N_; ++i) c[i] = std::complex<T>(xr[i], T(0));
                fwd_.execute(span<std::complex<T>>(c.get(), N_));
                std::complex<T>* Xr = out + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) Xr[k] = c[k];
            }
        });
    }

    void c2r_odd(const std::complex<T>* in, T* out, std::size_t rows, thread_pool* pool) const {
        parallel_for(pool, rows, rows * N_, [&](std::size_t b, std::size_t e, std::size_t) {
            const auto c = detail::make_unique_for_overwrite<std::complex<T>[]>(N_);
            for (std::size_t r = b; r < e; ++r) {
                const std::complex<T>* Xr = in + r * Nh_;
                for (std::size_t k = 0; k < Nh_; ++k) c[k] = Xr[k];
                for (std::size_t k = Nh_; k < N_; ++k) c[k] = std::conj(Xr[N_ - k]);
                inv_.execute(span<std::complex<T>>(c.get(), N_));
                T* xr = out + r * N_;
                for (std::size_t i = 0; i < N_; ++i) xr[i] = c[i].real();
            }
        });
    }

    std::size_t N_;
    bool even_;
    std::size_t M_, Nh_;
    plan_impl<T> fwd_, inv_;
    std::vector<std::complex<T>> tw_;
    bool batched_ = false;
    vp::multipass_tables<T> tab_;
};

template<typename T>
real_adm_plan<T>::real_adm_plan(std::size_t N, admiral::effort eff)
    : N_(N), even_(N % 2 == 0), M_(even_ ? N / 2 : N),
      Nh_(N / 2 + 1),
      fwd_(M_, true, 1, nullptr, eff),
      inv_(M_, false, 1, nullptr, eff) {
    if (even_) {
        const std::size_t half = N_ / 2;
        tw_.resize(half + 1);
        for (std::size_t k = 0; k <= half; ++k) {
            const auto [sn, cs] = portable_trig::sincos_turns<true>(k, N_);
            tw_[k] = std::complex<T>(static_cast<T>(cs), static_cast<T>(sn));
        }
        batched_ = M_ >= 2 && multipass_supported(M_);
        if (batched_) tab_.build(M_);
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

template<typename T>
class nd_real_plan {
public:
    explicit nd_real_plan(span<const std::size_t> shape, std::size_t nthreads = 1,
                          admiral::effort eff = admiral::effort::estimate);

    [[nodiscard]] std::size_t cplx_size() const noexcept { return m.total_c; }
    [[nodiscard]] std::size_t real_size() const noexcept { return m.rows * m.inner_len; }

    void forward(const T* in, std::complex<T>* out, const exec_options<T>& opts = {}) const;

    void inverse(std::complex<T>* spec, T* out, const exec_options<T>& opts = {}) const;

private:
    void run_outer(std::complex<T>* data, const std::vector<nd_axis_state<T>>& axes,
                   bool is_forward, thread_pool* pool) const {
        const std::size_t nouter = axes.size();
        std::size_t inner = m.Nh;
        for (std::size_t di = 0; di < nouter; ++di) {
            const std::size_t d = nouter - 1 - di;
            nd_apply_axis<T>(data, m.total_c, m.shape[d], inner,
                             false, is_forward, axes[d], std::nullopt, pool);
            inner *= m.shape[d];
        }
    }

    struct M {
        std::vector<std::size_t> shape;
        std::size_t inner_len;
        std::size_t Nh;
        std::size_t rows;
        std::size_t total_c;
        // Engaged for the whole lifetime: the ctor emplaces it or throws. Each `rp->` below
        // carries a NOLINT, because a per-function analysis cannot see across a constructor.
        std::optional<real_adm_plan<T>> rp;
        std::vector<nd_axis_state<T>> fwd_axes, inv_axes;
        std::unique_ptr<thread_pool> pool;
    } m;

    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how,
                                     const std::vector<nd_axis_state<T>>& axes) const {
        dbg_print("r2c rank=", m.shape.size(), " ", how, " real=", real_size(), " cplx=",
                  cplx_size(), m.pool ? " threaded" : " serial");
        if (level < dbg_shape) return;
        dbg_print_seq("  shape", m.shape);
        for (std::size_t d = 0; d + 1 < m.shape.size(); ++d) {
            const nd_axis_state<T>& ax = axes[d];
            dbg_print("  axis ", d, " len=", m.shape[d], " ",
                      ax.plan ? ax.plan->route_name() : "col_dif");
        }
    }
};

template<typename T>
nd_real_plan<T>::nd_real_plan(span<const std::size_t> shape, std::size_t nthreads,
                              admiral::effort eff) {
    m.shape.assign(shape.begin(), shape.end());
    const std::size_t n = m.shape.size();
    if (n == 0 || !extent_product(m.shape))
        throw size_error("Plan size must be greater than 0");
    m.inner_len = m.shape.back();
    m.Nh = m.inner_len / 2 + 1;
    m.rp.emplace(m.inner_len, eff);
    m.rows = 1;
    for (std::size_t d = 0; d + 1 < n; ++d) m.rows *= m.shape[d];
    m.total_c = m.rows * m.Nh;
    const std::size_t nouter = n - 1;
    m.fwd_axes.resize(nouter);
    m.inv_axes.resize(nouter);
    std::size_t inner = m.Nh;
    for (std::size_t di = 0; di < nouter; ++di) {
        const std::size_t d = nouter - 1 - di;
        m.fwd_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, true,
                                              false, 1, eff);
        m.inv_axes[d] = make_nd_axis_state<T>(m.shape[d], inner, false,
                                              false, 1, eff);
        inner *= m.shape[d];
    }
    if (nthreads == 0) {
        if (n >= 2 && m.rows >= 2 && m.total_c >= kThreadMinElems) {
            std::size_t dispatches = 1;
            double work_cyc = double(m.rows) * line_work_cyc<T>(m.Nh);
            for (std::size_t d = 0; d + 1 < n; ++d) {
                if (m.shape[d] <= 1) continue;
                ++dispatches;
                work_cyc += double(m.total_c / m.shape[d]) * line_work_cyc<T>(m.shape[d]);
            }
            nthreads = resolve_nthreads(0, m.total_c, dispatches, work_cyc / core_cyc_per_ns(),
                                        n >= 3 ? 2 : 1);
        } else {
            nthreads = 1;
        }
    }
    if (nthreads > 1) m.pool = std::make_unique<thread_pool>(nthreads);
}

template<typename T>
void nd_real_plan<T>::forward(const T* in, std::complex<T>* out, const exec_options<T>& opts) const {
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "fwd", m.fwd_axes);
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    m.rp->r2c(in, out, m.rows, m.pool.get());
    run_outer(out, m.fwd_axes, true, m.pool.get());
    if (opts.fct && *opts.fct != T(1))
        for (std::size_t i = 0; i < m.total_c; ++i) out[i] *= *opts.fct;
}

template<typename T>
void nd_real_plan<T>::inverse(std::complex<T>* spec, T* out, const exec_options<T>& opts) const {
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "inv", m.inv_axes);
    run_outer(spec, m.inv_axes, false, m.pool.get());
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    m.rp->c2r(spec, out, m.rows, m.pool.get());
    if (opts.fct) {
        const std::size_t Nr = real_size();
        const T def = T(1) / static_cast<T>(Nr);
        if (*opts.fct != def) {
            const T s = *opts.fct * static_cast<T>(Nr);
            for (std::size_t i = 0; i < Nr; ++i) out[i] *= s;
        }
    }
}

extern template class real_adm_plan<float>;
extern template class real_adm_plan<double>;
extern template class nd_real_plan<float>;
extern template class nd_real_plan<double>;

}
}

#include "undef_macros.hpp"
