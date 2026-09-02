#pragma once

// Straight-line mixed-radix Cooley-Tukey for one small N, unrolled at compile time. Compare
// Frigo, A Fast Fourier Transform Compiler, PLDI 1999, which generates the same shape of code.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <poet/poet.hpp>
#include "simd.hpp"

#include <admiral/detail/codelet_max.hpp>
#include "cxx_compat.hpp"

#include "butterfly.hpp"
#include "ct_math.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

template<unsigned N, unsigned R, typename T, bool Imag>
ADM_CONSTEVAL std::array<T, (R - 1) * (N / R)> make_twiddle_table() {
    std::array<T, (R - 1) * (N / R)> a{};
    constexpr unsigned M = N / R;
    for (unsigned q = 1; q < R; ++q) {
        for (unsigned j = 0; j < M; ++j) {
            const ct_sincos_t w = ct_sincos_turns(true, q * j, N);
            a[(q - 1) * M + j] = Imag ? static_cast<T>(w.s) : static_cast<T>(w.c);
        }
    }
    return a;
}

template<typename T>
struct yre_sink {
    T* yre;
    T* yim;
    template<typename V>
    ADM_ALWAYS_INLINE void operator()(std::size_t p, V outr, V outi) const {
        if constexpr (std::is_same_v<V, T>) {
            yre[p] = outr;
            yim[p] = outi;
        } else {
            outr.store_unaligned(yre + p);
            outi.store_unaligned(yim + p);
        }
    }
};

template<unsigned R, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk(const T* ADM_RESTRICT twre,
                                  const T* ADM_RESTRICT twim,
                                  const T* yre, const T* yim,
                                  std::size_t M, std::size_t j, Sink&& sink) {
    V tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const V ar = V::load_unaligned(yre + q * M + j);
        const V ai = V::load_unaligned(yim + q * M + j);
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const V wr = V::load_unaligned(twre + (q - 1) * M + j);
            const V wi = V::load_unaligned(twim + (q - 1) * M + j);
            tr[q] = wr * ar - wi * ai;
            ti[q] = wr * ai + wi * ar;
        }
    });

    dif_butterfly<T, R, V>(tr, ti,
        [&](auto L, V outr, V outi) { sink(L * M + j, outr, outi); });
}

template<unsigned R, unsigned N, std::size_t J, typename T, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk_scalar_ct(const T* yre, const T* yim,
                                            Sink&& sink) {
    constexpr std::size_t M = N / R;
    T tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const T ar = yre[q * M + J];
        const T ai = yim[q * M + J];
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const auto [wr, wi] = apply_stage_twiddle<T, N, decltype(q)::value * J, T>(ar, ai);
            tr[q] = wr;
            ti[q] = wi;
        }
    });
    dif_butterfly<T, R, T>(tr, ti,
        [&](auto L, T outr, T outi) { sink(L * M + J, outr, outi); });
}

template<typename T, std::size_t Wt = 2>
[[nodiscard]] ADM_CONSTEVAL std::size_t min_sized_tail_width() {
    if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, Wt>>) return Wt;
    else return min_sized_tail_width<T, Wt * 2>();
}

template<unsigned R, unsigned N, typename T, typename Sink>
ADM_ALWAYS_INLINE void radix_butterfly_ct(T* ADM_RESTRICT yre, T* ADM_RESTRICT yim,
                                          Sink&& sink) {
    constexpr std::size_t M = N / R;
    static constexpr auto twre = make_twiddle_table<N, R, T, false>();
    static constexpr auto twim = make_twiddle_table<N, R, T, true>();
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    constexpr std::size_t U = dif_pass_unroll<R>();
    constexpr std::size_t nfull = M / W;
    if constexpr (nfull > 0) {
        if constexpr (2u * R + 10u <= poet::vector_register_count() + 2u) {
            poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) ADM_LAMBDA_ALWAYS_INLINE {
                bfly_chunk<R, T, batch>(twre.data(), twim.data(), yre, yim, M, b * W, sink);
            });
        } else {
            poet::dynamic_for<U>(std::size_t{0}, nfull, [&](std::size_t b) {
                bfly_chunk<R, T, batch>(twre.data(), twim.data(), yre, yim, M, b * W, sink);
            });
        }
    }
    constexpr std::size_t rem = M - nfull * W;
    poet::static_for<1, detail::bit_width(W)>([&](auto S) {
        constexpr std::size_t Wt = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wt>;
        if constexpr (Wt >= 2 && !std::is_void_v<Vt> && (rem & Wt) != 0) {
            bfly_chunk<R, T, Vt>(twre.data(), twim.data(), yre, yim, M,
                                 nfull * W + (rem & ~(2 * Wt - 1)), sink);
        }
    });
    constexpr std::size_t smask = min_sized_tail_width<T>() - 1;
    constexpr std::size_t nscal = rem & smask;
    constexpr std::size_t jscal = nfull * W + (rem & ~smask);
    poet::static_for<0, nscal>([&](auto I) {
        bfly_chunk_scalar_ct<R, N, jscal + decltype(I)::value, T>(yre, yim, sink);
    });
}

template<unsigned R, unsigned N, typename T>
ADM_ALWAYS_INLINE void radix_butterfly_ct(T* ADM_RESTRICT yre, T* ADM_RESTRICT yim) {
    radix_butterfly_ct<R, N, T>(yre, yim, yre_sink<T>{yre, yim});
}

template<unsigned P, typename T, bool Imag>
ADM_CONSTEVAL std::array<T, P - 1> make_rader_bhat() {
    constexpr unsigned L = P - 1;
    constexpr std::size_t g = ct_primitive_root(P);
    struct cd { double re; double im; };
    const std::size_t ginv = ct_powmod(g, P - 2, P);
    std::array<cd, L> bp{};
    for (unsigned j = 0; j < L; ++j) {
        const std::size_t e = ct_powmod(ginv, j, P);
        const ct_sincos_t w = ct_sincos_turns(true, e, P);
        bp[j] = {w.c, w.s};
    }
    std::array<T, L> out{};
    for (unsigned k = 0; k < L; ++k) {
        double sr = 0.0, si = 0.0;
        for (unsigned j = 0; j < L; ++j) {
            const ct_sincos_t e = ct_sincos_turns(true, j * k % L, L);
            sr += bp[j].re * e.c - bp[j].im * e.s;
            si += bp[j].re * e.s + bp[j].im * e.c;
        }
        out[k] = static_cast<T>((Imag ? si : sr) / L);
    }
    return out;
}

template<typename T, std::size_t R, std::size_t W = 1>
[[nodiscard]] ADM_CONSTEVAL std::size_t cofactor_batch_width() {
    if constexpr (W >= xsimd::batch<T>::size)
        return xsimd::batch<T>::size;
    else if constexpr (W >= R && !std::is_void_v<xsimd::make_sized_batch_t<T, W>>)
        return W;
    else
        return cofactor_batch_width<T, R, W * 2>();
}

template<unsigned R, unsigned N, std::size_t J, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void bfly_chunk_batched_ct(const V* yre, const V* yim,
                                             Sink&& sink) {
    constexpr std::size_t M = N / R;
    V tr[R], ti[R];
    poet::static_for<0, R>([&](const auto q) {
        const V ar = yre[q * M + J];
        const V ai = yim[q * M + J];
        if constexpr (q == 0) {
            tr[0] = ar;
            ti[0] = ai;
        } else {
            const auto [wr, wi] = apply_stage_twiddle<T, N, decltype(q)::value * J, V>(ar, ai);
            tr[q] = wr;
            ti[q] = wi;
        }
    });
    dif_butterfly<T, R, V>(tr, ti,
        [&](auto L, V outr, V outi) { sink(L * M + J, outr, outi); });
}

template<unsigned R, unsigned N, typename T, typename V, typename Sink>
ADM_ALWAYS_INLINE void radix_butterfly_batched_ct(const V* ADM_RESTRICT yre,
                                                  const V* ADM_RESTRICT yim,
                                                  Sink&& sink) {
    constexpr std::size_t M = N / R;
    poet::static_for<0, M>([&](auto J) {
        bfly_chunk_batched_ct<R, N, decltype(J)::value, T, V>(yre, yim, sink);
    });
}

template<typename V>
struct batch_sink {
    V* yre;
    V* yim;
    ADM_ALWAYS_INLINE void operator()(std::size_t p, V outr, V outi) const {
        yre[p] = outr;
        yim[p] = outi;
    }
};

template<unsigned P, typename T, typename V = xsimd::batch<T>>
void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                         V* yre, V* yim);

template<unsigned N, typename T, bool Forward, typename V = xsimd::batch<T>>
struct kernel_batched {
    static_assert(Forward, "inverse routes through the swapped-domain specialization");
    static constexpr unsigned r = codelet_radix(N);
    static constexpr unsigned M = N / r;
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    static constexpr bool flat_leaf =
        N >= 2 && 2u * N <= (((N & (N - 1u)) == 0u)
                                 ? poet::vector_register_count()
                                 : usable_vector_regs(poet::vector_register_count()));

    static void apply(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim) {
        if constexpr (is_rader_prime(N)) {
            rader_apply_batched<N, T, V>(xre, xim, xstride, yre, yim);
            return;
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly_terminal<T, N, V>(tr, ti, [&](auto K, V re, V im) {
                yre[K] = re;
                yim[K] = im;
            });
        } else {
            poet::static_for<0, r>([&](const auto q) {
                kernel_batched<M, T, true, V>::apply(xre + q * xstride, xim + q * xstride,
                                                         xstride * r, yre + q * M, yim + q * M);
            });
            radix_butterfly_batched_ct<r, N, T, V>(yre, yim, batch_sink<V>{yre, yim});
        }
    }

    template<typename Sink>
    static void apply_sink(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim,
                           Sink&& sink) {
        if constexpr (is_rader_prime(N)) {
            rader_apply_batched<N, T, V>(xre, xim, xstride, yre, yim);
            poet::static_for<0, N>([&](auto P) { sink(P, yre[P], yim[P]); });
        } else if constexpr (flat_leaf) {
            V tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly_terminal<T, N, V>(tr, ti,
                [&](auto K, V re, V im) { sink(K, re, im); });
        } else {
            poet::static_for<0, r>([&](const auto q) {
                kernel_batched<M, T, true, V>::apply(xre + q * xstride, xim + q * xstride,
                                                         xstride * r, yre + q * M, yim + q * M);
            });
            radix_butterfly_batched_ct<r, N, T, V>(yre, yim, sink);
        }
    }
};

template<typename T, typename V>
struct kernel_batched<1, T, true, V> {
    static void apply(const V* xre, const V* xim, std::size_t , V* yre, V* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
};

template<unsigned N, typename T, typename V>
struct kernel_batched<N, T, false, V> {
    static void apply(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim) {
        kernel_batched<N, T, true, V>::apply(xim, xre, xstride, yim, yre);
    }
    template<typename Sink>
    static void apply_sink(const V* xre, const V* xim, std::size_t xstride, V* yre, V* yim,
                           Sink&& sink) {
        kernel_batched<N, T, true, V>::apply_sink(xim, xre, xstride, yim, yre,
            [&](std::size_t p, V outr, V outi) { sink(p, outi, outr); });
    }
};

template<unsigned P, typename T>
void rader_apply(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim);

inline constexpr unsigned kNoinlineMinSize = 16;

static_assert(kNoinlineMinSize >= xsimd::batch<float>::size && kNoinlineMinSize <= CODELET_CATALOG_MAX,
              "kNoinlineMinSize must sit between the SIMD width and the catalog max");

[[nodiscard]] ADM_CONSTEVAL bool kernel_should_noinline(std::size_t M) {
    return M >= kNoinlineMinSize
        && 2 * codelet_radix(M) + 10 > poet::vector_register_count();
}

inline constexpr std::size_t kMaskedLoad256MinWork = 27;

template<typename T, unsigned R>
[[nodiscard]] ADM_CONSTEVAL bool cofactor_simd_profitable(std::size_t M) {
    if (is_rader_prime(M)) {
        constexpr std::size_t Wc = cofactor_batch_width<T, R>();
        if constexpr (2u * R <= Wc) return false;
        return true;
    }
    const bool odd_m = M >= 3 && (M % 2 != 0) && !kernel_should_noinline(M);
    const bool pow2_m = (M == 4 || (M == 8 && 2 * M < poet::vector_register_count())) &&
                        cofactor_batch_width<T, R>() == R;
    const bool even_m = (M % 2 == 0) && M >= cofactor_batch_width<T, R>();
    if (!(odd_m || pow2_m || even_m)) return false;
    constexpr std::size_t Wc = cofactor_batch_width<T, R>();
    if (Wc == R) return true;
    if (2u * R <= Wc) return false;
    if (Wc * sizeof(T) <= 16u) return true;
    return R * M >= kMaskedLoad256MinWork;
}

template<typename T>
[[nodiscard]] ADM_CONSTEVAL std::size_t kernel_peel_radix(std::size_t N) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (N == 64 && 8 <= W && cofactor_simd_profitable<T, 8u>(8)) return 8;
    if (N % 8 == 0 && (N / 8) % 2 == 1 && 8 <= W && cofactor_simd_profitable<T, 8u>(N / 8))
        return 8;
    if (N % 4 == 0 && (N / 4) % 2 == 1 && 4 <= W && cofactor_simd_profitable<T, 4u>(N / 4))
        return 4;
    return codelet_radix_for<T>(N);
}

template<unsigned N, typename T>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim);

template<unsigned N, typename T, bool Forward>
struct kernel {
    static_assert(Forward, "inverse routes through the swapped-domain specialization");
    static constexpr unsigned r = kernel_peel_radix<T>(N);
    static constexpr unsigned M = N / r;
    static_assert(r * M == N, "codelet_radix(N) must divide N exactly");

    static constexpr bool flat_leaf =
        N >= 2 && (((N & (N - 1u)) == 0u) ? N <= 8u : N <= 13u)
        && 2u * N <= poet::vector_register_count()
        && !(M > r && r <= xsimd::batch<T>::size && cofactor_simd_profitable<T, r>(M));

    template<typename Sink>
    static void apply_impl(const T* xre, const T* xim, std::size_t xstride,
                           T* yre, T* yim, Sink&& sink) {
        if constexpr (is_rader_prime(N)) {
            rader_apply<N, T>(xre, xim, xstride, yre, yim);
            poet::static_for<0, N>([&](auto P) { sink(P, yre[P], yim[P]); });
            return;
        }
        if constexpr (flat_leaf) {
            T tr[N], ti[N];
            poet::static_for<0, N>([&](auto J) {
                tr[J] = xre[J * xstride];
                ti[J] = xim[J * xstride];
            });
            dif_butterfly<T, N, T>(tr, ti, [&](auto K, T re, T im) { sink(K, re, im); });
            return;
        }
        if constexpr (r <= xsimd::batch<T>::size && cofactor_simd_profitable<T, r>(M)) {
            constexpr std::size_t Wc = cofactor_batch_width<T, r>();
            using V = xsimd::make_sized_batch_t<T, Wc>;
            static_assert(!std::is_void_v<V>, "no SIMD batch wide enough for cofactor r");
            static_assert(Wc >= r, "cofactor batch must hold all r sub-transforms");
            using A = typename V::arch_type;
            if (xstride == 1) {
                constexpr std::size_t Mp =
                    (M % Wc != 0 && M >= Wc) ? ((M / Wc + 1) * Wc) : M;
                V ovr[Mp], ovi[Mp];
                {
                    poet::static_for<0, Mp - M>([&](auto P) {
                        if constexpr (M + P < Mp) {
                            ovr[M + P] = V(T(0));
                            ovi[M + P] = V(T(0));
                        }
                    });
                    if constexpr (Wc == r) {
                        ADM_COMPILER_BARRIER();
                        const bool kb_aligned =
                            (reinterpret_cast<std::uintptr_t>(xre) % alignof(V) == 0) &&
                            (reinterpret_cast<std::uintptr_t>(xim) % alignof(V) == 0);
                        if (kb_aligned) {
                            kernel_batched<M, T, true, V>::apply(
                                reinterpret_cast<const V*>(xre),
                                reinterpret_cast<const V*>(xim), 1, ovr, ovi);
                        } else {
                            V xv[M], iv[M];
                            for (std::size_t j = 0; j < M; ++j) {
                                xv[j] = V::load_unaligned(xre + r * j);
                                iv[j] = V::load_unaligned(xim + r * j);
                            }
                            kernel_batched<M, T, true, V>::apply(xv, iv, 1, ovr, ovi);
                        }
                    } else {
                        V xv[M], iv[M];
                        for (std::size_t j = 0; j < M; ++j) {
                            constexpr auto mask = xsimd::make_batch_bool_constant<T, lane_lt<r>, A>();
                            xv[j] = V::load(xre + r * j, mask, xsimd::unaligned_mode{});
                            iv[j] = V::load(xim + r * j, mask, xsimd::unaligned_mode{});
                        }
                        kernel_batched<M, T, true, V>::apply(xv, iv, 1, ovr, ovi);
                    }
                }
                constexpr std::size_t nft = M / Wc;
                static constexpr auto twre = make_twiddle_table<N, r, T, false>();
                static constexpr auto twim = make_twiddle_table<N, r, T, true>();
                if constexpr (nft > 0) {
                    poet::static_for<0, nft>([&](auto Tt) {
                        constexpr std::size_t k0 = decltype(Tt)::value * Wc;
                        xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                        xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                        V tr[r], ti[r];
                        poet::static_for<0, r>([&](const auto q) {
                            if constexpr (q == 0) {
                                tr[0] = ovr[k0];
                                ti[0] = ovi[k0];
                            } else {
                                const V ar = ovr[k0 + q];
                                const V ai = ovi[k0 + q];
                                const V wr = V::load_unaligned(twre.data() + (q - 1) * M + k0);
                                const V wi = V::load_unaligned(twim.data() + (q - 1) * M + k0);
                                tr[q] = wr * ar - wi * ai;
                                ti[q] = wr * ai + wi * ar;
                            }
                        });
                        dif_butterfly<T, r, V>(tr, ti,
                            [&](auto L, V outr, V outi) { sink(L * M + k0, outr, outi); });
                    });
                }
                if constexpr (M % Wc != 0 && M >= Wc) {
                    constexpr auto smask = xsimd::make_batch_bool_constant<T, lane_lt<M % Wc>, A>();
                    constexpr std::size_t k0 = nft * Wc;
                    xsimd::transpose(&ovr[k0], &ovr[k0] + Wc);
                    xsimd::transpose(&ovi[k0], &ovi[k0] + Wc);
                    poet::static_for<0, r>([&](const auto q) {
                        ovr[k0 + q].store(yre + q * M + k0, smask, xsimd::unaligned_mode{});
                        ovi[k0 + q].store(yim + q * M + k0, smask, xsimd::unaligned_mode{});
                    });
                } else if constexpr (M % Wc != 0) {
                    T sr[Wc], si[Wc];
                    for (std::size_t k = nft * Wc; k < M; ++k) {
                        ovr[k].store_unaligned(sr);
                        ovi[k].store_unaligned(si);
                        poet::static_for<0, r>([&](const auto q) {
                            yre[q * M + k] = sr[q];
                            yim[q * M + k] = si[q];
                        });
                    }
                }
                constexpr std::size_t C = M - nft * Wc;
                poet::static_for<1, detail::bit_width(Wc)>([&](auto S) {
                    constexpr std::size_t Wt = Wc >> S;
                    using Vt = xsimd::make_sized_batch_t<T, Wt>;
                    if constexpr (Wt >= 2 && !std::is_void_v<Vt> && (C & Wt) != 0) {
                        bfly_chunk<r, T, Vt>(twre.data(), twim.data(), yre, yim, M,
                                             nft * Wc + (C & ~(2 * Wt - 1)), sink);
                    }
                });
                constexpr std::size_t smask = min_sized_tail_width<T>() - 1;
                constexpr std::size_t nscal = C & smask;
                poet::static_for<0, nscal>([&](auto I) {
                    constexpr std::size_t J = nft * Wc + (C & ~smask) + decltype(I)::value;
                    bfly_chunk_scalar_ct<r, N, J, T>(yre, yim, sink);
                });
                return;
            }
        }
        poet::static_for<0, r>([&](const auto q) {
            if constexpr (kernel_should_noinline(M)) {
                kernel_apply_boundary<M, T>(xre + q * xstride, xim + q * xstride,
                                                     xstride * r, yre + q * M, yim + q * M);
            } else {
                kernel<M, T, true>::apply(xre + q * xstride, xim + q * xstride,
                                             xstride * r, yre + q * M, yim + q * M);
            }
        });
        radix_butterfly_ct<r, N, T>(yre, yim, sink);
    }

    static void apply(const T* xre, const T* xim, std::size_t xstride,
                      T* yre, T* yim) {
        apply_impl(xre, xim, xstride, yre, yim, yre_sink<T>{yre, yim});
    }

    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t xstride,
                           T* yre, T* yim, Sink&& sink) {
        apply_impl(xre, xim, xstride, yre, yim,
                   static_cast<Sink&&>(sink));
    }
};

template<typename T>
struct kernel<1, T, true> {
    static void apply(const T* xre, const T* xim, std::size_t ,
                      T* yre, T* yim) {
        yre[0] = xre[0];
        yim[0] = xim[0];
    }
    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t ,
                           T* , T* , Sink&& sink) {
        sink(std::size_t{0}, xre[0], xim[0]);
    }
};

template<unsigned N, typename T>
struct kernel<N, T, false> {
    static void apply(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim) {
        kernel<N, T, true>::apply(xim, xre, xstride, yim, yre);
    }
    template<typename Sink>
    static void apply_sink(const T* xre, const T* xim, std::size_t xstride, T* yre, T* yim,
                           Sink&& sink) {
        kernel<N, T, true>::apply_sink(xim, xre, xstride, yim, yre,
            [&](std::size_t p, auto outr, auto outi) { sink(p, outi, outr); });
    }
};

template<unsigned N, typename T>
ADM_NOINLINE void kernel_apply_boundary(const T* xre, const T* xim, std::size_t xstride,
                                        T* yre, T* yim) {
    kernel<N, T, true>::apply(xre, xim, xstride, yre, yim);
}

template<unsigned P, unsigned L = P - 1>
ADM_CONSTEVAL std::array<std::size_t, L> rader_gpow_table() {
    constexpr std::size_t g = ct_primitive_root(P);
    std::array<std::size_t, L> t{};
    std::size_t e = 1;
    for (unsigned q = 0; q < L; ++q) {
        t[q] = e;
        e = e * g % P;
    }
    return t;
}

template<unsigned P, unsigned L = P - 1>
ADM_CONSTEVAL std::array<std::size_t, L> rader_ginvpow_table() {
    constexpr std::size_t g = ct_primitive_root(P);
    const std::size_t gi = ct_powmod(g, P - 2, P);
    std::array<std::size_t, L> t{};
    std::size_t e = 1;
    for (unsigned m = 0; m < L; ++m) {
        t[m] = e;
        e = e * gi % P;
    }
    return t;
}

template<unsigned P, typename T>
ADM_NOINLINE void rader_apply(const T* xre, const T* xim, std::size_t xstride,
                              T* yre, T* yim) {
    constexpr unsigned L = P - 1;

    static constexpr std::array<std::size_t, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<std::size_t, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, T, true>();

    const T x0r = xre[0];
    const T x0i = xim[0];
    T are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = gpow[q] * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
    }

    T Are[L], Aim[L];
    kernel<L, T, true>::apply(are, aim, 1, Are, Aim);

    T Pre[L], Pim[L];
    using batch = xsimd::batch<T>;
    constexpr std::size_t Wd = batch::size;
    constexpr std::size_t Lv = L - L % Wd;
    for (std::size_t k = 0; k < Lv; k += Wd) {
        const batch ar = batch::load_unaligned(&Are[k]);
        const batch ai = batch::load_unaligned(&Aim[k]);
        const batch br = batch::load_unaligned(&Bre[k]);
        const batch bi = batch::load_unaligned(&Bim[k]);
        (ar * br - ai * bi).store_unaligned(&Pre[k]);
        (ar * bi + ai * br).store_unaligned(&Pim[k]);
    }
    if constexpr (L % Wd != 0) {
        for (std::size_t k = Lv; k < L; ++k) {
            Pre[k] = Are[k] * Bre[k] - Aim[k] * Bim[k];
            Pim[k] = Are[k] * Bim[k] + Aim[k] * Bre[k];
        }
    }

    T cre[L], cim[L];
    kernel<L, T, false>::apply(Pre, Pim, 1, cre, cim);

    yre[0] = x0r + Are[0];
    yim[0] = x0i + Aim[0];
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m];
        yim[ginvpow[m]] = x0i + cim[m];
    }
}

template<unsigned P, typename T, typename V>
ADM_NOINLINE void rader_apply_batched(const V* xre, const V* xim, std::size_t xstride,
                                      V* yre, V* yim) {
    constexpr unsigned L = P - 1;

    static constexpr std::array<std::size_t, L> gpow = rader_gpow_table<P>();
    static constexpr std::array<std::size_t, L> ginvpow = rader_ginvpow_table<P>();
    static constexpr std::array<T, L> Bre = make_rader_bhat<P, T, false>();
    static constexpr std::array<T, L> Bim = make_rader_bhat<P, T, true>();

    const V x0r = xre[0];
    const V x0i = xim[0];
    V are[L], aim[L];
    for (unsigned q = 0; q < L; ++q) {
        const std::size_t idx = gpow[q] * xstride;
        are[q] = xre[idx];
        aim[q] = xim[idx];
    }

    V Are[L], Aim[L];
    kernel_batched<L, T, true, V>::apply(are, aim, 1, Are, Aim);

    V Pre[L], Pim[L];
    for (unsigned k = 0; k < L; ++k) {
        const V br(Bre[k]);
        const V bi(Bim[k]);
        Pre[k] = Are[k] * br - Aim[k] * bi;
        Pim[k] = Are[k] * bi + Aim[k] * br;
    }

    V cre[L], cim[L];
    kernel_batched<L, T, false, V>::apply(Pre, Pim, 1, cre, cim);

    yre[0] = x0r + Are[0];
    yim[0] = x0i + Aim[0];
    for (unsigned m = 0; m < L; ++m) {
        yre[ginvpow[m]] = x0r + cre[m];
        yim[ginvpow[m]] = x0i + cim[m];
    }
}

}
}

#include "undef_macros.hpp"
