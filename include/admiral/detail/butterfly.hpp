#pragma once

// Cooley-Tukey decimation in frequency: Gentleman and Sande, AFIPS Fall Joint Comp. Conf. 1966.
// Odd radices use Singleton's sum/difference DFT, IEEE Trans. Audio Electroacoust. 17 (1969) 93.

#include <complex>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"

#include "ct_math.hpp"
#include "simd_swizzle.hpp"

#include "macros.hpp"

namespace admiral {
namespace detail {

template<typename T, std::size_t IP, typename V, typename Emit>
ADM_ALWAYS_INLINE void radix_sym_dft(const V (&xr)[IP],
                                     const V (&xi)[IP],
                                     Emit&& emit) {
    static_assert(IP % 2 == 1 && IP >= 3, "symmetric DFT is for odd radix >= 3");
    constexpr std::size_t H = (IP - 1) / 2;
    constexpr std::size_t HA = H + 1;
    V ar[HA], ai[HA], dr[HA], di[HA];
    poet::static_for<1, H + 1>([&](const auto m) {
        constexpr std::size_t mc = IP - m;
        ar[m] = xr[m] + xr[mc];
        ai[m] = xi[m] + xi[mc];
        dr[m] = xr[m] - xr[mc];
        di[m] = xi[m] - xi[mc];
    });
    {
        V sr = xr[0], si = xi[0];
        poet::static_for<1, H + 1>([&](const auto m) {
            sr = sr + ar[m];
            si = si + ai[m];
        });
        emit(IC<0>{}, sr, si);
    }
    poet::static_for<1, H + 1>([&](const auto k) {
        V PR = xr[0], PI = xi[0], QR = V(T(0)), QI = V(T(0));
        poet::static_for<1, H + 1>([&](const auto m) {
            constexpr auto w =
                ct_sincos_turns<ct_real_t<T>>(false, IC<(m * k) % IP>{}, IP);
            PR = piece_fma(V(static_cast<T>(w.c)), ar[m], PR);
            PI = piece_fma(V(static_cast<T>(w.c)), ai[m], PI);
            QR = piece_fma(V(static_cast<T>(w.s)), di[m], QR);
            QI = piece_fma(V(static_cast<T>(w.s)), dr[m], QI);
        });
        emit(IC<k>{}, PR + QR, PI - QI);
        emit(IC<IP - k>{}, PR - QR, PI + QI);
    });
}

template<std::size_t N1, std::size_t N2, std::size_t K1, std::size_t K2>
inline constexpr std::size_t crt_index = [] {
    for (std::size_t k = 0; k < N1 * N2; ++k)
        if (k % N1 == K1 && k % N2 == K2) return k;
    return N1 * N2;
}();

template<typename T, std::size_t IP, typename V, typename Emit>
ADM_ALWAYS_INLINE void pow2_dif_butterfly(const V (&xr)[IP], const V (&xi)[IP], Emit&& emit);

template<typename T, std::size_t N, typename V, typename Emit>
ADM_ALWAYS_INLINE void sub_dft(const V (&xr)[N], const V (&xi)[N], Emit&& emit) {
    if constexpr (detail::has_single_bit(N))
        pow2_dif_butterfly<T, N, V>(xr, xi, std::forward<Emit>(emit));
    else
        radix_sym_dft<T, N>(xr, xi, std::forward<Emit>(emit));
}

template<typename T, std::size_t N1, std::size_t N2, typename V, typename Emit>
ADM_ALWAYS_INLINE void pfa_dif_butterfly(const V (&tr)[N1 * N2],
                                         const V (&ti)[N1 * N2],
                                         Emit&& emit) {
    constexpr std::size_t IP = N1 * N2;
    static_assert(std::gcd(N1, N2) == 1, "PFA requires coprime factors");

    V ar[N1][N2], ai[N1][N2];
    poet::static_for<0, N2>([&](const auto n2) {
        V br[N1], bi[N1];
        poet::static_for<0, N1>([&](const auto n1) {
            constexpr std::size_t src = IC<(n1 * N2 + n2 * N1) % IP>{};
            br[n1] = tr[src];
            bi[n1] = ti[src];
        });
        sub_dft<T, N1>(br, bi, [&](auto K1, V yr, V yi) {
            ar[K1][n2] = yr;
            ai[K1][n2] = yi;
        });
    });

    poet::static_for<0, N1>([&](const auto k1) {
        sub_dft<T, N2>(ar[k1], ai[k1], [&](const auto k2, V yr, V yi) {
            constexpr auto out = crt_index<N1, N2, k1, k2>;
            static_assert(out < IP, "CRT index must exist for coprime N1,N2");
            emit(IC<out>{}, yr, yi);
        });
    });
}

[[nodiscard]] constexpr std::size_t sym_dft_ops(std::size_t n) noexcept {
    const std::size_t h = (n - 1) / 2;
    return 4 * h * h + 10 * h;
}

[[nodiscard]] constexpr std::size_t ct_dft_ops(std::size_t n1, std::size_t n2) noexcept {
    return n2 * sym_dft_ops(n1) + n1 * sym_dft_ops(n2) + 6 * (n1 - 1) * (n2 - 1);
}

[[nodiscard]] constexpr std::pair<std::size_t, std::size_t> odd_ct_split(std::size_t n) noexcept {
    if (n % 2 == 0 || coprime_split(n).first != 0) return {0, 0};
    for (std::size_t p = 3; p * p <= n; p += 2)
        if (n % p == 0 && ct_dft_ops(p, n / p) < sym_dft_ops(n)) return {p, n / p};
    return {0, 0};
}

template<typename T, std::size_t IP, std::size_t N, typename V>
[[nodiscard]] ADM_ALWAYS_INLINE std::pair<V, V> apply_stage_twiddle(V fr, V fi) {
    constexpr auto w = ct_sincos_turns<ct_real_t<T>>(true, N, IP);
    // The arm is picked by the integer root-form descriptor, not by FP tests on the folded
    // pair; the lattice (test_ct_root) proves the two agree on every shipped denominator.
    constexpr root_form form = ct_root<N, IP, true>();
    if constexpr (form == root_form::one) {
        return {fr, fi};
    } else if constexpr (form == root_form::neg_one) {
        return {-fr, -fi};
    } else if constexpr (form == root_form::neg_i) {
        return {fi, -fr};
    } else if constexpr (form == root_form::pos_i) {
        return {-fi, fr};
    } else if constexpr (form == root_form::diag) {
        const V c(static_cast<T>(w.c));
        return {c * (fr - fi), c * (fi + fr)};
    } else if constexpr (form == root_form::anti_diag) {
        const V c(static_cast<T>(w.c));
        return {c * (fr + fi), c * (fi - fr)};
    } else {
        const V c(static_cast<T>(w.c)), s(static_cast<T>(w.s));
        return {c * fr - s * fi, c * fi + s * fr};
    }
}

template<typename T, std::size_t IP, typename V, typename Emit>
ADM_ALWAYS_INLINE void pow2_dif_butterfly(const V (&xr)[IP],
                                          const V (&xi)[IP],
                                          Emit&& emit) {
    static_assert(IP >= 2 && detail::has_single_bit(IP), "pow2_dif_butterfly: IP must be a power of two >= 2");
    if constexpr (IP == 2) {
        emit(IC<0>{}, xr[0] + xr[1], xi[0] + xi[1]);
        emit(IC<1>{}, xr[0] - xr[1], xi[0] - xi[1]);
    } else {
        constexpr std::size_t H = IP / 2;
        {
            V er[H], ei[H];
            poet::static_for<0, H>([&](const auto n) {
                er[n] = xr[n] + xr[n + H];
                ei[n] = xi[n] + xi[n + H];
            });
            pow2_dif_butterfly<T, H, V>(er, ei, [&](auto Kc, V yr, V yi) {
                emit(IC<2 * Kc>{}, yr, yi);
            });
        }
        {
            V fr[H], fi[H];
            poet::static_for<0, H>([&](const auto n) {
                auto [tr, ti] = apply_stage_twiddle<T, IP, n, V>(
                    xr[n] - xr[n + H], xi[n] - xi[n + H]);
                fr[n] = tr;
                fi[n] = ti;
            });
            pow2_dif_butterfly<T, H, V>(fr, fi, [&](auto Kc, V yr, V yi) {
                emit(IC<2 * Kc + 1>{}, yr, yi);
            });
        }
    }
}

template<typename T, std::size_t N1, std::size_t N2, typename V, typename Emit>
ADM_ALWAYS_INLINE void ct_dif_butterfly(const V (&tr)[N1 * N2],
                                        const V (&ti)[N1 * N2],
                                        Emit&& emit) {
    constexpr std::size_t IP = N1 * N2;
    static_assert(N1 % 2 == 1 && N2 % 2 == 1 && N1 >= 3 && N2 >= 3,
                  "ct_dif_butterfly: both factors must be odd radices >= 3");

    V ar[N1][N2], ai[N1][N2];
    poet::static_for<0, N2>([&](const auto n2) {
        V br[N1], bi[N1];
        poet::static_for<0, N1>([&](const auto n1) {
            br[n1] = tr[n1 * N2 + n2];
            bi[n1] = ti[n1 * N2 + n2];
        });
        radix_sym_dft<T, N1>(br, bi, [&](const auto r, V yr, V yi) {
            const auto [fr, fi] = apply_stage_twiddle<T, IP, r * n2, V>(yr, yi);
            ar[r][n2] = fr;
            ai[r][n2] = fi;
        });
    });

    poet::static_for<0, N1>([&](const auto r) {
        radix_sym_dft<T, N2>(ar[r], ai[r], [&](const auto k2, V yr, V yi) {
            emit(IC<k2 * N1 + r>{}, yr, yi);
        });
    });
}

template<std::size_t IP>
inline constexpr bool dif_butterfly_wants_reload =
    butterfly_wants_reload(IP, poet::vector_register_count());

template<typename T, std::size_t IP, typename V, typename Emit>
ADM_ALWAYS_INLINE void dif_butterfly(const V (&tr)[IP],
                                     const V (&ti)[IP],
                                     Emit&& emit) {
    constexpr auto ct = odd_ct_split(IP);
    constexpr auto pf = coprime_split(IP);
    if constexpr (ct.first != 0) {
        ct_dif_butterfly<T, ct.first, ct.second>(tr, ti, std::forward<Emit>(emit));
    } else if constexpr (IP % 2 == 1 && IP >= 3) {
        radix_sym_dft<T, IP>(tr, ti, std::forward<Emit>(emit));
    } else if constexpr (IP % 2 == 0 && pf.first != 0) {
        pfa_dif_butterfly<T, pf.first, pf.second>(tr, ti, std::forward<Emit>(emit));
    } else if constexpr (IP >= 4 && detail::has_single_bit(IP)) {
        pow2_dif_butterfly<T, IP>(tr, ti, std::forward<Emit>(emit));
    } else {
        poet::static_for<0, IP>([&](const auto k) {
            V sr = tr[0], si = ti[0];
            poet::static_for<1, IP>([&](const auto jj) {
                constexpr auto w = ct_sincos_turns<ct_real_t<T>>(true, IC<jj * k>{}, IP);
                sr = piece_fnma(V(static_cast<T>(w.s)), ti[jj],
                                piece_fma(V(static_cast<T>(w.c)), tr[jj], sr));
                si = piece_fma(V(static_cast<T>(w.s)), tr[jj],
                               piece_fma(V(static_cast<T>(w.c)), ti[jj], si));
            });
            emit(IC<k>{}, sr, si);
        });
    }
}

template<typename T, std::size_t IP, typename V, typename Emit>
ADM_ALWAYS_INLINE void dif_butterfly_terminal(const V (&tr)[IP],
                                              const V (&ti)[IP],
                                              Emit&& emit) {
    constexpr auto split = coprime_split(IP);
    if constexpr (IP % 2 == 1 && split.first != 0) {
        pfa_dif_butterfly<T, split.first, split.second>(tr, ti,
                                                                 std::forward<Emit>(emit));
    } else {
        dif_butterfly<T, IP>(tr, ti, std::forward<Emit>(emit));
    }
}

// Gentleman-Sande outer split of one pow2 radix, IP = N1 * N2, staged through an L1 scratch.
// The monolithic radix keeps 2*IP vector registers live, so the whole set spills at IP >= 16.
// Stage A runs N2 radix-N1 butterflies over j = n + N2*m and writes a[r][n] = A_r(n)*w_IP^(n*r);
// stage B reads one r row back and runs a radix-N2 butterfly, emitting k = r + N1*k2.
// Peak live is 2*N2 + O(1) registers, and the scratch is IP*W elements per plane.
template<std::size_t IP>
[[nodiscard]] ADM_CONSTEVAL std::size_t staged_dif_n2() {
    return IP / 4u <= 8u ? IP / 4u : 8u;
}

// Stage B of the outer split: one radix-N2 butterfly over the scratch row.
template<typename T, std::size_t N2, std::size_t Stride, typename V, typename Emit>
ADM_ALWAYS_INLINE void staged_dif_stage_b(const T* ar, const T* ai, Emit&& emit) {
    constexpr std::size_t W = V::size;
    V cr[N2], ci[N2];
    poet::static_for<0, N2>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
        cr[n] = V::load_aligned(ar + n * Stride * W);
        ci[n] = V::load_aligned(ai + n * Stride * W);
    });
    sub_dft<T, N2, V>(cr, ci, std::forward<Emit>(emit));
}

template<typename T, std::size_t IP, typename V, typename Load, typename Emit>
ADM_ALWAYS_INLINE void staged_dif_butterfly(Load&& load, Emit&& emit) {
    constexpr std::size_t N2 = staged_dif_n2<IP>();
    constexpr std::size_t N1 = IP / N2;
    constexpr std::size_t W = V::size;
    static_assert(N1 * N2 == IP && N1 >= 2 && N2 >= 2, "staged split must factor IP");
    alignas(V::arch_type::alignment()) T ar[IP * W];
    alignas(V::arch_type::alignment()) T ai[IP * W];
    poet::static_for<0, N2>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
        V br[N1], bi[N1];
        poet::static_for<0, N1>([&](const auto m) ADM_LAMBDA_ALWAYS_INLINE {
            load(IC<n + N2 * m>{}, br[m], bi[m]);
        });
        sub_dft<T, N1, V>(br, bi, [&](const auto r, V yr, V yi) ADM_LAMBDA_ALWAYS_INLINE {
            constexpr std::size_t e = IC<(r * n) % IP>{};
            const auto [fr, fi] = apply_stage_twiddle<T, IP, e, V>(yr, yi);
            fr.store_aligned(ar + (r * N2 + n) * W);
            fi.store_aligned(ai + (r * N2 + n) * W);
        });
    });
    poet::static_for<0, N1>([&](const auto r) ADM_LAMBDA_ALWAYS_INLINE {
        constexpr std::size_t off = r * N2 * W;
        staged_dif_stage_b<T, N2, 1u, V>(
            ar + off, ai + off, [&](const auto k2, V yr, V yi) ADM_LAMBDA_ALWAYS_INLINE {
                emit(IC<r + N1 * k2>{}, yr, yi);
            });
    });
}

// Register budget for the small-N scalar butterfly. On x86-64, long double runs on the
// 8-register x87 FPU stack, so a monolithic 2*N working set spills once 2*N > 8. float and
// double keep the monolithic small path (their code is unchanged), so they get a budget that
// keeps the staged split off for N <= 16.
template<typename T>
[[nodiscard]] constexpr std::size_t scalar_dif_registers() {
    if constexpr (std::is_same_v<T, long double>) return 8u;
    return 32u;
}

// Scalar staged DIF for the small path: the Gentleman-Sande outer split on scalar lanes, so
// each stage keeps only N1 (stage A) or N2 (stage B) complex values live. For N <= 16 both
// fit the 8 x87 registers, so the monolithic spill disappears. N = N1 * N2, N2 =
// staged_dif_n2<N>(); input j = n + N2*m, output r + N1*k2, cross twiddle w_N^(r*n).
template<typename T, std::size_t N, typename Emit>
ADM_ALWAYS_INLINE void scalar_staged_dif(const T (&tr)[N], const T (&ti)[N], Emit&& emit) {
    constexpr std::size_t N2 = staged_dif_n2<N>();
    constexpr std::size_t N1 = N / N2;
    static_assert(N1 * N2 == N && N1 >= 2u && N2 >= 2u, "scalar staged split must factor N");
    T ar[N], ai[N];
    poet::static_for<0, N2>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
        T br[N1], bi[N1];
        poet::static_for<0, N1>([&](const auto m) ADM_LAMBDA_ALWAYS_INLINE {
            br[m] = tr[n + N2 * m];
            bi[m] = ti[n + N2 * m];
        });
        sub_dft<T, N1, T>(br, bi, [&](const auto r, T yr, T yi) ADM_LAMBDA_ALWAYS_INLINE {
            constexpr std::size_t e = IC<(r * n) % N>{};
            const auto [fr, fi] = apply_stage_twiddle<T, N, e, T>(yr, yi);
            ar[r * N2 + n] = fr;
            ai[r * N2 + n] = fi;
        });
    });
    poet::static_for<0, N1>([&](const auto r) ADM_LAMBDA_ALWAYS_INLINE {
        T cr[N2], ci[N2];
        poet::static_for<0, N2>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
            cr[n] = ar[r * N2 + n];
            ci[n] = ai[r * N2 + n];
        });
        sub_dft<T, N2, T>(cr, ci, [&](const auto k2, T yr, T yi) ADM_LAMBDA_ALWAYS_INLINE {
            emit(IC<r + N1 * k2>{}, yr, yi);
        });
    });
}

// Route the small scalar butterfly through the staged split on cost, not on N. Structural
// precondition: the 2-way split needs N to factor into two integers of 2 or more, so a prime N
// (17) cannot be split and stays monolithic. Cost test: take the split when the monolithic 2N
// live set is at least four register files wide (2N >= 4*regs), so the 80-bit spill traffic
// saved outweighs one full extra pass over the intermediate. float and double keep the
// monolithic path unchanged (their small-N code is byte-identical to base).
template<typename T, std::size_t N>
[[nodiscard]] constexpr bool use_scalar_staged() {
    constexpr std::size_t N2 = staged_dif_n2<N>();
    constexpr bool splittable = N >= 4u && N2 >= 2u && N % N2 == 0u && N / N2 >= 2u;
    return std::is_same_v<T, long double> &&
        splittable &&
        2u * N >= 4u * scalar_dif_registers<T>();
}

// Radices this large spill the monolithic butterfly's 2*IP live registers, so passes route
// them through the staged split above instead.
template<std::size_t IP>
inline constexpr bool dif_staged_radix =
    detail::has_single_bit(IP) && IP >= 16 &&
    dif_butterfly_wants_reload<IP> && poet::vector_register_count() >= 32;

template<std::size_t IP>
ADM_CONSTEVAL std::size_t dif_pass_unroll() {
    constexpr std::size_t peak_live = 2u * IP + 10u;
    constexpr std::size_t budget = poet::vector_register_count();
    constexpr std::size_t u = budget / peak_live;
    return u < 1u ? 1u : u;
}

template<typename T>
using r2c_small_fn = void (*)(const T*, std::complex<T>*);
template<typename T>
using c2r_small_fn = void (*)(const std::complex<T>*, T*);
template<typename T>
using c2c_small_fn = void (*)(std::complex<T>*);

// The whole r2c of one small N as one straight-line DFT of the real row. The imaginary inputs
// are zero and the bins above N / 2 are not stored, so the compiler drops their arithmetic.
// All loads come before the first store, so `out` may alias `in`. The zero folding needs the
// fast-math flags of the library build; without them the result is the same, only slower.
template<std::size_t N, typename T>
void r2c_small(const T* in, std::complex<T>* out) {
    T tr[N], ti[N];
    poet::static_for<0, N>([&](auto J) {
        tr[J] = in[J];
        ti[J] = T(0);
    });
    if constexpr (use_scalar_staged<T, N>()) {
        scalar_staged_dif<T, N>(tr, ti, [&](auto K, T re, T im) {
            if constexpr (K <= N / 2) out[K] = std::complex<T>(re, im);
        });
    } else {
        dif_butterfly<T, N, T>(tr, ti, [&](auto K, T re, T im) {
            if constexpr (K <= N / 2) out[K] = std::complex<T>(re, im);
        });
    }
}

// The unscaled c2r of one small N from its N / 2 + 1 bins, as one straight-line DFT. Bin N - k
// is the conjugate of bin k, and the imaginary parts of DC and Nyquist drop. The inverse DFT
// is the forward DFT with the real and imaginary parts swapped on input and output.
template<std::size_t N, typename T>
void c2r_small(const std::complex<T>* in, T* out) {
    T tr[N], ti[N];
    poet::static_for<0, N>([&](auto J) {
        constexpr std::size_t k = J <= N / 2 ? static_cast<std::size_t>(J) : N - J;
        if constexpr (k == 0 || 2 * k == N) tr[J] = T(0);
        else if constexpr (J <= N / 2) tr[J] = in[k].imag();
        else tr[J] = -in[k].imag();
        ti[J] = in[k].real();
    });
    dif_butterfly<T, N, T>(tr, ti, [&](auto K, T, T im) { out[K] = im; });
}

// One c2c of a small N in place, as one straight-line DFT. The inverse swaps the real and
// imaginary parts on input and output. All loads come before the first store.
template<std::size_t N, typename T, bool Forward>
void c2c_small(std::complex<T>* x) {
    T tr[N], ti[N];
    poet::static_for<0, N>([&](auto J) {
        tr[J] = Forward ? x[J].real() : x[J].imag();
        ti[J] = Forward ? x[J].imag() : x[J].real();
    });
    if constexpr (use_scalar_staged<T, N>()) {
        scalar_staged_dif<T, N>(tr, ti, [&](auto K, T re, T im) {
            x[K] = Forward ? std::complex<T>(re, im) : std::complex<T>(im, re);
        });
    } else {
        dif_butterfly<T, N, T>(tr, ti, [&](auto K, T re, T im) {
            x[K] = Forward ? std::complex<T>(re, im) : std::complex<T>(im, re);
        });
    }
}

template<typename T>
struct r2c_small_pick {
    template<std::size_t N>
    r2c_small_fn<T> operator()() const { return &r2c_small<N, T>; }
};

template<typename T>
struct c2r_small_pick {
    template<std::size_t N>
    c2r_small_fn<T> operator()() const { return &c2r_small<N, T>; }
};

template<typename T, bool Forward>
struct c2c_small_pick {
    template<std::size_t N>
    c2c_small_fn<T> operator()() const { return &c2c_small<N, T, Forward>; }
};

// The codelet of Pick for N in [2, 17], else null.
template<typename Pick>
auto pick_small(std::size_t N) {
    using sizes = std::integer_sequence<std::size_t, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                        15, 16, 17>;
    return poet::dispatch(Pick{}, poet::dispatch_param<sizes>{N});
}

}
}

#include "undef_macros.hpp"
