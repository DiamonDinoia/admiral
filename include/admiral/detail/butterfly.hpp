#pragma once

// Cooley-Tukey decimation in frequency: Gentleman and Sande, AFIPS Fall Joint Comp. Conf. 1966.
// Odd radices use Singleton's sum/difference DFT, IEEE Trans. Audio Electroacoust. 17 (1969) 93.

#include <cstddef>
#include <type_traits>
#include <utility>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"

#include "ct_math.hpp"

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
        emit(std::integral_constant<std::size_t, 0>{}, sr, si);
    }
    poet::static_for<1, H + 1>([&](const auto k) {
        V PR = xr[0], PI = xi[0], QR = V(T(0)), QI = V(T(0));
        poet::static_for<1, H + 1>([&](const auto m) {
            constexpr auto w = ct_sincos_turns<ct_real_t<T>>(false, (m * k) % IP, IP);
            PR = PR + V(static_cast<T>(w.c)) * ar[m];
            PI = PI + V(static_cast<T>(w.c)) * ai[m];
            QR = QR + V(static_cast<T>(w.s)) * di[m];
            QI = QI + V(static_cast<T>(w.s)) * dr[m];
        });
        emit(std::integral_constant<std::size_t, k>{}, PR + QR, PI - QI);
        emit(std::integral_constant<std::size_t, IP - k>{}, PR - QR, PI + QI);
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
            constexpr std::size_t src = (n1 * N2 + n2 * N1) % IP;
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
            emit(std::integral_constant<std::size_t, out>{}, yr, yi);
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
    if constexpr (w.s == 0 && w.c == 1) {
        return {fr, fi};
    } else if constexpr (w.s == 0 && w.c == -1) {
        return {-fr, -fi};
    } else if constexpr (w.c == 0 && w.s == -1) {
        return {fi, -fr};
    } else if constexpr (w.c == 0 && w.s == 1) {
        return {-fi, fr};
    } else if constexpr (w.c == w.s) {
        const V c(static_cast<T>(w.c));
        return {c * (fr - fi), c * (fi + fr)};
    } else if constexpr (w.c == -w.s) {
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
        emit(std::integral_constant<std::size_t, 0>{}, xr[0] + xr[1], xi[0] + xi[1]);
        emit(std::integral_constant<std::size_t, 1>{}, xr[0] - xr[1], xi[0] - xi[1]);
    } else {
        constexpr std::size_t H = IP / 2;
        {
            V er[H], ei[H];
            poet::static_for<0, H>([&](const auto n) {
                er[n] = xr[n] + xr[n + H];
                ei[n] = xi[n] + xi[n + H];
            });
            pow2_dif_butterfly<T, H, V>(er, ei, [&](auto Kc, V yr, V yi) {
                emit(std::integral_constant<std::size_t, 2 * Kc>{}, yr, yi);
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
                emit(std::integral_constant<std::size_t, 2 * Kc + 1>{}, yr, yi);
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
            constexpr std::size_t e = static_cast<std::size_t>(decltype(r)::value)
                                    * static_cast<std::size_t>(decltype(n2)::value);
            const auto [fr, fi] = apply_stage_twiddle<T, IP, e, V>(yr, yi);
            ar[r][n2] = fr;
            ai[r][n2] = fi;
        });
    });

    poet::static_for<0, N1>([&](const auto r) {
        radix_sym_dft<T, N2>(ar[r], ai[r], [&](const auto k2, V yr, V yi) {
            emit(std::integral_constant<std::size_t, k2 * N1 + r>{}, yr, yi);
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
                constexpr auto w = ct_sincos_turns<ct_real_t<T>>(true, jj * k, IP);
                sr = sr + V(static_cast<T>(w.c)) * tr[jj]
                        - V(static_cast<T>(w.s)) * ti[jj];
                si = si + V(static_cast<T>(w.c)) * ti[jj]
                        + V(static_cast<T>(w.s)) * tr[jj];
            });
            emit(std::integral_constant<std::size_t, k>{}, sr, si);
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

template<std::size_t IP>
ADM_CONSTEVAL std::size_t dif_pass_unroll() {
    constexpr std::size_t peak_live = 2u * IP + 10u;
    constexpr std::size_t budget = poet::vector_register_count();
    constexpr std::size_t u = budget / peak_live;
    return u < 1u ? 1u : u;
}

}
}

#include "undef_macros.hpp"
