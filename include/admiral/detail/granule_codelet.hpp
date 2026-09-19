#pragma once

// AoS-granule leaf kernels: V::size/2 interleaved complex granules per register, one
// granule op per complex op of the SoA text, lanes = lines/columns. Mirrors butterfly.hpp
// decision-for-decision on a granule-parallel register, so op-count parity holds by
// construction (each granule FP op replaces the SoA re/im pair: half the FP events).
// fp-contract safe by spelling: every FP op is an explicit xsimd::{add,sub,mul,fma},
// so no build flavor has anything left to contract. Constants fold from the same
// ct_sincos_turns source of truth as the SoA engine -- no decimal literals.
//
// Direction is a template bool folded into the constants (inverse = conjugated twiddles:
// the DIF structure is linear in its twiddles, so conjugating every constant yields the
// inverse transform). That is a new rounding contract versus the SoA inverse's swapped
// re/im planes, and it is deliberate: the granule dialect IS the new-contract family.

#include <array>
#include <cstddef>
#include <type_traits>

#include <poet/poet.hpp>
#include "simd.hpp"

#include "butterfly.hpp"
#include "ct_math.hpp"
#include "cxx_compat.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

// The one compile-time ISA probe: the arch families in which the pinned xsimd emits
// vfmaddsub-class for the pooled fmas forms. Same spelling class as kFusedFma
// (simd_swizzle.hpp), minus NEON/SVE/RVV/VSX/VXE: those lower the fmas idiom through
// select+neg (gate-synthesis measurements), unmeasured for this dialect, so excluded in v1.
// Below AVX2/FMA3 (e.g. x86-64-v2) the whole family is if constexpr-dead.
inline constexpr bool kGranuleFmaddsub = XSIMD_WITH_FMA3_SSE || XSIMD_WITH_FMA3_AVX
                                         || XSIMD_WITH_FMA3_AVX2 || XSIMD_WITH_FMA4
                                         || XSIMD_WITH_AVX512F;

// Admission: an entry is true only where the dialect was measured for that precision.
// 16 f64 is a measured LOSS (aos-sweep-n16 0.87x/0.93x) and stays out; N < 9 stays with
// the flat arms; N > 24 waits on register-cliff work. Long double never compiles here.
//
// ADM_GRANULE_ADMIT_OFF (CMake -DADM_GRANULE_ADMIT=OFF, WI-0a's per-class transfer
// audit) forces every entry false: admission is if constexpr, so the granule arms drop
// out at compile time and the SoA fallback runs. The knob serves one decision rule: an
// admission that does not beat OFF beyond 2x the in-job control spread on every node
// class goes OFF globally. The default-ON arm is textually the pre-knob predicate, so a
// default build carries the knob absent and unchanged object code.
#ifdef ADM_GRANULE_ADMIT_OFF
template<unsigned N, typename T>
inline constexpr bool granule_rows_admit_v = false;
template<unsigned N, typename T>
inline constexpr bool granule_cols_admit_v = false;
template<unsigned N, typename T>
inline constexpr bool granule_cube_admit_v = false;
#else
// Rows and cols are separate, independently-priced tables: a leaf that pays as a row
// tile does not have to pay as a column tile (13 admits rows only, because no col
// route reaches a 11-non-smooth length).
template<unsigned N, typename T>
inline constexpr bool granule_rows_admit_v =
    is_simd_precision_v<T> && kGranuleFmaddsub &&
    (N == 12u || N == 24u || (N == 16u && std::is_same_v<T, float>) ||
     (N == 10u && std::is_same_v<T, float>) ||
     (N == 10u && std::is_same_v<T, double>) ||
     (N == 14u && std::is_same_v<T, double>) ||
     (N == 14u && std::is_same_v<T, float>) ||
     (N == 15u && std::is_same_v<T, float>) ||
     (N == 15u && std::is_same_v<T, double>) ||
     (N == 13u && std::is_same_v<T, double>) ||
     (N == 9u && std::is_same_v<T, double>) ||
     (N == 13u && std::is_same_v<T, float>) ||
     (N == 11u && std::is_same_v<T, double>));
template<unsigned N, typename T>
inline constexpr bool granule_cols_admit_v =
    is_simd_precision_v<T> && kGranuleFmaddsub &&
    (N == 12u || N == 24u || (N == 16u && std::is_same_v<T, float>) ||
     (N == 10u && std::is_same_v<T, float>) ||
     (N == 10u && std::is_same_v<T, double>) ||
     (N == 14u && std::is_same_v<T, double>) ||
     (N == 14u && std::is_same_v<T, float>) ||
     (N == 11u && std::is_same_v<T, double>) ||
     (N == 15u && std::is_same_v<T, double>) ||
     (N == 15u && std::is_same_v<T, float>) ||
     (N == 9u && std::is_same_v<T, double>));
// The cube set is decoupled from rows/cols on purpose: the rows/cols admission keeps
// N < 9 with the flat arms, and extending either table to 4/8 would reroute every col
// axis at those lengths (e.g. {64,4,4}'s inner axes). The cube seat composes three
// granule passes into a rank-3 block transform whose competitor is the whole general
// chain, so its measured crossover set differs: N in {4, 8}. Do not merge the tables.
template<unsigned N, typename T>
inline constexpr bool granule_cube_admit_v =
    is_simd_precision_v<T> && kGranuleFmaddsub && (N == 4u || N == 8u);
#endif

// ----------------------------------------------------------------------
// Constant pools. One per-precision-per-width repeated pair {a, b} per granule,
// folded at compile time from ct_sincos_turns values. Function-scope statics, the
// codelet.hpp twiddle-table idiom; loaded with load_unaligned (alignof stays
// std::array's own).
// ----------------------------------------------------------------------
template<typename T, std::size_t W>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> granule_pair_pool(T a, T b) {
    static_assert(W % 2 == 0, "granule pools tile whole complex pairs");
    std::array<T, W> r{};
    for (std::size_t i = 0; i < W; i += 2) {
        r[i] = a;
        r[i + 1] = b;
    }
    return r;
}

// ----------------------------------------------------------------------
// Granule primitives. Index space is per-T-lane, so one spelling serves both
// precisions (adjacent re/im lanes swap at float and at double granularity alike)
// and every sized-batch rung (the generator reads the arch's width).
// ----------------------------------------------------------------------
struct granule_swap_lane {  // [re,im] -> [im,re] within every granule (fr_swap_pair's i^1)
    static constexpr std::size_t get(std::size_t i, std::size_t) { return i ^ 1; }
};

template<typename V>
ADM_ALWAYS_INLINE V granule_swp(V a) {
    return xsimd::swizzle(a, xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<
                                 typename V::value_type>,
                                 granule_swap_lane, typename V::arch_type>());
}

// Sign flips go through top-bit INTEGER masks: an FP pool of -0.0 entries is a constant
// the FP folder may rewrite under -ffast-math (gcc 13.3 at -O3 -ffast-math carries
// granule_rot_pos_i's function-scope {-0.0,+0.0,...} pool to {+0.0,-0.0,...} and inverts
// every odd output of the 12-point col body at v3). The integer mask folds identically
// on every compiler, and the integer xor lowers to the same vxorps-class op.
template<std::size_t Parity, typename V>
ADM_ALWAYS_INLINE V granule_flip(V u) {
    using T = typename V::value_type;
    using U = xsimd::as_unsigned_integer_t<T>;
    using B = xsimd::batch<U, typename V::arch_type>;
    constexpr unsigned top = sizeof(U) * 8 - 1;
    static constexpr auto mask = [] {
        std::array<U, V::size> m{};
        for (std::size_t i = Parity; i < V::size; i += 2) m[i] = U(1) << top;
        return m;
    }();
    return xsimd::bitwise_cast<T>(
        xsimd::bitwise_xor(xsimd::bitwise_cast<U>(u), B::load_unaligned(mask.data())));
}

// [gr,gi] -> [gi,-gr]  (multiply by -i): swap, then flip the new imaginary lane.
template<typename V>
ADM_ALWAYS_INLINE V granule_rot_neg_i(V u) {
    return granule_flip<1>(granule_swp(u));
}

// [gr,gi] -> [-gi,gr]  (multiply by +i): swap, then flip the new real lane.
template<typename V>
ADM_ALWAYS_INLINE V granule_rot_pos_i(V u) {
    return granule_flip<0>(granule_swp(u));
}

// ----------------------------------------------------------------------
// Stage twiddle, mirroring apply_stage_twiddle (butterfly.hpp) case-for-case on one
// granule register. The generic case is one broadcast fma plus one pooled fma over the
// swapped input: [c*gr - s*gi, c*gi + s*gr] = fma(V(c), u, mul(pool{-s,+s}, swp(u))),
// 1 shuffle + 1 mul + 1 fma per complex multiply.
// ----------------------------------------------------------------------
template<typename T, std::size_t W, typename F>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> granule_pool_of(const ct_sincos_v<F>& w, T) {
    return granule_pair_pool<T, W>(static_cast<T>(-w.s), static_cast<T>(w.s));
}

template<typename T, std::size_t IP, std::size_t N, typename V, bool Fwd>
[[nodiscard]] ADM_ALWAYS_INLINE V granule_stage_twiddle(V u) {
    constexpr auto w = ct_sincos_turns<ct_real_t<T>>(Fwd, N, IP);
    // The arm is picked by the integer root-form descriptor, as in apply_stage_twiddle;
    // the else arm still serves anti_diag and generic together.
    constexpr root_form form = ct_root<N, IP, Fwd>();
    if constexpr (form == root_form::one) {
        return u;
    } else if constexpr (form == root_form::neg_one) {
        return xsimd::neg(u);
    } else if constexpr (form == root_form::neg_i) {
        return granule_rot_neg_i(u);
    } else if constexpr (form == root_form::pos_i) {
        return granule_rot_pos_i(u);
    } else if constexpr (form == root_form::diag) {
        // {c*(gr-gi), c*(gi+gr)}: pooled lanes {-c,+c} instead of {-s,+s}.
        static constexpr auto pool = [] {
            constexpr auto wv = ct_sincos_turns<ct_real_t<T>>(Fwd, N, IP);
#ifdef ADM_GRANULE_TEST_POISON
            // Positive-control seam for the poison twin (WILL_FAIL), alongside the radix-3
            // one in granule_sym_s_pool: the 8-point stage is the only FP-pool consumer in
            // the 4^3/8^3 cubes (4^3's stage constants are all integer-mask rotations), so
            // lane-flipping both 8^3 pools must fail the twin's 8^3 direct-DFT gate. (The
            // size is this template's IP; N is the butterfly index.)
            if (IP == 8u)
                return granule_pair_pool<T, V::size>(static_cast<T>(wv.c),
                                                     static_cast<T>(-wv.c));
#endif
            return granule_pair_pool<T, V::size>(static_cast<T>(-wv.c), static_cast<T>(wv.c));
        }();
        return xsimd::fma(V(static_cast<T>(w.c)), u,
                          xsimd::mul(V::load_unaligned(pool.data()), granule_swp(u)));
    } else {
        // c == -w.s and the generic form share the same pool {-s,+s} through this helper.
        static constexpr auto pool = [] {
            constexpr auto wv = ct_sincos_turns<ct_real_t<T>>(Fwd, N, IP);
#ifdef ADM_GRANULE_TEST_POISON
            if (IP == 8u)
                return granule_pair_pool<T, V::size>(static_cast<T>(wv.s),
                                                     static_cast<T>(-wv.s));
#endif
            return granule_pool_of<T, V::size>(wv, T(0));
        }();
        return xsimd::fma(V(static_cast<T>(w.c)), u,
                          xsimd::mul(V::load_unaligned(pool.data()), granule_swp(u)));
    }
}

// ----------------------------------------------------------------------
// Radix bodies, mirroring butterfly.hpp one granule op per complex op.
// Every body takes (const V (&x)[IP], V (&y)[IP]); x == y is legal at the top level
// (each body's structure consumes its whole input into registers before the first
// output write, the same argument the SoA col engine's in-place reuse makes).
// ----------------------------------------------------------------------

// Pool for the sine terms of one (m,k) of radix_sym: lanes {+s,-s} so the pooled
// accumulate over swp(d[m]) yields the granule [QR,-QI] and the two emits are add/sub.
template<typename T, std::size_t W, bool Fwd, std::size_t MK, std::size_t IP>
struct granule_sym_s_pool {
    static constexpr auto arr = [] {
        constexpr auto w = ct_sincos_turns<ct_real_t<T>>(!Fwd, MK % IP, IP);
#ifdef ADM_GRANULE_TEST_POISON
        // Positive-control seam for the poison twin test (WILL_FAIL): one deliberately
        // lane-flipped radix-3 entry. Never defined in a shipped build.
        if (IP == 3u && MK % IP == 1u)
            return granule_pair_pool<T, W>(static_cast<T>(-w.s), static_cast<T>(w.s));
#endif
        return granule_pair_pool<T, W>(static_cast<T>(w.s), static_cast<T>(-w.s));
    }();
};

template<typename T, std::size_t IP, bool Fwd, std::size_t MK, typename V>
ADM_ALWAYS_INLINE V granule_sym_c_term(V acc, V am) {
    constexpr auto w = ct_sincos_turns<ct_real_t<T>>(!Fwd, MK % IP, IP);
    return xsimd::fma(V(static_cast<T>(w.c)), am, acc);
}

template<typename T, std::size_t IP, bool Fwd, std::size_t MK, typename V>
ADM_ALWAYS_INLINE V granule_sym_s_term(V acc, V dsm) {
    using P = granule_sym_s_pool<T, V::size, Fwd, MK, IP>;
    return xsimd::fma(V::load_unaligned(P::arr.data()), dsm, acc);
}

// Mirrors radix_sym_dft: SoA pays 4 fma per (m,k) (PR,PI,QR,QI chains); the granule
// pays 2 (one per P / pooled-Q chain), the dialect's census halving.
template<typename T, std::size_t IP, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_radix_sym(const V (&x)[IP], V (&y)[IP]) {
    static_assert(IP % 2 == 1 && IP >= 3, "symmetric DFT is for odd radix >= 3");
    constexpr std::size_t H = (IP - 1) / 2;
    V a[H + 1], ds[H + 1];
    const V x0 = x[0];  // snapshot: y[0]'s write precedes the k loop's P = x0 reads
    poet::static_for<1, H + 1>([&](const auto m) ADM_LAMBDA_ALWAYS_INLINE {
        a[m] = xsimd::add(x[m], x[IP - m]);
        ds[m] = granule_swp(xsimd::sub(x[m], x[IP - m]));  // swizzle hoisted per m
    });
    V s0 = x0;
    poet::static_for<1, H + 1>([&](const auto m) ADM_LAMBDA_ALWAYS_INLINE {
        s0 = xsimd::add(s0, a[m]);
    });
    y[0] = s0;
    poet::static_for<1, H + 1>([&](const auto k) ADM_LAMBDA_ALWAYS_INLINE {
        V P = x0, Q = V(T(0));
        poet::static_for<1, H + 1>([&](const auto m) ADM_LAMBDA_ALWAYS_INLINE {
#if defined(_MSC_VER) && !defined(__clang__)
            // Same MSVC C2131 class as butterfly.hpp's radix_sym_dft: the captured outer
            // integral_constant is a closure read to its constant evaluator; name the value.
            constexpr std::size_t mk = (m * decltype(k)::value) % IP;
#else
            constexpr std::size_t mk = (m * k) % IP;
#endif
            P = granule_sym_c_term<T, IP, Fwd, mk>(P, a[m]);
            Q = granule_sym_s_term<T, IP, Fwd, mk>(Q, ds[m]);
        });
        y[k] = xsimd::add(P, Q);        // SoA's {PR + QR, PI - QI}
        y[IP - k] = xsimd::sub(P, Q);   // SoA's {PR - QR, PI + QI}
    });
}

// Mirrors sub_dft: pow2 radices to granule_pow2, odd ones to granule_radix_sym.
template<typename T, std::size_t IP, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_sub_dft(const V (&x)[IP], V (&y)[IP]);

// Mirrors pow2_dif_butterfly: evens untwiddled, odds stage-twiddled, interleaved emit.
template<typename T, std::size_t IP, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_pow2(const V (&x)[IP], V (&y)[IP]) {
    static_assert(IP >= 2 && detail::has_single_bit(IP),
                  "granule_pow2: IP must be a power of two >= 2");
    if constexpr (IP == 2) {
        y[0] = xsimd::add(x[0], x[1]);
        y[1] = xsimd::sub(x[0], x[1]);
    } else {
        constexpr std::size_t H = IP / 2;
        V e[H], f[H], eo[H], fo[H];
        poet::static_for<0, H>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
            e[n] = xsimd::add(x[n], x[n + H]);
            f[n] = granule_stage_twiddle<T, IP, n, V, Fwd>(xsimd::sub(x[n], x[n + H]));
        });
        granule_pow2<T, H, V, Fwd>(e, eo);
        granule_pow2<T, H, V, Fwd>(f, fo);
        poet::static_for<0, H>([&](const auto n) ADM_LAMBDA_ALWAYS_INLINE {
            y[2 * n] = eo[n];
            y[2 * n + 1] = fo[n];
        });
    }
}

template<typename T, std::size_t IP, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_sub_dft(const V (&x)[IP], V (&y)[IP]) {
    if constexpr (detail::has_single_bit(IP))
        granule_pow2<T, IP, V, Fwd>(x, y);
    else
        granule_radix_sym<T, IP, V, Fwd>(x, y);
}

// Mirrors pfa_dif_butterfly: gather (n1*N2 + n2*N1) % IP, sub-DFTs of N1 then N2,
// CRT placement through the shared crt_index. No inter-stage twiddles (coprime).
template<typename T, std::size_t N1, std::size_t N2, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_pfa(const V (&x)[N1 * N2], V (&y)[N1 * N2]) {
    constexpr std::size_t IP = N1 * N2;
    static_assert(std::gcd(N1, N2) == 1, "PFA requires coprime factors");
    V a[N1][N2];
    poet::static_for<0, N2>([&](const auto n2) ADM_LAMBDA_ALWAYS_INLINE {
        V b[N1], t[N1];
        poet::static_for<0, N1>([&](const auto n1) ADM_LAMBDA_ALWAYS_INLINE {
            b[n1] = x[(n1 * N2 + n2 * N1) % IP];
        });
        granule_sub_dft<T, N1, V, Fwd>(b, t);
        poet::static_for<0, N1>([&](const auto k1) ADM_LAMBDA_ALWAYS_INLINE {
            a[k1][n2] = t[k1];
        });
    });
    poet::static_for<0, N1>([&](const auto k1) ADM_LAMBDA_ALWAYS_INLINE {
        V b[N2], t[N2];
        poet::static_for<0, N2>([&](const auto n2) ADM_LAMBDA_ALWAYS_INLINE {
            b[n2] = a[k1][n2];
        });
        granule_sub_dft<T, N2, V, Fwd>(b, t);
        poet::static_for<0, N2>([&](const auto k2) ADM_LAMBDA_ALWAYS_INLINE {
            y[crt_index<N1, N2, k1, k2>] = t[k2];
        });
    });
}

// Mirrors ct_dif_butterfly (both factors odd): contiguous gathers, a stage twiddle
// between the two radix_sym stages. Only N = 9 splits this way inside the built set.
template<typename T, std::size_t N1, std::size_t N2, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_ct(const V (&x)[N1 * N2], V (&y)[N1 * N2]) {
    constexpr std::size_t IP = N1 * N2;
    static_assert(N1 % 2 == 1 && N2 % 2 == 1 && N1 >= 3 && N2 >= 3,
                  "granule_ct: both factors must be odd radices >= 3");
    V a[N1][N2];
    poet::static_for<0, N2>([&](const auto n2) ADM_LAMBDA_ALWAYS_INLINE {
        V b[N1], t[N1];
        poet::static_for<0, N1>([&](const auto n1) ADM_LAMBDA_ALWAYS_INLINE {
            b[n1] = x[n1 * N2 + n2];
        });
        granule_radix_sym<T, N1, V, Fwd>(b, t);
        poet::static_for<0, N1>([&](const auto r) ADM_LAMBDA_ALWAYS_INLINE {
            a[r][n2] = granule_stage_twiddle<T, IP, r * n2, V, Fwd>(t[r]);
        });
    });
    poet::static_for<0, N1>([&](const auto r) ADM_LAMBDA_ALWAYS_INLINE {
        V t[N2];
        granule_radix_sym<T, N2, V, Fwd>(a[r], t);
        poet::static_for<0, N2>([&](const auto k2) ADM_LAMBDA_ALWAYS_INLINE {
            y[k2 * N1 + r] = t[k2];
        });
    });
}

// Mirrors dif_butterfly's decision chain for the sizes in the built set. A radix the
// mirror does not cover is a compile error here, not a silent fallthrough.
template<typename T, std::size_t IP, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_dif(const V (&x)[IP], V (&y)[IP]) {
    constexpr auto ct = odd_ct_split(IP);
    constexpr auto pf = coprime_split(IP);
    if constexpr (ct.first != 0) {
        granule_ct<T, ct.first, ct.second, V, Fwd>(x, y);
    } else if constexpr (IP % 2 == 1 && IP >= 3) {
        granule_radix_sym<T, IP, V, Fwd>(x, y);
    } else if constexpr (IP % 2 == 0 && pf.first != 0) {
        granule_pfa<T, pf.first, pf.second, V, Fwd>(x, y);
    } else if constexpr (IP >= 4 && detail::has_single_bit(IP)) {
        granule_pow2<T, IP, V, Fwd>(x, y);
    } else {
        static_assert(IP < 2, "granule_dif: no mirrored structure for this N");
    }
}

// The one whole-transform entry, mirroring dif_butterfly_terminal's switch.
// In place: each mirrored structure consumes its inputs before its first write.
template<std::size_t N, typename V, bool Fwd>
ADM_ALWAYS_INLINE void granule_dft(V (&g)[N]) {
    using T = typename V::value_type;
    static_assert(!is_rader_prime(N), "granule_dft: Rader has no granule form");
    constexpr auto split = coprime_split(N);
    if constexpr (N % 2 == 1 && split.first != 0) {
        granule_pfa<T, split.first, split.second, V, Fwd>(g, g);
    } else {
        granule_dif<T, N, V, Fwd>(g, g);
    }
}

// ----------------------------------------------------------------------
// Tile bodies (drivers live in src/granule_apply.hpp). Every load is the same
// granularity in every tile and unaligned, so no arm forks on buffer alignment and
// the dialect needs no numerics pin. Masked terminal blocks load/store through a
// compile-time mask; the pinned xsimd lowers those to fault-suppressing forms on
// AVX2 and AVX-512, so a terminal block is safe at the buffer end.
// ----------------------------------------------------------------------

// Two-source u64-lane gathers for the f64 128-bit-granule transpose (the n24
// tp_pair/tp_quad idiom). With W double lanes, granule q of register r sits at lane
// pair 2q; the 4x4 network is two pair-gathers per side plus one quad-merge per
// output register, 8 two-source shuffles total (vshuff64x2-class at AVX-512).
template<std::size_t Which>
struct granule_tp_pair {
    static constexpr std::size_t get(std::size_t i, std::size_t W) {
        const std::size_t half = W / 2;
        return i < half ? Which * half + i : W + Which * half + (i - half);
    }
};
template<std::size_t Which>
struct granule_tp_quad {
    static constexpr std::size_t get(std::size_t i, std::size_t W) {
        const std::size_t j = i / 2, odd = i % 2;
        return (j / 2) * W + ((j % 2) * 2 + Which) * 2 + odd;
    }
};

// In-place GL x GL granule transpose, GL = V::size / 2. f32 reads the block as
// 64-bit granules (double view, one xsimd::transpose); f64 granules are 128-bit and
// take the two-source shuffle network.
template<typename T, typename V, typename B>
ADM_ALWAYS_INLINE void granule_block_transpose(B (&b)[V::size / 2]) {
    if constexpr (sizeof(T) == 4) {
        xsimd::transpose(b, b + V::size / 2);
    } else {
        using A = typename V::arch_type;
        if constexpr (V::size == 8) {
            using u64 = xsimd::as_unsigned_integer_t<double>;
            const B u0 = xsimd::shuffle(b[0], b[1],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<0>, A>());
            const B u1 = xsimd::shuffle(b[0], b[1],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<1>, A>());
            const B u2 = xsimd::shuffle(b[2], b[3],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<0>, A>());
            const B u3 = xsimd::shuffle(b[2], b[3],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<1>, A>());
            b[0] = xsimd::shuffle(u0, u2, xsimd::make_batch_constant<u64, granule_tp_quad<0>, A>());
            b[1] = xsimd::shuffle(u0, u2, xsimd::make_batch_constant<u64, granule_tp_quad<1>, A>());
            b[2] = xsimd::shuffle(u1, u3, xsimd::make_batch_constant<u64, granule_tp_quad<0>, A>());
            b[3] = xsimd::shuffle(u1, u3, xsimd::make_batch_constant<u64, granule_tp_quad<1>, A>());
        } else {
            static_assert(V::size == 4, "granule f64 tiles need 2 or 4 granules per register");
            using u64 = xsimd::as_unsigned_integer_t<double>;
            const B u0 = xsimd::shuffle(b[0], b[1],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<0>, A>());
            const B u1 = xsimd::shuffle(b[0], b[1],
                                        xsimd::make_batch_constant<u64, granule_tp_pair<1>, A>());
            b[0] = u0;
            b[1] = u1;
        }
    }
}

// One row tile: GL consecutive lines (GL = V::size/2, lanes = lines), N granules per
// line decomposed into full GL-granule blocks plus one masked terminal block when
// GL does not divide N. in/out are T pointers, strides in complex units. Reads the
// whole tile into registers before writing, so in == out is safe block-locally.
template<unsigned N, typename T, bool Forward, bool Scaled, typename V>
ADM_ALWAYS_INLINE void granule_row_tile(const T* in, T* ADM_RESTRICT out,
                                        std::size_t in_stride, std::size_t out_stride, T fct) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t GL = W / 2;
    static_assert(GL >= 2, "granule tiles carry at least two granules per register");
    constexpr std::size_t DB = (sizeof(T) == 4) ? 1 : 2;   // double lanes per granule
    constexpr std::size_t NF = N / GL;                      // full blocks
    constexpr std::size_t NR = N % GL;                      // terminal-block granules
    using B = std::conditional_t<sizeof(T) == 4,
                                 xsimd::batch<double, typename V::arch_type>, V>;
    V g[N];
    poet::static_for<0, NF + (NR != 0 ? 1 : 0)>([&](auto qb_) ADM_LAMBDA_ALWAYS_INLINE {
        constexpr std::size_t qb = qb_;
        constexpr std::size_t r = qb < NF ? GL : NR;
        B b[GL];
        poet::static_for<0, GL>([&](auto l) ADM_LAMBDA_ALWAYS_INLINE {
            const double* p = reinterpret_cast<const double*>(in + l * 2 * in_stride) + qb * DB * GL;
            if constexpr (qb < NF) {
                b[l] = B::load_unaligned(p);
            } else {
                b[l] = B::load(p, xsimd::make_batch_bool_constant<double, lane_lt<DB * r>,
                                                                typename B::arch_type>(),
                               xsimd::unaligned_mode{});
            }
        });
        granule_block_transpose<T, V>(b);
        poet::static_for<0, r>([&](auto j) ADM_LAMBDA_ALWAYS_INLINE {
            if constexpr (sizeof(T) == 4) g[qb * GL + j] = xsimd::bitwise_cast<T>(b[j]);
            else g[qb * GL + j] = b[j];
        });
    });
    granule_dft<N, V, Forward>(g);
    if constexpr (Scaled) {
        const V f(fct);
        poet::static_for<0, N>([&](auto p) ADM_LAMBDA_ALWAYS_INLINE {
            g[p] = xsimd::mul(g[p], f);
        });
    }
    poet::static_for<0, NF + (NR != 0 ? 1 : 0)>([&](auto qb_) ADM_LAMBDA_ALWAYS_INLINE {
        constexpr std::size_t qb = qb_;
        // [[maybe_unused]]: MSVC's C4189 misses the reference through the j lambda's capture.
        [[maybe_unused]] constexpr std::size_t r = qb < NF ? GL : NR;
        B b[GL];
        poet::static_for<0, GL>([&](auto j) ADM_LAMBDA_ALWAYS_INLINE {
            if constexpr (j < r) {
                if constexpr (sizeof(T) == 4) b[j] = xsimd::bitwise_cast<double>(g[qb * GL + j]);
                else b[j] = g[qb * GL + j];
            } else {
                b[j] = B(0.0);
            }
        });
        granule_block_transpose<T, V>(b);
        poet::static_for<0, GL>([&](auto l) ADM_LAMBDA_ALWAYS_INLINE {
            double* p = reinterpret_cast<double*>(out + l * 2 * out_stride) + qb * DB * GL;
            if constexpr (qb < NF) {
                b[l].store_unaligned(p);
            } else {
                b[l].store(p, xsimd::make_batch_bool_constant<double, lane_lt<DB * r>,
                                                            typename B::arch_type>(),
                           xsimd::unaligned_mode{});
            }
        });
    });
}

// One column tile: GL columns (lanes = columns), registers = points, no transpose.
template<unsigned N, typename T, bool Forward, bool Scaled, typename V>
ADM_ALWAYS_INLINE void granule_col_tile(const std::complex<T>* in, std::size_t in_inner,
                                        std::complex<T>* ADM_RESTRICT out, std::size_t out_inner,
                                        std::size_t c, T scale) {
    V g[N];
    poet::static_for<0, N>([&](auto p) ADM_LAMBDA_ALWAYS_INLINE {
        g[p] = V::load_unaligned(reinterpret_cast<const T*>(in + p * in_inner + c));
    });
    granule_dft<N, V, Forward>(g);
    if constexpr (Scaled) {
        const V f(scale);
        poet::static_for<0, N>([&](auto p) ADM_LAMBDA_ALWAYS_INLINE {
            g[p] = xsimd::mul(g[p], f);
        });
    }
    poet::static_for<0, N>([&](auto p) ADM_LAMBDA_ALWAYS_INLINE {
        g[p].store_unaligned(reinterpret_cast<T*>(out + p * out_inner + c));
    });
}

}
}

#include "undef_macros.hpp"
