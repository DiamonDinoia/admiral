#pragma once

// Flat tiny row codelets (ADM_ND_FLATTINY): the flat_row shape pushed below N = 16. A whole
// N <= 8 complex row is 2N fp lanes, so at the campaign widths it sits in K = 2N/W in
// {1, 2, 4, 8} registers, or HALF a register at N = 4, W = 16 where one register braids two
// rows. The batched incumbent (many_gather_x + kernel_batched + many_scatter_x) pays two
// W x W lane transposes per W-row block; here the inter-register DIF stages are
// shuffle-free butterflies (reused fr_stages), the intra-register stages are one lane
// exchange plus one two-source merge per register per stage, and the bit-reversed output
// order resolves to whole-register store relabeling (C == 1), one lane permutation (K == 1),
// or K two-source merges. Admission is an equality whitelist per (N, W), cut down by the
// w2 static census: an (N, W) whose shuffle+spill count does not fall against the incumbent
// is excluded in-file with its rationale.

#include <array>
#include <cstddef>
#include <utility>

#include <poet/poet.hpp>

#include "cxx_compat.hpp"
#include "ct_math.hpp"
#include "flat_row.hpp"
#include "simd.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

// Admission: an equality whitelist per (N, W, sizeof(T)). The w2 static census
// (evidence w2-tiny-census-{master,on}.txt) kept every (N, W, T) whose shuffles+spills
// fell against the batched incumbent; the r5 wave A/B then cut the default-on set by
// measured floor (team-r5-shared/ab/w2/, reroll mds), landing on f64 at W <= 4 only:
//   ADMIT:  N in {4, 8}, sizeof(T) == 8, W in {2, 4}.
//   EXCLUDED W = 8 f64 (v4 class): ice 3d_8 -3.3% WIN vs genoa +3.1% REAL-REGRESSION
//     (reroll-icelake.md:14 / reroll-genoa.md:14, dir-consistent both instruments) —
//     same (N, W, T), opposite sign, no geometry key separates them; 3d_4 at-best-noise
//     both hosts (ice +2.0% / genoa -0.9%, under the 2% lottery tolerance).
//   EXCLUDED W = 16 f32 (v4 braid) and all remaining f32: never reached a beyond-floor
//     wave measurement; static-only wins stay off by default.
//   Census losers excluded at that stage: N4 W8 f32 (48 = 48 sh per block, +15% instr),
//   N8 W16 (sh 64 -> 96 UP).
template<unsigned N, typename V>
[[nodiscard]] ADM_CONSTEVAL bool ft_shape() {
    constexpr std::size_t W = V::size;
    constexpr std::size_t B = sizeof(typename V::value_type);
    if constexpr (N == 4 || N == 8) return B == 8u && (W == 2u || W == 4u);
    else return false;
}
template<unsigned N, typename V>
inline constexpr bool kFlatTiny =
    XSIMD_TARGET_X86 != 0 && (N & (N - 1u)) == 0u && ft_shape<N, V>() &&
    (2u * N >= V::size ? 2u * N / V::size : 1u) + 4u <=
        usable_vector_regs(poet::vector_register_count());

[[nodiscard]] ADM_CONSTEVAL std::size_t ft_log2(std::size_t x) {
    std::size_t r = 0;
    while (x > 1) { x >>= 1; ++r; }
    return r;
}

// Reverse the low log2(B) bits of x (B a power of two).
[[nodiscard]] ADM_CONSTEVAL std::size_t ft_rev(std::size_t x, std::size_t B) {
    std::size_t r = 0;
    for (std::size_t b = 0, n = ft_log2(B); b < n; ++b) r = 2 * r + ((x >> b) & 1u);
    return r;
}

// Stage-M intra partner map, in fp lanes: complex lane c exchanges with c ^ M (partner
// distance M complexes); each fp lane follows its complex lane. For the braided layout the
// map stays row-local because 2M <= N.
template<std::size_t M>
struct ft_xor {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        return 2u * ((i / 2u) ^ M) + (i % 2u);
    }
};

// Stage-M merge: lanes with bit log2(M) set take the (twiddled) diff in source b, the rest
// the sum in source a. Index >= size picks b, mirroring fr_merge_lo's convention.
template<std::size_t M>
struct ft_sel {
    static constexpr std::size_t get(std::size_t i, std::size_t size) {
        return (((i / 2u) & M) != 0u ? size : 0u) + i;
    }
};

// Per-lane stage twiddle for intra sub-length 2M: the diff at complex lane c gets
// w_{2M}^{c mod M}; the register index cancels because C is a multiple of 2M. Same AoS
// [u, v] pair form as fr_tw.
template<std::size_t M, bool Conj, typename T, std::size_t W>
[[nodiscard]] ADM_CONSTEVAL std::array<T, W> ft_tw(bool swapped) {
    constexpr std::size_t C = W / 2;
    std::array<T, W> a{};
    for (std::size_t c = 0; c < C; ++c) {
        const ct_sincos_t w = ct_sincos_turns(Conj, c % M, 2 * M);
        a[2 * c] = swapped ? -static_cast<T>(w.s) : static_cast<T>(w.c);
        a[2 * c + 1] = swapped ? static_cast<T>(w.s) : static_cast<T>(w.c);
    }
    return a;
}

// Braid assembly: row 0 lands in fp lanes [0, 2N), row 1 in [2N, 4N), sourced from the two
// masked loads. And the inverse: pull row 1 down to the low 2N lanes for its masked store
// (the upper lanes are junk, never stored).
template<unsigned N>
struct ft_braid_ld {
    static constexpr std::size_t get(std::size_t i, std::size_t size) {
        return i < 2u * N ? i : size + (i - 2u * N);
    }
};
template<unsigned N>
struct ft_braid_hi {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        return i < 2u * N ? i + 2u * N : i;
    }
};

// Bit-reversed complex lane permutation of a C-complex register (the K == 1 output order).
template<std::size_t C>
struct ft_rev_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        return 2u * ft_rev(i / 2u, C) + (i % 2u);
    }
};
// Braided two-row analogue: reverse the two low bits of each four-complex-lane half.
struct ft_rev_lane2 {
    static constexpr std::size_t get(std::size_t i, std::size_t) {
        const std::size_t c = i / 2u;
        const std::size_t b = c % 4u;
        return 2u * ((c / 4u) * 4u + (b % 2u) * 2u + b / 2u) + (i % 2u);
    }
};

// The K >= 2 multi-register output merge. After the last stage register k holds positions
// k * C + c, and position p owns output slot rev_N(p) = rev_C(p mod C) * K + rev_K(p / C):
// output register o, complex lane m sources register ft_rev((o * C + m) mod K, K) at lane
// ft_rev((o * C + m) / K, C). The admitted shapes give two sources per output register, so
// this is one two-source merge each, K in total.
template<std::size_t O, std::size_t K, std::size_t C>
[[nodiscard]] ADM_CONSTEVAL std::pair<std::size_t, std::size_t> ft_out_srcs() {
    std::size_t k0 = K, k1 = K;
    for (std::size_t m = 0; m < C; ++m) {
        const std::size_t k = ft_rev((O * C + m) % K, K);
        if (k0 == K) { k0 = k; }
        else if (k != k0) { k1 = k; return {k0, k1}; }
    }
    return {k0, k1};
}
template<std::size_t O, std::size_t K, std::size_t C>
struct ft_outm {
    static constexpr std::size_t get(std::size_t i, std::size_t size) {
        const std::size_t m = i / 2u;
        const std::size_t s = O * C + m;
        const std::size_t k = ft_rev(s % K, K);
        const std::size_t l = ft_rev(s / K, C);
        return (k == ft_out_srcs<O, K, C>().first ? 0u : size) + 2u * l + (i % 2u);
    }
};

// d[j + M] = (d[j] - d[j + M]) * w_{2M}^j and d[j] = d[j] + d[j + M], intra-register. With
// p = partner(d), the diff that belongs at lane j + M is d[j] - d[j + M] = p - d there
// (sum lanes of p - d are junk, discarded by the merge), so the diff is p - d, NOT d - p:
// d - p holds the negated value at exactly the lanes the merge keeps.
template<std::size_t M, bool Forward, typename T, typename V>
ADM_ALWAYS_INLINE void ft_intra_stage(V& d) {
    using index = xsimd::as_unsigned_integer_t<T>;
    using arch = typename V::arch_type;
    constexpr std::size_t W = V::size;
    const V p = xsimd::swizzle(d, xsimd::make_batch_constant<index, ft_xor<M>, arch>());
    const V s = d + p;
    V df = p - d;
    if constexpr (M > 1) {
        static constexpr auto tu = ft_tw<M, Forward, T, W>(false);
        static constexpr auto tv = ft_tw<M, Forward, T, W>(true);
        df = fr_cmul(df, V::load_unaligned(tu.data()), V::load_unaligned(tv.data()));
    }
    d = xsimd::shuffle(s, df, xsimd::make_batch_constant<index, ft_sel<M>, arch>());
}

// The intra stages M = Mfirst down to 1, run identically in all K registers.
template<std::size_t Mfirst, std::size_t K, bool Forward, typename T, typename V>
ADM_ALWAYS_INLINE void ft_intra_all(V* d) {
    if constexpr (Mfirst >= 1) {
        poet::static_for<0, K>([&](auto k) { ft_intra_stage<Mfirst, Forward, T, V>(d[k]); });
        ft_intra_all<Mfirst / 2, K, Forward, T, V>(d);
    }
}

// One whole row, 2N/W >= 1 registers: contiguous loads, inter-register DIF stages while the
// partner distance spans registers (reused fr_stages), intra stages below one register, then
// the output order fix per (K, C) shape. The stores are relabeled or merged, never a
// transpose tree, and the scale folds into them.
template<unsigned N, typename T, typename V, bool Forward>
ADM_ALWAYS_INLINE void flat_tiny_apply(const T* ADM_RESTRICT srcp, T* ADM_RESTRICT dstp,
                                       V f) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t C = W / 2;
    constexpr std::size_t K = 2 * N / W;
    static_assert(K >= 1 && 2u * N == K * W, "flat tiny: the row must tile whole registers");
    static_assert(K == 1 || K == 2 || K == 4 || K == 8, "derived for K in {1, 2, 4, 8}");
    using index = xsimd::as_unsigned_integer_t<T>;
    using arch = typename V::arch_type;

    V d[K];
    poet::static_for<0, K>([&](auto k) { d[k] = V::load_unaligned(srcp + k * W); });
    fr_stages<N, N / 2, Forward, T, V>(d);
    if constexpr (C >= 2) ft_intra_all<C / 2, K, Forward, T, V>(d);

    if constexpr (K == 1) {
        const V r = xsimd::swizzle(d[0],
                                   xsimd::make_batch_constant<index, ft_rev_lane<C>, arch>());
        (r * f).store_unaligned(dstp);
    } else if constexpr (C == 1) {
        poet::static_for<0, K>([&](auto k) {
            constexpr std::size_t kk = k;
            (d[k] * f).store_unaligned(dstp + ft_rev(kk, K) * W);
        });
    } else {
        static_assert(K <= 4 && C <= 4, "the two-source output merge set is (K, C) in "
                                        "{(2, 2), (2, 4), (4, 2)}");
        poet::static_for<0, K>([&](auto O) {
            constexpr std::size_t o = O;
            constexpr auto pr = ft_out_srcs<o, K, C>();
            const V m = xsimd::shuffle(
                d[pr.first], d[pr.second],
                xsimd::make_batch_constant<index, ft_outm<o, K, C>, arch>());
            (m * f).store_unaligned(dstp + o * W);
        });
    }
}

// Two rows braided in one register (4N == W): masked half loads and one merge in, the same
// intra network (the maps are row-local), then rev2-within-half, one store of each half
// with the scale folded. Strides are fp elements between the two braided rows.
template<unsigned N, typename T, typename V, bool Forward>
ADM_ALWAYS_INLINE void flat_tiny2_apply(const T* ADM_RESTRICT srcp, T* ADM_RESTRICT dstp,
                                        std::size_t in_fp, std::size_t out_fp, V f) {
    constexpr std::size_t W = V::size;
    static_assert(4u * N == W, "braid: one register holds exactly two rows");
    static_assert(N == 4, "braid derived for N = 4 (two per-row intra stages)");
    using index = xsimd::as_unsigned_integer_t<T>;
    using arch = typename V::arch_type;
    const auto m = xsimd::make_batch_bool_constant<T, lane_lt<2u * N>, arch>();
    const V a = V::load(srcp, m, xsimd::unaligned_mode{});
    const V b = V::load(srcp + in_fp, m, xsimd::unaligned_mode{});
    V d = xsimd::shuffle(a, b, xsimd::make_batch_constant<index, ft_braid_ld<N>, arch>());
    ft_intra_all<N / 2, 1, Forward, T, V>(&d);
    d = xsimd::swizzle(d, xsimd::make_batch_constant<index, ft_rev_lane2, arch>()) * f;
    d.store(dstp, m, xsimd::unaligned_mode{});
    const V h = xsimd::swizzle(d, xsimd::make_batch_constant<index, ft_braid_hi<N>, arch>());
    h.store(dstp + out_fp, m, xsimd::unaligned_mode{});
}

}
}

#include "undef_macros.hpp"
