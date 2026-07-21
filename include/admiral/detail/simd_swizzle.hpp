#pragma once

#include <cstddef>

#include <xsimd/xsimd.hpp>

#include "macros.hpp"  // ADM_ALWAYS_INLINE, ADM_RESTRICT (undef'd at end of header)

namespace admiral {
namespace detail {

// ----------------------------------------------------------------------------
// SIMD AoS<->SoA boundary swizzle at the DIF pass boundaries.
//
// W consecutive AoS complex values are 2*W contiguous reals [r0,i0,r1,i1,...];
// the fused first/last passes need them as a pair of planar batch<T> and back.
// Both directions are 2-source, compile-time-mask permutes expressed purely with
// xsimd primitives, so the ISA-optimal lowering is chosen by xsimd dispatch with
// no target intrinsics here:
//   deinterleave (AoS -> SoA): a 2-source shuffle per plane (even lanes -> re,
//     odd lanes -> im). Lowers to 1 vpermt2ps/pd on AVX-512, 1 shufps on SSE,
//     ~2 ops on AVX2 -- vs the 2*log2(W) shuffle-port ops of the zip-round
//     identity it replaces.
//   interleave (SoA -> AoS): zip_lo/zip_hi ARE the perfect shuffle (one pair for
//     any W); on AVX-512 each is a single vpermt2ps (xsimd upstream fix), on
//     AVX2/SSE a vunpck(+vperm)/unpck.

// make_batch_constant generators: lane i of the result reads source index get(i)
// from the concatenated lo||hi (indices 0..W-1 -> lo, W..2W-1 -> hi).
struct aos_even_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i; }
};
struct aos_odd_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i + 1; }
};

// Load W contiguous AoS complex from `src` (as 2*W reals) into planar re/im batches.
// Batch type is deduced so width-adaptive callers (codelet_apply tail) can pass a
// narrower-than-native sized batch.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_deinterleave(const T* ADM_RESTRICT src, Batch& re, Batch& im) {
    using arch = typename Batch::arch_type;
    using index = xsimd::as_unsigned_integer_t<T>;
    constexpr std::size_t W = Batch::size;
    const Batch lo = Batch::load_unaligned(src);
    const Batch hi = Batch::load_unaligned(src + W);
    re = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<index, aos_even_lane, arch>());
    im = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<index, aos_odd_lane, arch>());
}

// Store planar re/im batches as W contiguous AoS complex (2*W reals) to `dst`.
// Batch type is deduced so width-adaptive callers (dif_pass_last) can pass a
// narrower-than-native sized batch.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    xsimd::zip_lo(re, im).store_unaligned(dst);
    xsimd::zip_hi(re, im).store_unaligned(dst + W);
}

// Store only the FIRST R (1 <= R <= W) of the W planar complex as AoS (2*R
// reals) — the partial-block store of the dif_pass_last masked tail. R == W is
// the unmasked pair above. R < W uses compile-time prefix masks: contiguous
// constant masks lower to plain moves, e.g. R=3 at W=4 f32 is movups+movsd
// rather than a vmaskmov.
template<std::size_t R, typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_prefix(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    static_assert(R >= 1 && R <= W);
    using arch = typename Batch::arch_type;
    if constexpr (R == W) {
        aos_interleave<T, Batch>(dst, re, im);
    } else if constexpr (2 * R <= W) {
        struct pfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i < 2 * R; }
        };
        xsimd::zip_lo(re, im).store(dst, xsimd::make_batch_bool_constant<T, pfx, arch>(),
                                    xsimd::unaligned_mode{});
    } else {
        struct pfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i < 2 * R - W; }
        };
        xsimd::zip_lo(re, im).store_unaligned(dst);
        xsimd::zip_hi(re, im).store(dst + W, xsimd::make_batch_bool_constant<T, pfx, arch>(),
                                    xsimd::unaligned_mode{});
    }
}

// Store only complex lanes [m0, m1) (0 <= m0 <= m1 <= W) of the interleaved
// (re, im) block to `dst` (dst is the T* for column 0 of the block; lane i lands
// at dst + 2*i). Used by the column last pass to write the misaligned head and
// ragged tail of a tile with the SAME vector arithmetic as the aligned bulk
// (a scalar butterfly would contract FMAs differently and break nthreads=1-vs-N
// bit identity), while touching each output column exactly once (no re-stores).
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_window(T* ADM_RESTRICT dst, Batch re, Batch im,
                                             std::size_t m0, std::size_t m1) {
    constexpr std::size_t W = Batch::size;
    if (m0 >= m1) return;
    if (m0 == 0 && m1 == W) { aos_interleave<T, Batch>(dst, re, im); return; }
    const Batch lo = xsimd::zip_lo(re, im);
    const Batch hi = xsimd::zip_hi(re, im);
    alignas(sizeof(T) * W) T idx[W];
    for (std::size_t i = 0; i < W; ++i) idx[i] = static_cast<T>(i);
    const Batch iv = Batch::load_aligned(idx);
    const std::size_t r0 = 2 * m0, r1 = 2 * m1;  // real-lane half-open range [r0, r1)
    if (r0 < W) {                                // part lands in the low half [0, W)
        const auto m = (iv >= Batch(static_cast<T>(r0))) &
                       (iv < Batch(static_cast<T>(r1 < W ? r1 : W)));
        lo.store(dst, m, xsimd::unaligned_mode{});
    }
    if (r1 > W) {                                // part lands in the high half [W, 2W)
        const std::size_t h0 = r0 > W ? r0 - W : 0;
        const auto m = (iv >= Batch(static_cast<T>(h0))) &
                       (iv < Batch(static_cast<T>(r1 - W)));
        hi.store(dst + W, m, xsimd::unaligned_mode{});
    }
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
