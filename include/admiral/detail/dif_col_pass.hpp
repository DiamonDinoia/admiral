#pragma once

// ============================================================================
// DIF (Gentleman-Sande) column passes: batched-along-stride analogue of
// dif_passes.hpp. Used by the N-D row-column driver for every non-innermost
// transform axis. Vectorizes over contiguous column lane c in [0,B) with a
// broadcast scalar twiddle (axis pos a only, not column).
//
// Working-buffer layout (planar SoA, axis_extent * B elements):
//   element (axis pos p, column c) at index p * B + c.
//   Input:  p = a + ido*(j + IP*b)   (j radix, b group, a in [0,ido))
//   Output: p = a + ido*(b + l1*k)
//
// AoS boundary (first/last/fused) reads/writes std::complex<T>* with
// axis_stride between consecutive axis positions; aos_deinterleave /
// aos_interleave applies identically to the 1D fused passes.
// All radix math reused via dif_butterfly<T,Fwd,IP> (V-generic); the last pass
// uses dif_butterfly_terminal, which prefers the PFA where emit is a bare store.
// ============================================================================

#include <array>  // lane_prefix_mask lane-index sequence
#include <bit>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <poet/poet.hpp>  // poet::static_for (runtime tail width -> compile-time mask)
#include "simd.hpp"

#include "butterfly.hpp"      // dif_butterfly
#include "cache.hpp"          // kCacheLine
#include "simd_swizzle.hpp"   // aos_deinterleave / aos_interleave, sized_cover
#include "macros.hpp"         // ADM_ALWAYS_INLINE / ADM_NOINLINE / ADM_FLATTEN

namespace admiral {
namespace detail {

// Sub-batch column tails. On a long strided axis the L2 budget drives B BELOW W, where
// the vector loop `c + W <= B` never runs -- not a thin remainder.
//
// Covered by sized_cover (simd_swizzle.hpp): exact-width pieces, widest first, plus one
// backward-aligned full-width piece where that beats narrowing. No runtime mask -- the
// same mechanism and gate as the row driver's small-ido pass, out here in the column
// index.
//
// The tail gets its OWN loop nest rather than sharing the bulk one: its start column is
// loop-invariant, and a second body inside the bulk nest degrades the bulk loop's codegen
// even at widths where the tail is provably dead. Two nests keep the bulk literal and
// cost a second pass over the tile only when B % W != 0.
//
// Each piece is a free function template with PW as a template parameter, NOT a generic
// lambda in the pass body: gcc 14.2 ICEs on an alias template instantiated from a
// generic lambda's own parameter at this instantiation depth.

// One radix-IP butterfly over the PW contiguous columns at c: planar in, planar out.
template<typename T, std::size_t IP, std::size_t PW>
ADM_ALWAYS_INLINE void dif_col_piece(const T* ccre, const T* ccim,
                                     T* chre, T* chim,
                                     std::size_t l1, std::size_t ido, std::size_t B,
                                     std::size_t a, std::size_t b, std::size_t c,
                                     const T* twre, const T* twim) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = a + ido * (j + IP * b);
        tr[j] = load_piece<T, PW>(ccre + p * B + c);
        ti[j] = load_piece<T, PW>(ccim + p * B + c);
    }
    dif_butterfly<T, IP, V>(tr, ti, [&](const auto k, V sr, V si) {
        const std::size_t p = a + ido * (b + l1 * k);
        if constexpr (k > 0u) {
            const V owr(twre[(k - 1u) * ido + a]);
            const V owi(twim[(k - 1u) * ido + a]);
            // Same expression form as the full-width bulk block, so a column computed
            // either way contracts identically (nthreads=1-vs-N bit identity).
            store_piece<T, PW>(chre + p * B + c, owr * sr - owi * si);
            store_piece<T, PW>(chim + p * B + c, owr * si + owi * sr);
        } else {
            store_piece<T, PW>(chre + p * B + c, sr);
            store_piece<T, PW>(chim + p * B + c, si);
        }
    });
}

// [c, B) in ONE full-width masked butterfly. Planar both sides, so a lane mask is the
// whole story -- no swizzle to widen. Whichever of this and the piece cover needs fewer
// ops wins; the crossover is "does the cover finish in one piece" (one_piece_cover below
// picks per call).
//
// Which mask FORM is cheaper is an ISA property, not a winner:
//   AVX-512: a k-mask is a native operand, so building it at runtime is a few
//   loop-invariant ops for the whole pass, whereas a constant TailN needs a W-1 arm
//   chain that grows the shared tail function.
//   AVX2: no mask register, so a constant mask lowers to plain narrow moves rather than
//   vmaskmov, and the arm chain is short.
// 32 vector registers is this codebase's wide-ISA proxy (see dif_wide_radices); on x86
// it coincides with having mask registers.
inline constexpr bool kRuntimeTailMask = poet::vector_register_count() >= 32;

// True when sized_cover covers [first, last) with a single piece: the remainder is itself
// an available width, or the backward-aligned overlap applies.
// Precondition: the remainder is a sub-batch one, i.e. last - first < W. Callers pass a
// W-aligned first; the mask is 64-bit, so the shift is in range for any shipped width.
template<typename T>
[[nodiscard]] constexpr bool one_piece_cover(std::size_t first, std::size_t last) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t rem = last - first;
    assert(rem < W);
    return rem == 0 || ((kPieceWidths<T> >> rem) & 1u) != 0u || (last >= W && 2 * rem >= W);
}

// Mask generic over its form: xsimd's masked load/store take a batch_bool_constant just as
// well as a batch_bool, so both gate arms share this one body.
// FLATTEN, same reason as dif_pass_last_block: without it gcc-14 emits an out-of-line
// radix_sym_dft for this butterfly, then folds the BULK pass's identical butterfly onto
// it and calls it, so cells that never enter the tail pay for it.
template<typename T, std::size_t IP>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_masked(const T* ccre,
                                            const T* ccim,
                                            T* chre, T* chim,
                                            std::size_t l1, std::size_t ido, std::size_t B,
                                            std::size_t a, std::size_t b, std::size_t c,
                                            const T* twre,
                                            const T* twim, const auto m) {
    using batch = xsimd::batch<T>;
    batch tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = a + ido * (j + IP * b);
        tr[j] = batch::load(ccre + p * B + c, m, xsimd::unaligned_mode{});
        ti[j] = batch::load(ccim + p * B + c, m, xsimd::unaligned_mode{});
    }
    dif_butterfly<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
        const std::size_t p = a + ido * (b + l1 * k);
        if constexpr (k > 0u) {
            const batch owr(twre[(k - 1u) * ido + a]);
            const batch owi(twim[(k - 1u) * ido + a]);
            (owr * sr - owi * si).store(chre + p * B + c, m, xsimd::unaligned_mode{});
            (owr * si + owi * sr).store(chim + p * B + c, m, xsimd::unaligned_mode{});
        } else {
            sr.store(chre + p * B + c, m, xsimd::unaligned_mode{});
            si.store(chim + p * B + c, m, xsimd::unaligned_mode{});
        }
    });
}

// Same, reading AoS (axis_stride between axis positions) and writing planar.
template<typename T, bool Forward, std::size_t IP, std::size_t PW>
ADM_ALWAYS_INLINE void dif_col_piece_first(const std::complex<T>* data,
                                           std::size_t axis_stride,
                                           T* chre, T* chim,
                                           std::size_t l1, std::size_t ido, std::size_t B,
                                           std::size_t a, std::size_t b, std::size_t c,
                                           const T* twre,
                                           const T* twim) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = a + ido * (j + IP * b);
        auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
        aos_deinterleave_piece<T, PW>(reinterpret_cast<const T*>(data + p * axis_stride + c),
                                      dr, di);
    }
    dif_butterfly<T, IP, V>(tr, ti, [&](const auto k, V sr, V si) {
        const std::size_t p = a + ido * (b + l1 * k);
        if constexpr (k > 0u) {
            const V owr(twre[(k - 1u) * ido + a]);
            const V owi(twim[(k - 1u) * ido + a]);
            store_piece<T, PW>(chre + p * B + c, owr * sr - owi * si);
            store_piece<T, PW>(chim + p * B + c, owr * si + owi * sr);
        } else {
            store_piece<T, PW>(chre + p * B + c, sr);
            store_piece<T, PW>(chim + p * B + c, si);
        }
    });
}

// Masked twin of dif_col_piece_first: ONE full-width piece for a sub-batch block, instead of
// a width descent. AoS side masked to 2*rem reals, planar side to rem lanes -- the planar row
// stride is B, so an unmasked store would clobber the next row.
template<typename T, bool Forward, std::size_t IP, bool HiHalf>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_first_masked(
    const std::complex<T>* data, std::size_t axis_stride, T* chre,
    T* chim, std::size_t l1, std::size_t ido, std::size_t B, std::size_t a,
    std::size_t b, std::size_t c, const T* twre, const T* twim,
    const auto m, const auto am) {
    using batch = xsimd::batch<T>;
    batch tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = a + ido * (j + IP * b);
        auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
        aos_deinterleave_masked<HiHalf, T>(reinterpret_cast<const T*>(data + p * axis_stride + c),
                                           dr, di, am);
    }
    dif_butterfly<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
        const std::size_t p = a + ido * (b + l1 * k);
        if constexpr (k > 0u) {
            const batch owr(twre[(k - 1u) * ido + a]);
            const batch owi(twim[(k - 1u) * ido + a]);
            (owr * sr - owi * si).store(chre + p * B + c, m, xsimd::unaligned_mode{});
            (owr * si + owi * sr).store(chim + p * B + c, m, xsimd::unaligned_mode{});
        } else {
            sr.store(chre + p * B + c, m, xsimd::unaligned_mode{});
            si.store(chim + p * B + c, m, xsimd::unaligned_mode{});
        }
    });
}

// Terminal piece, planar in -> AoS out. ido == 1, so no output twiddle.
template<typename T, bool Forward, std::size_t IP, std::size_t PW>
ADM_ALWAYS_INLINE void dif_col_piece_last(const T* ccre, const T* ccim,
                                          std::complex<T>* data,
                                          std::size_t axis_stride, std::size_t l1, std::size_t B,
                                          std::size_t b, std::size_t c, T scale_val) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = j + IP * b;
        tr[j] = load_piece<T, PW>(ccre + p * B + c);
        ti[j] = load_piece<T, PW>(ccim + p * B + c);
    }
    dif_butterfly_terminal<T, IP, V>(tr, ti, [&](const auto k, V sr, V si) {
        T* dst = reinterpret_cast<T*>(data + (b + l1 * k) * axis_stride + c);
        const V sv(scale_val);
        const auto [xr, xi] = plane_vals<Forward>(sr * sv, si * sv);
        aos_interleave_piece<T, PW>(dst, xr, xi);
    });
}

// Masked twin of dif_col_piece_last: planar loads masked (stride B, and the last row would
// otherwise read past the scratch buffer), AoS store masked to 2*B reals.
template<typename T, bool Forward, std::size_t IP, bool HiHalf>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_last_masked(
    const T* ccre, const T* ccim, std::complex<T>* data,
    std::size_t axis_stride, std::size_t l1, std::size_t B, std::size_t b, std::size_t c,
    T scale_val, const auto m, const auto am) {
    using batch = xsimd::batch<T>;
    batch tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        const std::size_t p = j + IP * b;
        tr[j] = batch::load(ccre + p * B + c, m, xsimd::unaligned_mode{});
        ti[j] = batch::load(ccim + p * B + c, m, xsimd::unaligned_mode{});
    }
    dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
        T* dst = reinterpret_cast<T*>(data + (b + l1 * k) * axis_stride + c);
        const batch sv(scale_val);
        const auto [xr, xi] = plane_vals<Forward>(sr * sv, si * sv);
        aos_interleave_masked<HiHalf, T>(dst, xr, xi, am);
    });
}

// Terminal piece, AoS in place. l1 == 1 and ido == 1, so no output twiddle.
template<typename T, bool Forward, std::size_t IP, std::size_t PW>
ADM_ALWAYS_INLINE void dif_col_piece_fused(std::complex<T>* data,
                                           std::size_t axis_stride, std::size_t l1,
                                           std::size_t b, std::size_t c, T scale_val) {
    using V = sized_piece_t<T, PW>;
    V tr[IP], ti[IP];
    for (std::size_t j = 0; j < IP; ++j) {
        auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
        aos_deinterleave_piece<T, PW>(
            reinterpret_cast<const T*>(data + (j + IP * b) * axis_stride + c), dr, di);
    }
    dif_butterfly_terminal<T, IP, V>(tr, ti, [&](const auto k, V sr, V si) {
        T* dst = reinterpret_cast<T*>(data + (b + l1 * k) * axis_stride + c);
        const V sv(scale_val);
        const auto [xr, xi] = plane_vals<Forward>(sr * sv, si * sv);
        aos_interleave_piece<T, PW>(dst, xr, xi);
    });
}

// Rows of the compile-time-mask tail, one instantiation per width. NOINLINE for the reason
// the whole tail is outlined: W-1 copies of this nest inside dif_col_tail regress the cover
// path too. The runtime-mask arm has no wrapper at all -- even an ALWAYS_INLINE one pushes
// dif_col_pass over gcc's inlining budget, which then outlines the bulk butterfly.
template<typename T, std::size_t IP, std::size_t TailN>
ADM_NOINLINE void dif_col_masked_rows_ct(const T* ccre,
                                         const T* ccim, T* chre,
                                         T* chim, std::size_t l1, std::size_t ido,
                                         std::size_t B, std::size_t c,
                                         const T* twre,
                                         const T* twim) {
    using arch = typename xsimd::batch<T>::arch_type;
    static_assert(TailN >= 1 && TailN < xsimd::batch<T>::size);
    constexpr auto m = xsimd::make_batch_bool_constant<T, lane_lt<TailN>, arch>();
    for (std::size_t b = 0; b < l1; ++b)
        for (std::size_t a = 0; a < ido; ++a)
            dif_col_piece_masked<T, IP>(ccre, ccim, chre, chim, l1, ido, B, a, b, c,
                                                 twre, twim, m);
}

// Masked arm of the first pass, own frame for the reason the whole tail is outlined. HiHalf is
// fixed per arm so the body carries no branch.
template<typename T, bool Forward, std::size_t IP, bool HiHalf>
ADM_NOINLINE void dif_col_masked_rows_first(const std::complex<T>* data,
                                            std::size_t axis_stride, T* chre,
                                            T* chim, std::size_t l1, std::size_t ido,
                                            std::size_t B, std::size_t cfull,
                                            const T* twre,
                                            const T* twim) {
    const std::size_t rem = B - cfull;
    const auto m = lane_prefix_mask<T>(rem);
    const auto am = make_aos_prefix_masks<T>(rem);
    for (std::size_t b = 0; b < l1; ++b)
        for (std::size_t a = 0; a < ido; ++a)
            dif_col_piece_first_masked<T, Forward, IP, HiHalf>(data, axis_stride, chre, chim, l1,
                                                              ido, B, a, b, cfull, twre, twim, m,
                                                              am);
}

// Compile-time-mask twins of the two nests above, for the arm where a constant mask is the
// cheaper form. TailN fixes HiHalf too, so there is one arm per width, not two.
template<typename T, bool Forward, std::size_t IP, std::size_t TailN>
ADM_NOINLINE void dif_col_masked_rows_first_ct(const std::complex<T>* data,
                                              std::size_t axis_stride, T* chre,
                                              T* chim, std::size_t l1,
                                              std::size_t ido, std::size_t B, std::size_t cfull,
                                              const T* twre,
                                              const T* twim) {
    using arch = typename xsimd::batch<T>::arch_type;
    constexpr std::size_t W = xsimd::batch<T>::size;
    static_assert(TailN >= 1 && TailN < W);
    constexpr auto m = xsimd::make_batch_bool_constant<T, lane_lt<TailN>, arch>();
    for (std::size_t b = 0; b < l1; ++b)
        for (std::size_t a = 0; a < ido; ++a)
            dif_col_piece_first_masked<T, Forward, IP, (2 * TailN > W)>(
                data, axis_stride, chre, chim, l1, ido, B, a, b, cfull, twre, twim, m,
                aos_ct_masks<TailN, T>{});
}

template<typename T, bool Forward, std::size_t IP, std::size_t TailN>
ADM_NOINLINE void dif_col_masked_rows_last_ct(const T* ccre,
                                              const T* ccim,
                                              std::complex<T>* data,
                                              std::size_t axis_stride, std::size_t l1,
                                              std::size_t B, T scale_val) {
    using arch = typename xsimd::batch<T>::arch_type;
    constexpr std::size_t W = xsimd::batch<T>::size;
    static_assert(TailN >= 1 && TailN < W);
    constexpr auto m = xsimd::make_batch_bool_constant<T, lane_lt<TailN>, arch>();
    for (std::size_t b = 0; b < l1; ++b)
        dif_col_piece_last_masked<T, Forward, IP, (2 * TailN > W)>(
            ccre, ccim, data, axis_stride, l1, B, b, 0, scale_val, m, aos_ct_masks<TailN, T>{});
}

// Runs f with the sub-batch width as a compile-time constant, over the widths that are NOT
// themselves piece widths -- the only ones a cover would have to descend for, so the arm
// chain is far shorter than W-1. False when no arm matches, so callers keep their cover as
// the fallback rather than silently skipping the tail.
template<typename T, typename F>
[[nodiscard]] ADM_ALWAYS_INLINE bool dispatch_masked_width(std::size_t rem, const F& f) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    bool hit = false;
    poet::static_for<1, std::ptrdiff_t{W}>([&](auto R) {
        constexpr std::size_t Rn{R.value};
        if constexpr (((kPieceWidths<T> >> Rn) & 1u) == 0u)
            if (!hit && Rn == rem) {
                f(std::integral_constant<std::size_t, Rn>{});
                hit = true;
            }
    });
    return hit;
}

// ----------------------------------------------------------------------------
// Column-tail nests, one per pass. ADM_NOINLINE: even in its own loop nest the
// tail degrades the bulk loop's codegen at widths where it provably never runs
// (B % W == 0 or B < W). Outlining costs one call per pass invocation and keeps
// the bulk literal.
// ----------------------------------------------------------------------------

// Runs f with the piece width as a compile-time constant when [cfull, cfull+rem) is
// covered by ONE piece of an available width >= 2, else returns false. Width 1 is
// excluded: a specialised scalar piece does not beat the generic body. Widths >= W are
// unreachable: every caller enters a tail only for a sub-batch remainder -- cfull != B
// gives rem = B % W, and the _last pass gates on B < W.
template<typename T, typename F>
[[nodiscard]] ADM_ALWAYS_INLINE bool dispatch_one_piece(std::size_t rem, F&& f) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    assert(rem < W);
    bool hit = false;
    poet::static_for<1, std::ptrdiff_t{std::bit_width(W)} - 1>([&](auto E) {
        constexpr std::size_t PW = std::size_t{1} << E.value;
        if constexpr (((kPieceWidths<T> >> PW) & 1u) != 0u) {
            if (!hit && rem == PW) {
                f(std::integral_constant<std::size_t, PW>{});
                hit = true;
            }
        }
    });
    return hit;
}

template<typename T, std::size_t IP, std::size_t PW>
ADM_NOINLINE void dif_col_tail_one_piece(const T* ccre, const T* ccim,
                                         T* chre, T* chim,
                                         std::size_t l1, std::size_t ido, std::size_t B,
                                         std::size_t c,
                                         const T* twre, const T* twim) {
    for (std::size_t b = 0; b < l1; ++b)
        for (std::size_t a = 0; a < ido; ++a)
            dif_col_piece<T, IP, PW>(ccre, ccim, chre, chim, l1, ido, B, a, b, c, twre,
                                              twim);
}

template<typename T, bool Forward, std::size_t IP, std::size_t PW>
ADM_NOINLINE void dif_col_tail_first_one_piece(const std::complex<T>* data,
                                               std::size_t axis_stride, T* chre,
                                               T* chim, std::size_t l1,
                                               std::size_t ido, std::size_t B, std::size_t c,
                                               const T* twre,
                                               const T* twim) {
    for (std::size_t b = 0; b < l1; ++b)
        for (std::size_t a = 0; a < ido; ++a)
            dif_col_piece_first<T, Forward, IP, PW>(data, axis_stride, chre, chim, l1, ido, B, a,
                                                   b, c, twre, twim);
}

template<typename T, bool Forward, std::size_t IP, std::size_t PW>
ADM_NOINLINE void dif_col_tail_last_one_piece(const T* ccre,
                                              const T* ccim,
                                              std::complex<T>* data,
                                              std::size_t axis_stride, std::size_t l1,
                                              std::size_t B, std::size_t c, T scale_val) {
    for (std::size_t b = 0; b < l1; ++b)
        dif_col_piece_last<T, Forward, IP, PW>(ccre, ccim, data, axis_stride, l1, B, b, c,
                                               scale_val);
}

template<typename T, std::size_t IP>
ADM_NOINLINE void dif_col_tail(const T* ccre, const T* ccim,
                               T* chre, T* chim,
                               std::size_t l1, std::size_t ido, std::size_t B,
                               std::size_t cfull,
                               const T* twre, const T* twim) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    // Out of place, so the backward-aligned overlap is legal: the recomputed columns
    // are rewritten with identical values.
    // Cover outermost: the width descent is per pass, not per (b, a) -- see the note
    // on dif_col_tail_last_general.
    if (!one_piece_cover<T>(cfull, B)) {
        if constexpr (kRuntimeTailMask) {
            const auto m = lane_prefix_mask<T>(B - cfull);
            for (std::size_t b = 0; b < l1; ++b)
                for (std::size_t a = 0; a < ido; ++a)
                    dif_col_piece_masked<T, IP>(ccre, ccim, chre, chim, l1, ido, B, a,
                                                         b, cfull, twre, twim, m);
            return;
        } else if (dispatch_masked_width<T>(B - cfull, [&](auto R) {
                       dif_col_masked_rows_ct<T, IP, R.value>(ccre, ccim, chre, chim, l1,
                                                                     ido, B, cfull, twre, twim);
                   })) {
            return;
        }
    }
    sized_cover<T, W, true>(cfull, B, [&](auto PWc, std::size_t c) {
        for (std::size_t b = 0; b < l1; ++b)
            for (std::size_t a = 0; a < ido; ++a)
                dif_col_piece<T, IP, PWc.value>(ccre, ccim, chre, chim, l1, ido, B,
                                                         a, b, c, twre, twim);
    });
}

template<typename T, bool Forward, std::size_t IP>
ADM_NOINLINE void dif_col_tail_first(const std::complex<T>* data,
                                     std::size_t axis_stride,
                                     T* chre, T* chim,
                                     std::size_t l1, std::size_t ido, std::size_t B,
                                     std::size_t cfull,
                                     const T* twre, const T* twim) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    // Same two-arm shape as dif_col_tail: a block that is not itself a piece width would
    // otherwise descend to scalar (f32 has no 2-wide batch, so 3 = 1+1+1 is three full row
    // passes). One masked full-width piece replaces the descent, in whichever mask form the
    // ISA prefers.
    if (!one_piece_cover<T>(cfull, B)) {
        if constexpr (kRuntimeTailMask) {
            if (2 * (B - cfull) > W)
                dif_col_masked_rows_first<T, Forward, IP, true>(data, axis_stride, chre, chim, l1,
                                                               ido, B, cfull, twre, twim);
            else
                dif_col_masked_rows_first<T, Forward, IP, false>(data, axis_stride, chre, chim, l1,
                                                                ido, B, cfull, twre, twim);
            return;
        } else if (dispatch_masked_width<T>(B - cfull, [&](auto R) {
                       dif_col_masked_rows_first_ct<T, Forward, IP, R.value>(
                           data, axis_stride, chre, chim, l1, ido, B, cfull, twre, twim);
                   })) {
            return;
        }
    }
    // Reads AoS input, writes a separate planar buffer, so the overlap is legal.
    sized_cover<T, W, true>(cfull, B, [&](auto PWc, std::size_t c) {
        for (std::size_t b = 0; b < l1; ++b)
            for (std::size_t a = 0; a < ido; ++a)
                dif_col_piece_first<T, Forward, IP, PWc.value>(data, axis_stride, chre, chim,
                                                              l1, ido, B, a, b, c, twre, twim);
    });
}

// Whole tile narrower than a vector, so there is no bulk and no store-align peel.
template<typename T, bool Forward, std::size_t IP>
ADM_NOINLINE void dif_col_tail_last_general(const T* ccre,
                                           const T* ccim,
                                           std::complex<T>* data,
                                           std::size_t axis_stride, std::size_t l1, std::size_t B,
                                           T scale_val) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    // Cover outside the row loop: the width descent (one guard per candidate width) is
    // then paid once per pass instead of once per row.
    sized_cover<T, W, true>(0, B, [&](auto PWc, std::size_t c) {
        for (std::size_t b = 0; b < l1; ++b)
            dif_col_piece_last<T, Forward, IP, PWc.value>(ccre, ccim, data, axis_stride,
                                                          l1, B, b, c, scale_val);
    });
}

template<typename T, bool Forward, std::size_t IP, bool HiHalf>
ADM_NOINLINE void dif_col_tail_last_masked(const T* ccre, const T* ccim,
                                           std::complex<T>* data,
                                           std::size_t axis_stride, std::size_t l1, std::size_t B,
                                           T scale_val) {
    const auto m = lane_prefix_mask<T>(B);
    const auto am = make_aos_prefix_masks<T>(B);
    for (std::size_t b = 0; b < l1; ++b)
        dif_col_piece_last_masked<T, Forward, IP, HiHalf>(ccre, ccim, data, axis_stride, l1,
                                                          B, b, 0, scale_val, m, am);
}

// Three frames on purpose. The arm chain shares a stack frame with nothing else:
// dif_col_pass_last is register-tight from the AoS store-align peel, so folding the chain
// into the caller -- or placing it in front of the general cover -- regresses cells whose
// executed path never reaches it. The extra call on the general path amortises over l1 rows.
template<typename T, bool Forward, std::size_t IP>
ADM_NOINLINE void dif_col_tail_last(const T* ccre, const T* ccim,
                                    std::complex<T>* data, std::size_t axis_stride,
                                    std::size_t l1, std::size_t B, T scale_val) {
    if (dispatch_one_piece<T>(B, [&](auto PW) {
            dif_col_tail_last_one_piece<T, Forward, IP, PW.value>(ccre, ccim, data,
                                                                  axis_stride, l1, B, 0,
                                                                  scale_val);
        }))
        return;
    // B == 1 keeps the scalar piece: one move per row beats a masked vector.
    if (B > 1) {
        if constexpr (kRuntimeTailMask) {
            if (2 * B > xsimd::batch<T>::size)
                dif_col_tail_last_masked<T, Forward, IP, true>(ccre, ccim, data, axis_stride,
                                                               l1, B, scale_val);
            else
                dif_col_tail_last_masked<T, Forward, IP, false>(ccre, ccim, data,
                                                                axis_stride, l1, B,
                                                                scale_val);
            return;
        } else if (dispatch_masked_width<T>(B, [&](auto R) {
                       dif_col_masked_rows_last_ct<T, Forward, IP, R.value>(
                           ccre, ccim, data, axis_stride, l1, B, scale_val);
                   })) {
            return;
        }
    }
    dif_col_tail_last_general<T, Forward, IP>(ccre, ccim, data, axis_stride, l1, B,
                                              scale_val);
}

template<typename T, bool Forward, std::size_t IP>
ADM_NOINLINE void dif_col_tail_fused(std::complex<T>* data, std::size_t axis_stride,
                                     std::size_t l1, std::size_t B, std::size_t cfull,
                                     T scale_val) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    // Overlap disabled: this pass is IN PLACE, so a backward-aligned piece would
    // re-read the columns the bulk loop already overwrote. Exact-width pieces only.
    // Row loop OUTSIDE the cover here, unlike the other three passes: an in-place piece
    // cannot keep its loads live across rows the way a separate-buffer piece can, and
    // hoisting the cover regresses it.
    for (std::size_t b = 0; b < l1; ++b)
        sized_cover<T, W, false>(cfull, B, [&](auto PWc, std::size_t c) {
            dif_col_piece_fused<T, Forward, IP, PWc.value>(data, axis_stride, l1, b, c,
                                                           scale_val);
        });
}

// Generic vectorized column DIF pass: planar SoA in -> planar SoA out.
// Vectorizes over the contiguous column lane c in [0,B); broadcast twiddle.
template<typename T, std::size_t IP>
void dif_col_pass(const T* ccre, const T* ccim,
                  T* chre, T* chim,
                  std::size_t l1, std::size_t ido, std::size_t B,
                  const T* twre, const T* twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t cfull = B - B % W;

    if (B >= W) {
        for (std::size_t b = 0; b < l1; ++b) {
            for (std::size_t a = 0; a < ido; ++a) {
                for (std::size_t c = 0; c < cfull; c += W) {
                    batch tr[IP], ti[IP];
                    for (std::size_t j = 0; j < IP; ++j) {
                        const std::size_t p = a + ido * (j + IP * b);
                        tr[j] = batch::load_unaligned(ccre + p * B + c);
                        ti[j] = batch::load_unaligned(ccim + p * B + c);
                    }
                    dif_butterfly<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
                        const std::size_t p = a + ido * (b + l1 * k);
                        if constexpr (k > 0u) {
                            const batch owr(twre[(k - 1u) * ido + a]);  // broadcast scalar
                            const batch owi(twim[(k - 1u) * ido + a]);
                            (owr * sr - owi * si).store_unaligned(chre + p * B + c);
                            (owr * si + owi * sr).store_unaligned(chim + p * B + c);
                        } else {
                            sr.store_unaligned(chre + p * B + c);
                            si.store_unaligned(chim + p * B + c);
                        }
                    });
                }
            }
        }
    }
    if (cfull != B)
        if (!dispatch_one_piece<T>(B - cfull, [&](auto PW) {
                dif_col_tail_one_piece<T, IP, PW.value>(ccre, ccim, chre, chim, l1, ido,
                                                                B, cfull, twre, twim);
            }))
            dif_col_tail<T, IP>(ccre, ccim, chre, chim, l1, ido, B, cfull, twre, twim);
}

// First pass: reads AoS std::complex<T>* (axis_stride between axis positions),
// writes planar SoA. l1 == 1 for the first pass; ido >= 2 (single-factor axes
// take the fused path), so the output twiddle is always present.
template<typename T, bool Forward, std::size_t IP>
void dif_col_pass_first(const std::complex<T>* data, std::size_t axis_stride,
                        T* chre, T* chim,
                        std::size_t l1, std::size_t ido, std::size_t B,
                        const T* twre, const T* twim) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;
    const std::size_t cfull = B - B % W;

    if (B >= W) {
        for (std::size_t b = 0; b < l1; ++b) {
            for (std::size_t a = 0; a < ido; ++a) {
                for (std::size_t c = 0; c < cfull; c += W) {
                    batch tr[IP], ti[IP];
                    for (std::size_t j = 0; j < IP; ++j) {
                        const std::size_t p = a + ido * (j + IP * b);
                        const T* src = reinterpret_cast<const T*>(data + p * axis_stride + c);
                        auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                        aos_deinterleave<T>(src, dr, di);
                    }
                    dif_butterfly<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
                        const std::size_t p = a + ido * (b + l1 * k);
                        if constexpr (k > 0u) {
                            const batch owr(twre[(k - 1u) * ido + a]);
                            const batch owi(twim[(k - 1u) * ido + a]);
                            (owr * sr - owi * si).store_unaligned(chre + p * B + c);
                            (owr * si + owi * sr).store_unaligned(chim + p * B + c);
                        } else {
                            sr.store_unaligned(chre + p * B + c);
                            si.store_unaligned(chim + p * B + c);
                        }
                    });
                }
            }
        }
    }
    if (cfull != B)
        if (!dispatch_one_piece<T>(B - cfull, [&](auto PW) {
                dif_col_tail_first_one_piece<T, Forward, IP, PW.value>(data, axis_stride, chre,
                                                                      chim, l1, ido, B, cfull,
                                                                      twre, twim);
            }))
            dif_col_tail_first<T, Forward, IP>(data, axis_stride, chre, chim, l1, ido, B, cfull,
                                              twre, twim);
}

// Last pass: reads planar SoA, writes AoS std::complex<T>* (axis_stride between
// axis positions). ido == 1 always for the last pass (l1 == axis_extent/IP), so
// the output twiddle is W^0 = 1 (twre/twim unused).
template<typename T, bool Forward, std::size_t IP>
void dif_col_pass_last(const T* ccre, const T* ccim,
                       std::complex<T>* data, std::size_t axis_stride,
                       std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                       [[maybe_unused]] const T* twre,
                       [[maybe_unused]] const T* twim, T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;

    // Invariant: the last DIF pass always has ido == 1: no output twiddle.
    assert(ido == 1);

    if (B < W) {
        dif_col_tail_last<T, Forward, IP>(ccre, ccim, data, axis_stride, l1, B, scale_val);
        return;
    }

    // Peel leading columns to align the scattered AoS output stores to cache lines
    // (see aos_store_align_peel). Invariant across b (depends only on data, stride).
    const std::size_t peel = aos_store_align_peel<T>(data, axis_stride, B);

    for (std::size_t b = 0; b < l1; ++b) {
        // Butterfly W columns at c, then store via `store` (dst, re, im). The store
        // is a template functor so the aligned bulk (plain aos_interleave) and the
        // head/tail (compile-time prefix/suffix dispatch) are SEPARATE instantiations:
        // the hot bulk body carries none of the partial-store dispatch. Each column is
        // written exactly once with identical vector arithmetic -- preserves 1-vs-N-thread
        // bit identity (scalar FMAs contract differently).
        const auto vec_block = [&](std::size_t c, auto store) {
            batch tr[IP], ti[IP];
            for (std::size_t j = 0; j < IP; ++j) {
                const std::size_t p = j + IP * b;
                tr[j] = batch::load_unaligned(ccre + p * B + c);
                ti[j] = batch::load_unaligned(ccim + p * B + c);
            }
            dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
                const std::size_t p = b + l1 * k;
                T* dst = reinterpret_cast<T*>(data + p * axis_stride + c);
                const batch sv(scale_val);
                const auto [xr, xi] = plane_vals<Forward>(sr * sv, si * sv);
                store(dst, xr, xi);
            });
        };
        // Head [0,peel) aligns bulk stores; tail block at B-W covers the remainder.
        // Both in-bounds and disjoint from bulk: each column written exactly once.
        std::size_t c = 0;
        if (peel > 0) {                            // head: store lanes [0,peel)
            vec_block(0, [peel](T* d, batch r, batch i) { aos_interleave_prefix_n<T>(d, r, i, peel); });
            c = peel;
        }
        const auto full_store = [](T* d, batch r, batch i) { aos_interleave<T>(d, r, i); };
        for (; c + W <= B; c += W) vec_block(c, full_store);   // aligned bulk, full block
        if (c < B) {                                          // tail: store lanes [m0,W)
            const std::size_t m0 = c - (B - W);
            vec_block(B - W, [m0](T* d, batch r, batch i) { aos_interleave_suffix_n<T>(d, r, i, m0); });
        }
    }
}

// Single-pass (fused first+last): reads and writes AoS. Reached when the axis
// length factors to a single radix, so l1 == 1 and ido == 1 (twiddle trivial).
// Invariant: ido == 1; the ido>1 twiddle branches are dead and have been removed.
template<typename T, bool Forward, std::size_t IP>
void dif_col_pass_fused(std::complex<T>* data, std::size_t axis_stride,
                        std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                        [[maybe_unused]] const T* twre,
                        [[maybe_unused]] const T* twim, T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;

    // ido == 1 invariant: twiddles are W^0 = 1 (not needed).
    const std::size_t cfull = B - B % W;

    if (B >= W) {
        for (std::size_t b = 0; b < l1; ++b) {
            // a = 0 only (ido == 1).
            for (std::size_t c = 0; c < cfull; c += W) {
                batch tr[IP], ti[IP];
                for (std::size_t j = 0; j < IP; ++j) {
                    const std::size_t p = j + IP * b;  // a==0, ido==1 → a + ido*(j+IP*b) = j+IP*b
                    const T* src = reinterpret_cast<const T*>(data + p * axis_stride + c);
                    auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                    aos_deinterleave<T>(src, dr, di);
                }
                dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
                    const std::size_t p = b + l1 * k;  // a==0, ido==1 → a + ido*(b+l1*k) = b+l1*k
                    T* dst = reinterpret_cast<T*>(data + p * axis_stride + c);
                    const batch sv(scale_val);
                    const auto [xr, xi] = plane_vals<Forward>(sr * sv, si * sv);
                    aos_interleave<T>(dst, xr, xi);
                });
            }
        }
    }
    if (cfull != B)
        dif_col_tail_fused<T, Forward, IP>(data, axis_stride, l1, B, cfull, scale_val);
}

// ----------------------------------------------------------------------------
// Runtime-radix dispatch functors (mirror dif_passes.hpp).
// ----------------------------------------------------------------------------

template<typename T>
inline constexpr auto dif_col_pass_invoke = []<std::size_t IP>(
        const T* ccre, const T* ccim,
        T* chre, T* chim,
        std::size_t l1, std::size_t ido, std::size_t B,
        const T* twre, const T* twim) {
    dif_col_pass<T, IP>(ccre, ccim, chre, chim, l1, ido, B, twre, twim);
};

template<typename T, bool Forward>
inline constexpr auto dif_col_pass_first_invoke = []<std::size_t IP>(
        const std::complex<T>* data, std::size_t axis_stride,
        T* chre, T* chim,
        std::size_t l1, std::size_t ido, std::size_t B,
        const T* twre, const T* twim) {
    dif_col_pass_first<T, Forward, IP>(data, axis_stride, chre, chim, l1, ido, B, twre, twim);
};

template<typename T, bool Forward>
inline constexpr auto dif_col_pass_last_invoke = []<std::size_t IP>(
        const T* ccre, const T* ccim,
        std::complex<T>* data, std::size_t axis_stride,
        std::size_t l1, std::size_t ido, std::size_t B,
        const T* twre, const T* twim, T scale_val) {
    dif_col_pass_last<T, Forward, IP>(ccre, ccim, data, axis_stride, l1, ido, B, twre, twim,
                                      scale_val);
};

template<typename T, bool Forward>
inline constexpr auto dif_col_pass_fused_invoke = []<std::size_t IP>(
        std::complex<T>* data, std::size_t axis_stride,
        std::size_t l1, std::size_t ido, std::size_t B,
        const T* twre, const T* twim, T scale_val) {
    dif_col_pass_fused<T, Forward, IP>(data, axis_stride, l1, ido, B, twre, twim, scale_val);
};

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
