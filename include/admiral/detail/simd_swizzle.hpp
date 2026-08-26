#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>  // std::is_void_v, std::conditional_t, std::integral_constant
#include <tuple>        // std::tie (plane_refs)
#include <utility>      // std::index_sequence (piece-width mask), std::pair

#include <poet/poet.hpp>  // poet::static_for (runtime bound -> compile-time mask dispatch)
#include "cxx_compat.hpp"  // ADM_CONSTEVAL
#include "simd.hpp"

#include "cache.hpp"   // kCacheLine (AoS store peel)
#include "macros.hpp"  // ADM_ALWAYS_INLINE, ADM_RESTRICT (undef'd at end of header)

namespace admiral {
namespace detail {

// ----------------------------------------------------------------------------
// SIMD AoS<->SoA swizzle at DIF pass boundaries.
//
// W consecutive AoS complex = 2*W reals [r0,i0,r1,i1,...]; fused first/last
// passes need them as planar batch<T> pairs and back.
//   deinterleave (AoS->SoA): 2-source shuffle per plane (even->re, odd->im).
//   interleave (SoA->AoS): zip_lo/zip_hi.
// Pure xsimd operations. xsimd dispatch picks the ISA lowering (AVX2 needs a shuffle
// plus a cross-lane permute; no single AVX2 instruction avoids it).

// Permute-index generators for xsimd::make_batch_constant: lane i reads get(i)
// from the concatenated lo||hi (0..W-1 -> lo, W..2W-1 -> hi).
struct aos_even_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i; }
};
struct aos_odd_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i + 1; }
};

// ----------------------------------------------------------------------------
// The engine's direction boundary.
//
// Every DIF butterfly and every SoA->SoA pass computes the FORWARD transform only
// (see butterfly.hpp): the inverse is that same code run in *swapped domain*, where
// the planes hold (im, re). swap(fwd(swap x)) == inv(x); the equivalent
// conj(fwd(conj x)) costs 2N sign flips.
//
// These two are the only place the exchange happens. A boundary pass routes the
// destination pair of its AoS *load* through plane_refs and the value pair of its AoS
// *store* through plane_vals; everything between the two boundaries is direction-free.
// Two forms because a load needs lvalues to write through and a store takes computed
// values.
// ----------------------------------------------------------------------------
template<bool Forward, typename V>
[[nodiscard]] ADM_ALWAYS_INLINE auto plane_refs(V& re, V& im) {
    if constexpr (Forward) return std::tie(re, im);
    else                   return std::tie(im, re);
}

template<bool Forward, typename V>
[[nodiscard]] ADM_ALWAYS_INLINE std::pair<V, V> plane_vals(V re, V im) {
    if constexpr (Forward) return {re, im};
    else                   return {im, re};
}

// Load W contiguous AoS complex (2*W reals) into planar re/im batches.
// Batch is deduced so width-adaptive callers can pass a narrower sized_batch.
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

// Widest xsimd piece width <= N, halving until make_sized_batch_t is non-void.
// E.g. make_sized_batch_t<float,2> is void → f32: 7=4+1+1+1, f64: 7=4+2+1.
// Shared by both DIF drivers: the row driver's small-ido pieces and the column
// driver's sub-batch tail are the same problem in different index dimensions.
// A PW-wide piece also fixes the AoS footprint for free: aos_deinterleave with a
// narrower Batch touches exactly 2*PW reals, so a partial block neither over-reads
// past the axis buffer nor needs a mask.
template<typename T, std::size_t N>
ADM_CONSTEVAL std::size_t sized_piece_width() {
    if constexpr (N <= 1) return 1;
    else if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, N>>) return N;
    else return sized_piece_width<T, N / 2>();
}

// Piece value type: a narrower native batch, or plain T for the 1-wide piece.
template<typename T, std::size_t PW>
using sized_piece_t = std::conditional_t<PW == 1, T, xsimd::make_sized_batch_t<T, PW>>;

// Load/store one PW-wide piece of a planar array; PW == 1 is the scalar piece.
// xsimd's arithmetic has scalar overloads, so one V-generic body serves every width
// down to and including PW == 1. Route the multiply-accumulate through piece_fnma /
// piece_fma below rather than xsimd::fma directly, and never through fms.
template<typename T, std::size_t PW>
[[nodiscard]] ADM_ALWAYS_INLINE sized_piece_t<T, PW> load_piece(const T* p) {
    if constexpr (PW == 1) { return *p; } else { return sized_piece_t<T, PW>::load_unaligned(p); }
}

template<typename T, std::size_t PW>
ADM_ALWAYS_INLINE void store_piece(T* p, sized_piece_t<T, PW> v) {
    if constexpr (PW == 1) { *p = v; } else { v.store_unaligned(p); }
}

// True where one instruction computes a*b + c. Where the flag is false, xsimd's scalar
// fma and fnma are std::fma, which is a libm CALL, and xsimd's generic batch fnma is
// negate-mul-add, one op longer than the plain expression. piece_fnma and piece_fma are
// the only readers of the flag, so no kernel carries an ISA test of its own.
inline constexpr bool kFusedFma = XSIMD_WITH_FMA3_SSE || XSIMD_WITH_FMA3_AVX
                                 || XSIMD_WITH_FMA3_AVX2 || XSIMD_WITH_FMA4
                                 || XSIMD_WITH_AVX512F || XSIMD_WITH_NEON64
                                 || XSIMD_WITH_SVE || XSIMD_WITH_RVV || XSIMD_WITH_VSX
                                 || XSIMD_WITH_VXE;

// c - a*b and a*b + c over a piece of any width, PW == 1 included. On FMA hardware the
// explicit call pins ONE association at every width. gcc contracts the plain form as
// vfnmadd in the vector body but as vfmsub in the PW == 1 tail, so a tail rounds
// differently from the row it belongs to. Off FMA hardware nothing contracts, so the
// plain form holds one association at every width and is the shorter sequence.
template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fnma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fnma(a, b, c); } else { return c - a * b; }
}

template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fma(a, b, c); } else { return a * b + c; }
}

// Bit w set iff xsimd can materialise a piece of exactly width w. Bit 1 is the
// scalar piece, always available. f64: 1,2,4,8; f32: 1,4,8,16. There is no
// 2-wide float, which is why the tail policy below cannot be width-agnostic.
// 64-bit: W >= 32 would shift a 32-bit mask out of range (no shipped ISA is
// that wide; this keeps the constant total).
template<typename T, std::size_t... Ws>
ADM_CONSTEVAL std::uint64_t piece_width_mask(std::index_sequence<Ws...>) {
    return std::uint64_t{2} | ((sized_piece_width<T, Ws + 2>() == Ws + 2 ? std::uint64_t{1} << (Ws + 2)
                                                                          : std::uint64_t{0}) | ...);
}
template<typename T>
inline constexpr std::uint64_t kPieceWidths =
    piece_width_mask<T>(std::make_index_sequence<xsimd::batch<T>::size - 1>{});

// Widest-first cover of [i, n) by EXACT-width pieces, halving from PW: every lane
// of every piece is a real element, so nothing is masked and nothing is wasted.
// `emit(integral_constant<PW>, i)` runs one piece. Callers whose hot loop must stay
// literal run their own full-width loop first and call this only for the tail (the
// leading loop here then degenerates to zero iterations).
//
// Overlap: a fat tail that is not itself an available width takes ONE BACKWARD-ALIGNED
// PW-wide piece instead of a narrower cover. Output element i depends only on input
// element i, so recomputing the overlap is bit-identical and needs no mask. Pass false
// for in-place callers, where the overlap would re-read what the bulk already wrote.
// The overlap is taken only when no single narrower piece finishes the range (rem not an
// available width) and the tail is fat (2*rem >= PW, else the overlap recomputes more
// elements than the narrow pieces it replaces).
//
// Hoisting matters: the descent costs one guard per candidate width, so a caller that
// nests this INSIDE its row loop pays them per row. The out-of-place column passes hoist
// it out (row loop inside the emitted piece); in-place callers want the opposite (see
// dif_col_tail_fused).
template<typename T, std::size_t PW, bool Overlap, typename Emit>
ADM_ALWAYS_INLINE void sized_cover(std::size_t i, std::size_t n, const Emit& emit) {
    for (; i + PW <= n; i += PW) emit(std::integral_constant<std::size_t, PW>{}, i);
    if constexpr (PW > 1) {
        if constexpr (Overlap) {
            const std::size_t rem = n - i;
            if (rem != 0 && n >= PW && 2 * rem >= PW && !((kPieceWidths<T> >> rem) & 1u)) {
                emit(std::integral_constant<std::size_t, PW>{}, n - PW);
                return;
            }
        }
        sized_cover<T, sized_piece_width<T, PW / 2>(), Overlap, Emit>(i, n, emit);
    }
}

// Store planar re/im batches as W contiguous AoS complex (2*W reals) to `dst`.
// Batch is deduced so width-adaptive callers can pass a narrower sized_batch.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    xsimd::zip_lo(re, im).store_unaligned(dst);
    xsimd::zip_hi(re, im).store_unaligned(dst + W);
}

// PW-wide AoS<->SoA piece. A lone complex is already two adjacent reals, so the
// scalar piece needs no swizzle. Either way the piece touches exactly 2*PW reals, so a
// partial block neither over-reads past the axis buffer nor needs a mask.
template<typename T, std::size_t PW>
ADM_ALWAYS_INLINE void aos_deinterleave_piece(const T* ADM_RESTRICT src,
                                              sized_piece_t<T, PW>& re,
                                              sized_piece_t<T, PW>& im) {
    if constexpr (PW == 1) {
        re = src[0];
        im = src[1];
    } else {
        aos_deinterleave<T, sized_piece_t<T, PW>>(src, re, im);
    }
}

template<typename T, std::size_t PW>
ADM_ALWAYS_INLINE void aos_interleave_piece(T* ADM_RESTRICT dst, sized_piece_t<T, PW> re,
                                            sized_piece_t<T, PW> im) {
    if constexpr (PW == 1) {
        dst[0] = re;
        dst[1] = im;
    } else {
        aos_interleave<T, sized_piece_t<T, PW>>(dst, re, im);
    }
}

// Compile-time lane predicate i < N, the generator every masked prefix in this header and
// in the DIF passes needs. make_batch_bool_constant's result type is parameterised by the
// lane VALUES, not by the generator type, so one shared generator is codegen-identical to
// any per-site local struct.
template<std::size_t N>
struct lane_lt {
    static constexpr bool get(std::size_t i, std::size_t) noexcept { return i < N; }
};

// Runtime twin of lane_lt: lanes [0, n). n >= W gives all-true, n == 0 all-false, so
// callers can clamp a saturating bound into it instead of branching.
// Derived from a compare, NOT from batch_bool::from_mask((1<<n)-1): the integer form is
// rematerialised per butterfly instead of staying in a k-register on the B < W cells
// where the whole pass is one tail. Do not "simplify" this.
template<typename T>
[[nodiscard]] ADM_ALWAYS_INLINE xsimd::batch_bool<T> lane_prefix_mask(std::size_t n) {
    using batch = xsimd::batch<T>;
    static constexpr auto seq = [] {
        std::array<T, batch::size> s{};
        for (std::size_t i = 0; i < batch::size; ++i) s[i] = static_cast<T>(i);
        return s;
    }();
    return batch::load_unaligned(seq.data()) < batch(static_cast<T>(n));
}

// Store the first R (1<=R<=W) planar complex as AoS (2*R reals), the partial-block
// store for dif_pass_last masked tail. Compile-time prefix masks lower to plain
// moves (e.g. R=3 at W=4 f32: movups+movsd rather than vmaskmov).
template<std::size_t R, typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_prefix(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    static_assert(R >= 1 && R <= W);
    using arch = typename Batch::arch_type;
    if constexpr (R == W) {
        aos_interleave<T, Batch>(dst, re, im);
    } else if constexpr (2 * R <= W) {
        xsimd::zip_lo(re, im).store(dst, xsimd::make_batch_bool_constant<T, lane_lt<2 * R>, arch>(),
                                    xsimd::unaligned_mode{});
    } else {
        xsimd::zip_lo(re, im).store_unaligned(dst);
        xsimd::zip_hi(re, im).store(dst + W,
                                    xsimd::make_batch_bool_constant<T, lane_lt<2 * R - W>, arch>(),
                                    xsimd::unaligned_mode{});
    }
}

// Peel leading elements so a run of Wc-complex AoS stores lands cache-line aligned:
// off-line wide stores straddle two lines and split. Requires stride % LANE == 0, so
// data+i*stride stays line-aligned for every row i and all rows share one peel taken
// from data's own offset; otherwise return 0.
// Wc is the store width in complex values, NOT always xsimd::batch<T>::size:
// dif_pass_last sizes its batch to the radix (dif_last_batch).
template<typename T, std::size_t Wc = xsimd::batch<T>::size>
ADM_ALWAYS_INLINE std::size_t aos_store_align_peel(const std::complex<T>* data,
                                                   std::size_t stride, std::size_t B) {
    constexpr std::size_t LANE = kCacheLine / sizeof(std::complex<T>);  // complex values per line
    // Disabled when Wc < LANE (SSE2 f64 Wc=2, SSE f32 Wc=4 vs LANE 4/8, or a small radix):
    // a peel of LANE-1 would exceed the store width, silently dropping elements [Wc,peel).
    if constexpr (LANE <= 1 || Wc < LANE) {
        return 0;
    } else {
        if (stride % LANE != 0) return 0;
        const std::size_t off =
            (reinterpret_cast<std::uintptr_t>(data) / sizeof(std::complex<T>)) % LANE;
        const std::size_t peel = (LANE - off) % LANE;
        return peel < B ? peel : B;
    }
}

// Store the last (W-M0) planar complex as AoS, the partial-block suffix store,
// mirror of aos_interleave_prefix. Compile-time suffix masks lower to plain moves.
template<std::size_t M0, typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_suffix(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    static_assert(M0 >= 1 && M0 < W);
    using arch = typename Batch::arch_type;
    constexpr std::size_t r0 = 2 * M0;  // first active real lane of the interleaved block
    if constexpr (r0 >= W) {            // all active lanes in the high half [W, 2W)
        struct sfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i + W >= r0; }
        };
        xsimd::zip_hi(re, im).store(dst + W, xsimd::make_batch_bool_constant<T, sfx, arch>(),
                                    xsimd::unaligned_mode{});
    } else {                           // low half masked [r0, W), high half full
        struct sfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i >= r0; }
        };
        xsimd::zip_lo(re, im).store(dst, xsimd::make_batch_bool_constant<T, sfx, arch>(),
                                    xsimd::unaligned_mode{});
        xsimd::zip_hi(re, im).store_unaligned(dst + W);
    }
}

// Real-lane prefix masks for a partial AoS block of R complex (2*R reals), split over the
// lo and hi halves of the zip. Built once per pass so the piece body carries neither a width
// descent nor a per-piece branch: 2*R >= W makes lo all-true, 2*R <= W makes hi all-false.
template<typename T>
struct aos_prefix_masks {
    xsimd::batch_bool<T> lo, hi;
};

template<typename T>
[[nodiscard]] ADM_ALWAYS_INLINE aos_prefix_masks<T> make_aos_prefix_masks(std::size_t R) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    return {lane_prefix_mask<T>(2 * R), lane_prefix_mask<T>(2 * R > W ? 2 * R - W : 0)};
}

// Compile-time twin of the above, same .lo/.hi shape so the masked helpers below take either.
// Which form is cheaper is an ISA property, not a preference; see kRuntimeTailMask in
// dif_col_pass.hpp. `hi` is all-false when the block does not reach the second zip half, and
// that arm is then dead by HiHalf rather than by testing the mask.
template<std::size_t R, typename T>
struct aos_ct_masks {
    using arch = typename xsimd::batch<T>::arch_type;
    static constexpr std::size_t W = xsimd::batch<T>::size;
    static constexpr auto lo =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R < W ? 2 * R : W)>, arch>();
    static constexpr auto hi =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R > W ? 2 * R - W : 0)>, arch>();
};

// Runtime-R twins of aos_deinterleave / aos_interleave_prefix. One masked full-width
// piece replaces the width descent for a sub-batch block that is not itself a piece
// width (f32 has no 2-wide batch, so sized_cover would take B=3 to 1+1+1). The masks
// also make this the only safe form for a planar row of stride B: an unmasked load runs
// into the next row, or past the buffer on the last one.
//
// HiHalf = 2*R > W is whether the interleaved block reaches the second zip half; it is
// loop-invariant, so the caller picks the arm once per pass. NOT a runtime test on m.hi
// being empty: a k0-masked vmovups is still a load uop, and `src + W` would be
// past-the-end pointer arithmetic on a row of fewer than W/2 complex.
template<bool HiHalf, typename T, typename Mask>
ADM_ALWAYS_INLINE void aos_deinterleave_masked(const T* ADM_RESTRICT src, xsimd::batch<T>& re,
                                               xsimd::batch<T>& im, const Mask m) {
    using batch = xsimd::batch<T>;
    using arch = typename batch::arch_type;
    using index = xsimd::as_unsigned_integer_t<T>;
    constexpr std::size_t W = batch::size;
    const batch lo = batch::load(src, m.lo, xsimd::unaligned_mode{});
    batch hi(T(0));
    if constexpr (HiHalf) hi = batch::load(src + W, m.hi, xsimd::unaligned_mode{});
    re = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<index, aos_even_lane, arch>());
    im = xsimd::shuffle(lo, hi, xsimd::make_batch_constant<index, aos_odd_lane, arch>());
}

template<bool HiHalf, typename T, typename Mask>
ADM_ALWAYS_INLINE void aos_interleave_masked(T* ADM_RESTRICT dst, xsimd::batch<T> re,
                                             xsimd::batch<T> im, const Mask m) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    xsimd::zip_lo(re, im).store(dst, m.lo, xsimd::unaligned_mode{});
    if constexpr (HiHalf) xsimd::zip_hi(re, im).store(dst + W, m.hi, xsimd::unaligned_mode{});
}

// Copy of a run of at most 2*W reals: at most one full batch plus one masked batch.
// The run length is loop-invariant at every call site (a band width), so the mask is
// built once and each copy is two vector moves. std::copy_n on a runtime count compiles
// to a memmove PLT call. Widen the piece count before using this on longer runs.
template<typename T>
struct real_run_copy {
    bool full;                    // n >= W: one unmasked batch leads
    bool masked;                  // reals left over after that batch
    xsimd::batch_bool<T> mask;    // prefix mask for those leftover reals

    [[nodiscard]] static real_run_copy make(std::size_t n) {
        constexpr std::size_t W = xsimd::batch<T>::size;
        const bool f = n >= W;
        const std::size_t r = f ? n - W : n;
        return {f, r != 0, lane_prefix_mask<T>(r)};
    }

    ADM_ALWAYS_INLINE void operator()(const T* ADM_RESTRICT src, T* ADM_RESTRICT dst) const {
        using batch = xsimd::batch<T>;
        if (full) {
            batch::load_unaligned(src).store_unaligned(dst);
            src += batch::size;
            dst += batch::size;
        }
        if (masked)
            batch::load(src, mask, xsimd::unaligned_mode{})
                .store(dst, mask, xsimd::unaligned_mode{});
    }
};

// Runtime prefix/suffix stores for the column last pass's misaligned head (prefix [0,n))
// and ragged tail (suffix [m0,W)): static_for dispatches the loop-invariant runtime bound
// to a compile-time batch_bool_constant store, and each arm is a fully specialised masked
// store the compiler lowers to plain moves. Kept out of the aligned-bulk store
// (plain aos_interleave) so the hot path carries none of this dispatch.
//
// ALWAYS_INLINE: the peel runs per column block, not per execute, so a call here sits on
// the hot path. Do not outline.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_prefix_n(T* ADM_RESTRICT dst, Batch re, Batch im,
                                               std::size_t n) {
    constexpr std::size_t W = Batch::size;
    if (n == 0) return;
    if (n >= W) { aos_interleave<T, Batch>(dst, re, im); return; }
    // Braced init, not static_cast: R.value is a constant expression that provably
    // fits, so the conversion is checked at compile time instead of asserted.
    poet::static_for<1, std::ptrdiff_t{W}>([&](auto R) {
        constexpr std::size_t R_n{R.value};
        if (R_n == n) aos_interleave_prefix<R_n, T, Batch>(dst, re, im);
    });
}

template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_suffix_n(T* ADM_RESTRICT dst, Batch re, Batch im,
                                               std::size_t m0) {
    constexpr std::size_t W = Batch::size;
    if (m0 == 0) { aos_interleave<T, Batch>(dst, re, im); return; }
    poet::static_for<1, std::ptrdiff_t{W}>([&](auto M) {
        constexpr std::size_t M_0{M.value};
        if (M_0 == m0) aos_interleave_suffix<M_0, T, Batch>(dst, re, im);
    });
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
