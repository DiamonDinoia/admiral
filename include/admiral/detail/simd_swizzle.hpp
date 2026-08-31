#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>  // `std::is_void_v`, `std::conditional_t`, `std::integral_constant`
#include <tuple>        // `std::tie` (`plane_refs`)
#include <utility>      // `std::index_sequence` (piece-width mask), `std::pair`

#include <poet/poet.hpp>  // `poet::static_for` (runtime bound -> compile-time mask dispatch)
#include "cxx_compat.hpp"  // `ADM_CONSTEVAL`
#include "simd.hpp"

#include "cache.hpp"   // `kCacheLine` (AoS store peel)
#include "macros.hpp"  // `ADM_ALWAYS_INLINE`, `ADM_RESTRICT` (undef'd at end of header)

namespace admiral {
namespace detail {

// SIMD AoS<->SoA swizzle at DIF pass boundaries. `W` consecutive AoS complex
// = 2*W reals r0,i0,r1,i1 through rW-1,iW-1; the fused first/last passes
// need planar `batch<T>` pairs and back. Deinterleave: one 2-source shuffle
// per plane (even->re, odd->im). Interleave: `zip_lo`/`zip_hi`.

// Permute-index generators for `xsimd::make_batch_constant`: lane `i` reads
// `get(i)` from the concatenated `lo||hi` (0..W-1 -> `lo`, W..2W-1 -> `hi`).
struct aos_even_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i; }
};
struct aos_odd_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i + 1; }
};

// The engine's direction boundary, and the only place the plane exchange
// happens. Every DIF pass computes the forward transform only (`butterfly.hpp`);
// the inverse runs the same code with the planes swapped. A boundary pass
// routes its load's destination pair through `plane_refs` and its store's
// value pair through `plane_vals`, so everything between the boundaries is
// direction-free. Two forms because a load needs lvalues and a store takes
// computed values.
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

// Load `W` contiguous AoS complex (2*W reals) into planar re/im batches.
// `Batch` is deduced, so width-adaptive callers can pass a narrower
// `sized_batch`.
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

// Widest xsimd piece width <= `N`, halving until `make_sized_batch_t` is
// non-void. A PW-wide piece fixes the AoS footprint for free.
// `aos_deinterleave` at width `PW` touches exactly 2*PW reals, so a partial
// block never over-reads past the axis buffer and never needs a mask.
template<typename T, std::size_t N>
ADM_CONSTEVAL std::size_t sized_piece_width() {
    if constexpr (N <= 1) return 1;
    else if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, N>>) return N;
    else return sized_piece_width<T, N / 2>();
}

template<typename T, std::size_t PW>
using sized_piece_t = std::conditional_t<PW == 1, T, xsimd::make_sized_batch_t<T, PW>>;

// Load/store one PW-wide piece of a planar array. Scalar overloads let one
// V-generic body serve every width. Route multiply-accumulate through
// `piece_fnma` / `piece_fma` below, never through `fms`.
template<typename T, std::size_t PW>
[[nodiscard]] ADM_ALWAYS_INLINE sized_piece_t<T, PW> load_piece(const T* p) {
    if constexpr (PW == 1) { return *p; } else { return sized_piece_t<T, PW>::load_unaligned(p); }
}

template<typename T, std::size_t PW>
ADM_ALWAYS_INLINE void store_piece(T* p, sized_piece_t<T, PW> v) {
    if constexpr (PW == 1) { *p = v; } else { v.store_unaligned(p); }
}

// True where one instruction computes a*b + c. Where false, xsimd's scalar
// `fma` and `fnma` are `std::fma` (a libm call), and xsimd's generic batch
// `fnma` is negate-mul-add. Negate-mul-add runs one op longer than the plain
// expression. `piece_fnma` and `piece_fma` are the only readers of the flag.
inline constexpr bool kFusedFma = XSIMD_WITH_FMA3_SSE || XSIMD_WITH_FMA3_AVX
                                 || XSIMD_WITH_FMA3_AVX2 || XSIMD_WITH_FMA4
                                 || XSIMD_WITH_AVX512F || XSIMD_WITH_NEON64
                                 || XSIMD_WITH_SVE || XSIMD_WITH_RVV || XSIMD_WITH_VSX
                                 || XSIMD_WITH_VXE;

// c - a*b and a*b + c over a piece of any width, PW == 1 included. On FMA
// hardware the explicit call pins ONE association at every width. gcc
// contracts the plain form as `vfnmadd` in the vector body but as `vfmsub` in
// the scalar tail. A tail would then round differently from the row the tail
// belongs to.
template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fnma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fnma(a, b, c); } else { return c - a * b; }
}

template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fma(a, b, c); } else { return a * b + c; }
}

// Bit `w` set iff xsimd can materialise a piece of exactly width `w`. Bit 1
// is the scalar piece, always available. 64-bit storage: `W >= 32` would
// shift a 32-bit mask out of range (no shipped ISA is that wide anyway).
template<typename T, std::size_t... Ws>
ADM_CONSTEVAL std::uint64_t piece_width_mask(std::index_sequence<Ws...>) {
    return std::uint64_t{2} | ((sized_piece_width<T, Ws + 2>() == Ws + 2 ? std::uint64_t{1} << (Ws + 2)
                                                                          : std::uint64_t{0}) | ...);
}
template<typename T>
inline constexpr std::uint64_t kPieceWidths =
    piece_width_mask<T>(std::make_index_sequence<xsimd::batch<T>::size - 1>{});

// Widest-first cover of [i, n) by EXACT-width pieces, halving from `PW`:
// every lane of every piece is a real element, so nothing is masked.
// `emit(integral_constant<PW>, i)` runs one piece. If the hot loop must stay
// literal, run a full-width loop first and call `sized_cover` only for the
// tail.
//
// Overlap: a fat tail with no available width takes ONE BACKWARD-ALIGNED
// PW-wide piece instead of a narrower cover. Output element `i` depends only
// on input element `i`, so recomputing the overlap is bit-identical. If the
// caller runs in place, pass false: the overlap would re-read what the bulk
// wrote. The overlap fires only when `rem` is not an available width and
// `2*rem >= PW`. With `2*rem < PW`, the overlap would recompute more elements
// than the narrow pieces the overlap replaces.
//
// Hoisting matters: the descent costs one guard per candidate width. The
// out-of-place column passes hoist the descent out of the row loop; in-place
// callers want the opposite (`dif_col_tail_fused`).
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

// Store planar re/im batches as `W` contiguous AoS complex (2*W reals) to `dst`.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    xsimd::zip_lo(re, im).store_unaligned(dst);
    xsimd::zip_hi(re, im).store_unaligned(dst + W);
}

// PW-wide AoS<->SoA piece. A lone complex is already two adjacent reals, so
// the scalar piece needs no swizzle.
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

// Compile-time lane predicate `i < N` for every masked prefix. The result
// type parameterises on the lane VALUES, not on the generator type, so one
// shared generator is codegen-identical to any per-site local struct.
template<std::size_t N>
struct lane_lt {
    static constexpr bool get(std::size_t i, std::size_t) noexcept { return i < N; }
};

// Runtime twin of `lane_lt` over lanes [0, n). `n >= W` gives all-true,
// `n == 0` all-false, so callers can clamp a saturating bound instead of
// branching. Derive from a compare, not from `batch_bool::from_mask((1<<n)-1)`.
// The integer form rematerialises per butterfly; the mask form holds a
// k-register on every `B < W` cell, where the whole pass is one tail.
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

// Store the first `R` (1<=R<=W) planar complex as AoS (2*R reals), the
// partial-block store for the `dif_pass_last` masked tail. Compile-time
// prefix masks lower to plain moves (`R=3` at `W=4` f32: `movups`+`movsd`,
// not `vmaskmov`).
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

// Peel leading elements so a run of Wc-complex AoS stores lands cache-line
// aligned; an off-line wide store straddles two lines and splits. The peel
// requires `stride % LANE == 0`, so every row is line-aligned and shares one
// peel taken from `data`'s own offset; otherwise return 0. `Wc` is the store
// width in complex values: `dif_pass_last` sizes its batch to the radix
// (`dif_last_batch`).
template<typename T, std::size_t Wc = xsimd::batch<T>::size>
ADM_ALWAYS_INLINE std::size_t aos_store_align_peel(const std::complex<T>* data,
                                                   std::size_t stride, std::size_t B) {
    constexpr std::size_t LANE = kCacheLine / sizeof(std::complex<T>);  // complex values per line
    // Disabled when `Wc < LANE`: a peel of `LANE-1` would exceed the store
    // width and silently drop elements `[Wc, peel)`.
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

// Store the last (W-M0) planar complex as AoS; the partial-block suffix
// mirror of `aos_interleave_prefix`, lowered the same way.
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

// Real-lane prefix masks for a partial AoS block of `R` complex (2*R reals),
// split over the `lo` and `hi` halves of the zip. The masks build once per
// pass, so the piece body carries no width descent and no per-piece branch.
template<typename T>
struct aos_prefix_masks {
    xsimd::batch_bool<T> lo, hi;
};

template<typename T>
[[nodiscard]] ADM_ALWAYS_INLINE aos_prefix_masks<T> make_aos_prefix_masks(std::size_t R) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    return {lane_prefix_mask<T>(2 * R), lane_prefix_mask<T>(2 * R > W ? 2 * R - W : 0)};
}

// Compile-time twin of `aos_prefix_masks`, same `.lo`/`.hi` shape, so the
// masked helpers below take either form (the choice is an ISA property; see
// `kRuntimeTailMask` in `dif_col_pass.hpp`). `hi` is all-false when the block
// stops before the second zip half; that arm then dies by `HiHalf`, not by
// testing the mask.
template<std::size_t R, typename T>
struct aos_ct_masks {
    using arch = typename xsimd::batch<T>::arch_type;
    static constexpr std::size_t W = xsimd::batch<T>::size;
    static constexpr auto lo =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R < W ? 2 * R : W)>, arch>();
    static constexpr auto hi =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R > W ? 2 * R - W : 0)>, arch>();
};

// Runtime-R twins of `aos_deinterleave` / `aos_interleave`. One masked
// full-width piece replaces the width descent for a block that has no piece
// width. f32 has no 2-wide batch, so `B=3` would descend to 1+1+1.
// The masks also make the masked form the only safe form for a planar row of
// stride `B`. An unmasked load would run into the next row, or past the
// buffer on the last row.
//
// `HiHalf = 2*R > W` tells whether the block reaches the second zip half. The
// flag is compile-time because the hi arm must die, not mask. A k0-masked
// `vmovups` is still a load uop, and `src + W` is past-the-end pointer
// arithmetic on a row of fewer than W/2 complex.
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

// Copy of a run of at most 2*W reals: one full batch plus one masked batch.
// The run length is loop-invariant at every call site (a band width), so the
// mask builds once. `std::copy_n` on a runtime count compiles to a `memmove`
// PLT call. Before using `real_run_copy` on longer runs, widen the piece
// count.
template<typename T>
struct real_run_copy {
    bool full;                    // `n >= W`: one unmasked batch leads
    bool masked;                  // reals left over after the full batch
    xsimd::batch_bool<T> mask;    // prefix mask for the leftover reals

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

// Runtime prefix/suffix stores for the column last pass's misaligned head
// (prefix [0,n)) and ragged tail (suffix [m0,W)). `static_for` dispatches the
// loop-invariant runtime bound to a compile-time masked store lowered to plain
// moves. The stores stay out of the aligned-bulk store, so the hot path
// carries none of this dispatch. `ADM_ALWAYS_INLINE` because the peel runs per
// column block.
template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_prefix_n(T* ADM_RESTRICT dst, Batch re, Batch im,
                                               std::size_t n) {
    constexpr std::size_t W = Batch::size;
    if (n == 0) return;
    if (n >= W) { aos_interleave<T, Batch>(dst, re, im); return; }
    // Braced init, not `static_cast`: `R.value` is a constant expression that
    // fits, so the conversion is checked at compile time, not asserted.
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
