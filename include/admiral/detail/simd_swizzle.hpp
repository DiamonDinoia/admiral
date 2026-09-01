#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <tuple>
#include <utility>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"
#include "simd.hpp"

#include "cache.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

struct aos_even_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i; }
};
struct aos_odd_lane {
    static constexpr std::size_t get(std::size_t i, std::size_t) { return 2 * i + 1; }
};

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

template<typename T, std::size_t N>
ADM_CONSTEVAL std::size_t sized_piece_width() {
    if constexpr (N <= 1) return 1;
    else if constexpr (!std::is_void_v<xsimd::make_sized_batch_t<T, N>>) return N;
    else return sized_piece_width<T, N / 2>();
}

template<typename T, std::size_t PW>
using sized_piece_t = std::conditional_t<PW == 1, T, xsimd::make_sized_batch_t<T, PW>>;

template<typename T, std::size_t PW>
[[nodiscard]] ADM_ALWAYS_INLINE sized_piece_t<T, PW> load_piece(const T* p) {
    if constexpr (PW == 1) { return *p; } else { return sized_piece_t<T, PW>::load_unaligned(p); }
}

template<typename T, std::size_t PW>
ADM_ALWAYS_INLINE void store_piece(T* p, sized_piece_t<T, PW> v) {
    if constexpr (PW == 1) { *p = v; } else { v.store_unaligned(p); }
}

inline constexpr bool kFusedFma = XSIMD_WITH_FMA3_SSE || XSIMD_WITH_FMA3_AVX
                                 || XSIMD_WITH_FMA3_AVX2 || XSIMD_WITH_FMA4
                                 || XSIMD_WITH_AVX512F || XSIMD_WITH_NEON64
                                 || XSIMD_WITH_SVE || XSIMD_WITH_RVV || XSIMD_WITH_VSX
                                 || XSIMD_WITH_VXE;

template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fnma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fnma(a, b, c); } else { return c - a * b; }
}

template<typename V>
[[nodiscard]] ADM_ALWAYS_INLINE V piece_fma(V a, V b, V c) {
    if constexpr (kFusedFma) { return xsimd::fma(a, b, c); } else { return a * b + c; }
}

template<typename T, std::size_t... Ws>
ADM_CONSTEVAL std::uint64_t piece_width_mask(std::index_sequence<Ws...>) {
    return std::uint64_t{2} | ((sized_piece_width<T, Ws + 2>() == Ws + 2 ? std::uint64_t{1} << (Ws + 2)
                                                                          : std::uint64_t{0}) | ...);
}
template<typename T>
inline constexpr std::uint64_t kPieceWidths =
    piece_width_mask<T>(std::make_index_sequence<xsimd::batch<T>::size - 1>{});

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

template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    xsimd::zip_lo(re, im).store_unaligned(dst);
    xsimd::zip_hi(re, im).store_unaligned(dst + W);
}

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

template<std::size_t N>
struct lane_lt {
    static constexpr bool get(std::size_t i, std::size_t) noexcept { return i < N; }
};

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

template<typename T, std::size_t Wc = xsimd::batch<T>::size>
ADM_ALWAYS_INLINE std::size_t aos_store_align_peel(const std::complex<T>* data,
                                                   std::size_t stride, std::size_t B) {
    constexpr std::size_t LANE = kCacheLine / sizeof(std::complex<T>);
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

template<std::size_t M0, typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_suffix(T* ADM_RESTRICT dst, Batch re, Batch im) {
    constexpr std::size_t W = Batch::size;
    static_assert(M0 >= 1 && M0 < W);
    using arch = typename Batch::arch_type;
    constexpr std::size_t r0 = 2 * M0;
    if constexpr (r0 >= W) {
        struct sfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i + W >= r0; }
        };
        xsimd::zip_hi(re, im).store(dst + W, xsimd::make_batch_bool_constant<T, sfx, arch>(),
                                    xsimd::unaligned_mode{});
    } else {
        struct sfx {
            static constexpr bool get(std::size_t i, std::size_t) { return i >= r0; }
        };
        xsimd::zip_lo(re, im).store(dst, xsimd::make_batch_bool_constant<T, sfx, arch>(),
                                    xsimd::unaligned_mode{});
        xsimd::zip_hi(re, im).store_unaligned(dst + W);
    }
}

template<typename T>
struct aos_prefix_masks {
    xsimd::batch_bool<T> lo, hi;
};

template<typename T>
[[nodiscard]] ADM_ALWAYS_INLINE aos_prefix_masks<T> make_aos_prefix_masks(std::size_t R) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    return {lane_prefix_mask<T>(2 * R), lane_prefix_mask<T>(2 * R > W ? 2 * R - W : 0)};
}

template<std::size_t R, typename T>
struct aos_ct_masks {
    using arch = typename xsimd::batch<T>::arch_type;
    static constexpr std::size_t W = xsimd::batch<T>::size;
    static constexpr auto lo =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R < W ? 2 * R : W)>, arch>();
    static constexpr auto hi =
        xsimd::make_batch_bool_constant<T, lane_lt<(2 * R > W ? 2 * R - W : 0)>, arch>();
};

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

template<typename T>
struct real_run_copy {
    bool full;
    bool masked;
    xsimd::batch_bool<T> mask;

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

template<typename T, typename Batch = xsimd::batch<T>>
ADM_ALWAYS_INLINE void aos_interleave_prefix_n(T* ADM_RESTRICT dst, Batch re, Batch im,
                                               std::size_t n) {
    constexpr std::size_t W = Batch::size;
    if (n == 0) return;
    if (n >= W) { aos_interleave<T, Batch>(dst, re, im); return; }
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

}
}

#include "undef_macros.hpp"
