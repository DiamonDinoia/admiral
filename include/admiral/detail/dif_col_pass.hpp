#pragma once

// DIF passes down columns, one SIMD lane per column. Butterflies come from butterfly.hpp.

#include <array>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"
#include "simd.hpp"

#include "butterfly.hpp"
#include "cache.hpp"
#include "simd_swizzle.hpp"
#include "macros.hpp"

namespace admiral {
namespace detail {

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
            store_piece<T, PW>(chre + p * B + c, owr * sr - owi * si);
            store_piece<T, PW>(chim + p * B + c, owr * si + owi * sr);
        } else {
            store_piece<T, PW>(chre + p * B + c, sr);
            store_piece<T, PW>(chim + p * B + c, si);
        }
    });
}

inline constexpr bool kRuntimeTailMask = poet::vector_register_count() >= 32;

template<typename T>
[[nodiscard]] constexpr bool one_piece_cover(std::size_t first, std::size_t last) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t rem = last - first;
    assert(rem < W);
    return rem == 0 || ((kPieceWidths<T> >> rem) & 1u) != 0u || (last >= W && 2 * rem >= W);
}

template<typename T, std::size_t IP, typename Mask>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_masked(const T* ccre,
                                            const T* ccim,
                                            T* chre, T* chim,
                                            std::size_t l1, std::size_t ido, std::size_t B,
                                            std::size_t a, std::size_t b, std::size_t c,
                                            const T* twre,
                                            const T* twim, const Mask m) {
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

template<typename T, bool Forward, std::size_t IP, bool HiHalf, typename Mask, typename AMask>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_first_masked(
    const std::complex<T>* data, std::size_t axis_stride, T* chre,
    T* chim, std::size_t l1, std::size_t ido, std::size_t B, std::size_t a,
    std::size_t b, std::size_t c, const T* twre, const T* twim,
    const Mask m, const AMask am) {
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

template<typename T, bool Forward, std::size_t IP, bool HiHalf, typename Mask, typename AMask>
ADM_ALWAYS_INLINE ADM_FLATTEN void dif_col_piece_last_masked(
    const T* ccre, const T* ccim, std::complex<T>* data,
    std::size_t axis_stride, std::size_t l1, std::size_t B, std::size_t b, std::size_t c,
    T scale_val, const Mask m, const AMask am) {
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

template<typename T, typename F>
[[nodiscard]] ADM_ALWAYS_INLINE bool dispatch_one_piece(std::size_t rem, F&& f) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    assert(rem < W);
    bool hit = false;
    poet::static_for<1, std::ptrdiff_t{detail::bit_width(W)} - 1>([&](auto E) {
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
    sized_cover<T, W, true>(cfull, B, [&](auto PWc, std::size_t c) {
        for (std::size_t b = 0; b < l1; ++b)
            for (std::size_t a = 0; a < ido; ++a)
                dif_col_piece_first<T, Forward, IP, PWc.value>(data, axis_stride, chre, chim,
                                                              l1, ido, B, a, b, c, twre, twim);
    });
}

template<typename T, bool Forward, std::size_t IP>
ADM_NOINLINE void dif_col_tail_last_general(const T* ccre,
                                           const T* ccim,
                                           std::complex<T>* data,
                                           std::size_t axis_stride, std::size_t l1, std::size_t B,
                                           T scale_val) {
    constexpr std::size_t W = xsimd::batch<T>::size;
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
    for (std::size_t b = 0; b < l1; ++b)
        sized_cover<T, W, false>(cfull, B, [&](auto PWc, std::size_t c) {
            dif_col_piece_fused<T, Forward, IP, PWc.value>(data, axis_stride, l1, b, c,
                                                           scale_val);
        });
}

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
                dif_col_tail_one_piece<T, IP, PW.value>(ccre, ccim, chre, chim, l1, ido,
                                                                B, cfull, twre, twim);
            }))
            dif_col_tail<T, IP>(ccre, ccim, chre, chim, l1, ido, B, cfull, twre, twim);
}

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

template<typename T, bool Forward, std::size_t IP>
void dif_col_pass_last(const T* ccre, const T* ccim,
                       std::complex<T>* data, std::size_t axis_stride,
                       std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                       [[maybe_unused]] const T* twre,
                       [[maybe_unused]] const T* twim, T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;

    assert(ido == 1);

    if (B < W) {
        dif_col_tail_last<T, Forward, IP>(ccre, ccim, data, axis_stride, l1, B, scale_val);
        return;
    }

    const std::size_t peel = aos_store_align_peel<T>(data, axis_stride, B);

    for (std::size_t b = 0; b < l1; ++b) {
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
        std::size_t c = 0;
        if (peel > 0) {
            vec_block(0, [peel](T* d, batch r, batch i) { aos_interleave_prefix_n<T>(d, r, i, peel); });
            c = peel;
        }
        const auto full_store = [](T* d, batch r, batch i) { aos_interleave<T>(d, r, i); };
        for (; c + W <= B; c += W) vec_block(c, full_store);
        if (c < B) {
            const std::size_t m0 = c - (B - W);
            vec_block(B - W, [m0](T* d, batch r, batch i) { aos_interleave_suffix_n<T>(d, r, i, m0); });
        }
    }
}

template<typename T, bool Forward, std::size_t IP>
void dif_col_pass_fused(std::complex<T>* data, std::size_t axis_stride,
                        std::size_t l1, [[maybe_unused]] std::size_t ido, std::size_t B,
                        [[maybe_unused]] const T* twre,
                        [[maybe_unused]] const T* twim, T scale_val = T(1)) {
    using batch = xsimd::batch<T>;
    constexpr std::size_t W = batch::size;

    const std::size_t cfull = B - B % W;

    if (B >= W) {
        for (std::size_t b = 0; b < l1; ++b) {
            for (std::size_t c = 0; c < cfull; c += W) {
                batch tr[IP], ti[IP];
                for (std::size_t j = 0; j < IP; ++j) {
                    const std::size_t p = j + IP * b;
                    const T* src = reinterpret_cast<const T*>(data + p * axis_stride + c);
                    auto [dr, di] = plane_refs<Forward>(tr[j], ti[j]);
                    aos_deinterleave<T>(src, dr, di);
                }
                dif_butterfly_terminal<T, IP>(tr, ti, [&](const auto k, batch sr, batch si) {
                    const std::size_t p = b + l1 * k;
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

template<typename T>
struct dif_col_pass_invoke_t {
    template<std::size_t IP>
    void operator()(const T* ccre, const T* ccim,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* twre, const T* twim) const {
        dif_col_pass<T, IP>(ccre, ccim, chre, chim, l1, ido, B, twre, twim);
    }
};
template<typename T>
inline constexpr dif_col_pass_invoke_t<T> dif_col_pass_invoke{};

template<typename T, bool Forward>
struct dif_col_pass_first_invoke_t {
    template<std::size_t IP>
    void operator()(const std::complex<T>* data, std::size_t axis_stride,
                    T* chre, T* chim,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* twre, const T* twim) const {
        dif_col_pass_first<T, Forward, IP>(data, axis_stride, chre, chim, l1, ido, B, twre, twim);
    }
};
template<typename T, bool Forward>
inline constexpr dif_col_pass_first_invoke_t<T, Forward> dif_col_pass_first_invoke{};

template<typename T, bool Forward>
struct dif_col_pass_last_invoke_t {
    template<std::size_t IP>
    void operator()(const T* ccre, const T* ccim,
                    std::complex<T>* data, std::size_t axis_stride,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* twre, const T* twim, T scale_val) const {
        dif_col_pass_last<T, Forward, IP>(ccre, ccim, data, axis_stride, l1, ido, B, twre, twim,
                                          scale_val);
    }
};
template<typename T, bool Forward>
inline constexpr dif_col_pass_last_invoke_t<T, Forward> dif_col_pass_last_invoke{};

template<typename T, bool Forward>
struct dif_col_pass_fused_invoke_t {
    template<std::size_t IP>
    void operator()(std::complex<T>* data, std::size_t axis_stride,
                    std::size_t l1, std::size_t ido, std::size_t B,
                    const T* twre, const T* twim, T scale_val) const {
        dif_col_pass_fused<T, Forward, IP>(data, axis_stride, l1, ido, B, twre, twim, scale_val);
    }
};
template<typename T, bool Forward>
inline constexpr dif_col_pass_fused_invoke_t<T, Forward> dif_col_pass_fused_invoke{};

}
}

#include "undef_macros.hpp"
