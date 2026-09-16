#pragma once

// Granule (AoS) leaf drivers: the line-count ladders over the tile bodies in
// include/admiral/detail/granule_codelet.hpp, engaged from the two choke points in
// codelet_apply.hpp behind granule_*_admit_v. Both drivers branch once on
// fct == 1: the unit-scale instantiation carries no multiply at all, so the
// dialect never pays the row/col scatter's wasted unit-scale muls.

#include <complex>
#include <cstddef>
#include <type_traits>

#include "admiral/detail/granule_codelet.hpp"
#include "admiral/detail/macros.hpp"

#include "admiral/detail/math.hpp"  // scale_inplace

namespace admiral {
namespace detail {

// Existing residual owners the drivers hand their leftovers to (one tail owner,
// no duplicated tail code). Definitions live below the include point in
// codelet_apply.hpp.
template<unsigned N, typename T, bool Forward>
void codelet_apply(const std::complex<T>* in, std::complex<T>* out);
template<unsigned N, typename T>
void col_codelet_body(const std::complex<T>* in, std::size_t in_inner, std::complex<T>* out,
                      std::size_t out_inner, std::size_t ncols, T scale, bool fwd);

// Row axis: full native tiles of GL = V::size/2 lines, then the sized-batch ladder
// (each rung halves the line count; GLv >= 2 lines per granule tile), then the
// existing per-line residual for the odd line out.
// The ladder itself is shared with the col driver: one walk, the per-rung tile call
// as the body (vt points null, its pointee is the rung's batch type).
template<typename T, std::size_t W, typename Body>
ADM_ALWAYS_INLINE void granule_rung_ladder(std::size_t n, std::size_t& pos, Body&& body) {
    poet::static_for<1, bit_width(W)>([&](auto S) ADM_LAMBDA_ALWAYS_INLINE {
        constexpr std::size_t Wv = W >> S;
        using Vt = xsimd::make_sized_batch_t<T, Wv>;
        if constexpr (Wv >= 4 && !std::is_void_v<Vt>) {
            if (pos + Wv / 2 <= n) {
                body(static_cast<const Vt*>(nullptr));
                pos += Wv / 2;
            }
        }
    });
}

template<unsigned N, typename T, bool Forward, bool Scaled, typename V>
ADM_ALWAYS_INLINE void granule_rows_run(const std::complex<T>* in, std::complex<T>* out,
                                        std::size_t nlines, std::size_t in_stride,
                                        std::size_t out_stride, T fct) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t GL = W / 2;
    std::size_t r = 0;
    for (; r + GL <= nlines; r += GL)
        granule_row_tile<N, T, Forward, Scaled, V>(
            reinterpret_cast<const T*>(in + r * in_stride),
            reinterpret_cast<T*>(out + r * out_stride), in_stride, out_stride, fct);
    granule_rung_ladder<T, W>(nlines, r, [&](const auto* vt) ADM_LAMBDA_ALWAYS_INLINE {
        using Vt = std::remove_cv_t<std::remove_pointer_t<decltype(vt)>>;
        granule_row_tile<N, T, Forward, Scaled, Vt>(
            reinterpret_cast<const T*>(in + r * in_stride),
            reinterpret_cast<T*>(out + r * out_stride), in_stride, out_stride, fct);
    });
    for (; r < nlines; ++r) {
        codelet_apply<N, T, Forward>(in + r * in_stride, out + r * out_stride);
        if constexpr (Scaled) scale_inplace(out + r * out_stride, N, fct);
    }
}

// Col axis: the same ladder over column widths, the leftover columns handed back to
// the existing col body on the shifted range.
template<unsigned N, typename T, bool Forward, bool Scaled, typename V>
ADM_ALWAYS_INLINE void granule_cols_run(const std::complex<T>* in, std::size_t in_inner,
                                        std::complex<T>* out, std::size_t out_inner,
                                        std::size_t ncols, T scale) {
    constexpr std::size_t W = V::size;
    constexpr std::size_t GL = W / 2;
    std::size_t c = 0;
    for (; c + GL <= ncols; c += GL)
        granule_col_tile<N, T, Forward, Scaled, V>(in, in_inner, out, out_inner, c, scale);
    granule_rung_ladder<T, W>(ncols, c, [&](const auto* vt) ADM_LAMBDA_ALWAYS_INLINE {
        using Vt = std::remove_cv_t<std::remove_pointer_t<decltype(vt)>>;
        granule_col_tile<N, T, Forward, Scaled, Vt>(in, in_inner, out, out_inner, c, scale);
    });
    if (c < ncols)
        col_codelet_body<N, T>(in + c, in_inner, out + c, out_inner, ncols - c, scale,
                               Forward);
}

template<unsigned N, typename T, bool Forward>
void granule_apply_rows_oop(const std::complex<T>* in, std::complex<T>* out,
                            std::size_t nlines, std::size_t in_stride,
                            std::size_t out_stride, T fct) {
    using V = xsimd::batch<T>;
    if (fct == T(1))
        granule_rows_run<N, T, Forward, false, V>(in, out, nlines, in_stride, out_stride, fct);
    else
        granule_rows_run<N, T, Forward, true, V>(in, out, nlines, in_stride, out_stride, fct);
}

template<unsigned N, typename T, bool Forward>
void granule_col_apply(const std::complex<T>* in, std::size_t in_inner,
                       std::complex<T>* out, std::size_t out_inner,
                       std::size_t ncols, T scale) {
    using V = xsimd::batch<T>;
    if (scale == T(1))
        granule_cols_run<N, T, Forward, false, V>(in, in_inner, out, out_inner, ncols, scale);
    else
        granule_cols_run<N, T, Forward, true, V>(in, in_inner, out, out_inner, ncols, scale);
}

}
}

#include "admiral/detail/undef_macros.hpp"
