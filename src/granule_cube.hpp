#pragma once

// Granule (AoS) cube driver: one N^3 block transform as three through-memory passes
// over the existing rows/cols drivers (granule_apply.hpp); seated by nd_plan.hpp's
// fast3d gate for N in {4, 8} (granule_cube_admit_v). No new kernels: granule_pow2
// covers both sizes, and the only transpose is granule_row_tile's own.
//
//   pass 1: innermost axis -- the rows driver, nlines = N^2, stride N
//   pass 2: middle axis    -- N col-driver calls, one per plane, in_inner = N, ncols = N
//   pass 3: outermost axis -- one col-driver call, in_inner = N^2, ncols = N^2
//
// Each pass consumes the previous pass's complete output and the row/col tiles are
// block-locally in-place safe, so in == out is legal (the row ladder's masked terminal
// block never fires at these sizes: N^2 is a multiple of every granule tile width).
//
// The whole scale folds into pass 1 (the Scaled fold of granule_apply.hpp's fct branch,
// and make_scale_plan's innermost-axis convention). The default inverse factor 1/N^3 is
// an exact binary fraction at N in {4, 8}, so the fold costs nothing numerically.

#include <complex>
#include <cstddef>

#include "granule_apply.hpp"

#include "admiral/detail/macros.hpp"

namespace admiral {
namespace detail {

template<unsigned N, typename T, bool Forward, bool Scaled, typename V>
ADM_ALWAYS_INLINE void granule_cube_run(const std::complex<T>* in, std::complex<T>* out,
                                        T fct) {
    granule_rows_run<N, T, Forward, Scaled, V>(in, out, N * N, N, N, fct);
    for (std::size_t p = 0; p < N; ++p)
        granule_cols_run<N, T, Forward, false, V>(out + p * N * N, N, out + p * N * N, N,
                                                  N, T(1));
    granule_cols_run<N, T, Forward, false, V>(out, N * N, out, N * N, N * N, T(1));
}

template<unsigned N, typename T, bool Forward>
void granule_cube_apply(const std::complex<T>* in, std::complex<T>* out, T fct) {
    using V = xsimd::batch<T>;
    if (fct == T(1)) granule_cube_run<N, T, Forward, false, V>(in, out, fct);
    else             granule_cube_run<N, T, Forward, true, V>(in, out, fct);
}

}
}

#include "admiral/detail/undef_macros.hpp"
