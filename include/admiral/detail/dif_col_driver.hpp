#pragma once

// ============================================================================
// Batched column DIF pass-chain driver, the strided analogue of
// `iterative_dif_execute_ws` (`dif_driver.hpp`): a slab data[p * axis_stride + c] of
// shape [axis_extent][batch_count] over SIMD-lane columns. axis_stride == batch_count
// for a contiguous row-major slab. `dtw` is the ordinary 1D `dif_twiddle_set` for
// axis_extent. Caller-owned SoA scratch cc0/cc1 >= axis_extent*batch_count. `scale_val`
// folds 1/L into `dif_col_pass_last`'s stores; x*1 is bitwise identity.
// ============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <admiral/errors.hpp>  // `internal_error`

#include <poet/poet.hpp>
#include "simd.hpp"     // `batch<T>::size` (SIMD-lane block alignment)

#include "cache.hpp"         // `cache_bytes`, `cpu_cache`
#include "dif_col_pass.hpp"  // `dif_col_pass[_first/_last/_fused]` + invokers
#include "twiddles.hpp"      // `dif_twiddle_set`, `dif_radix_set`

namespace admiral {
namespace detail {

// Bytes one column tile may plan against. A serial tile stays under L2; a threaded tile
// shrinks with the sharing-group thread count, or the aggregate footprint thrashes L3.
// The /6 leaves room for other axes and scratch. `choose_line_route` compares against
// the returned budget, so tile and route agree on what "stays cached" means.
[[nodiscard]] inline std::size_t col_cache_budget(std::size_t nthreads) {
    const cache_bytes& cc = cpu_cache();
    // `l3` is per sharing-group: divide by the threads sharing the group, not a
    // machine-wide count. Unknown group size (`l3_cores` == 0) falls back to `nthreads`.
    const std::size_t nt = std::max<std::size_t>(1, nthreads);
    const std::size_t sharing = cc.l3_cores ? std::min(nt, cc.l3_cores) : nt;
    return std::min(cc.l2, (cc.l3 / 6) / sharing);
}

// Tiles per worker the balance term aims for. If runs alone outnumber the workers, the
// balance term gains nothing.
inline constexpr std::size_t kTilesPerWorker = 4;

// Column-tile width selection: widest W-multiple fitting the thread-adjusted budget.
// `col_budget_block` depends only on the axis and thread count, so it is the only width
// knowable at plan time. `nd_col_block` then clamps to [W, run_len] over `nruns` runs
// and applies the balance term: UNITS = nruns*ntiles, not ntiles. Sized off the run
// alone, the cap collapses to the SIMD floor when run_len < kTilesPerWorker*nthreads*W.
// Balancing tiles against each other gains nothing.
template<typename T>
[[nodiscard]] inline std::size_t col_budget_block(std::size_t len, std::size_t nthreads) {
    // Per column: one `std::complex<T>` of data (2*sizeof(T)) + 4 planar scratch reals.
    constexpr std::size_t per_col = 6 * sizeof(T);
    return len == 0 ? 0 : col_cache_budget(nthreads) / (len * per_col);
}

// `nruns` is deliberately not defaulted: `nruns` shifts the balance cap by
// kTilesPerWorker*nthreads, and an omitted value silently yields the narrowest tile.
template<typename T>
[[nodiscard]] inline std::size_t nd_col_block(std::size_t len, std::size_t run_len,
                                              std::size_t nthreads, std::size_t nruns) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (len == 0) return run_len;
    std::size_t bt = col_budget_block<T>(len, nthreads);
    if (nthreads > 1) {
        // Tiles each run must yield for the units to reach `kTilesPerWorker` per worker;
        // one (a no-op cap) as soon as the runs alone get there.
        const std::size_t tiles = (kTilesPerWorker * nthreads + nruns - 1) / nruns;
        bt = std::min(bt, run_len / tiles);
    }
    // Threaded tiles get line-sized granules so adjacent units on different workers
    // share no coherence line; `gran` == W once W covers the line (AVX-512).
    constexpr std::size_t line_elems = kCacheLine / sizeof(std::complex<T>);
    const std::size_t gran =
        nthreads > 1 && W < line_elems ? line_elems : W;  // both pow2
    if (bt < gran) return std::min(run_len, gran);  // never below one SIMD batch
    bt -= bt % gran;                                // align to the granule
    return std::min(bt, run_len);
}

// `first_src`: when set, the first pass reads AoS from `first_src` instead of `data`
// (fused copy-in for `four_step_large`). `first_src` shares `data`'s extent but carries
// its own stride.
template<typename T, bool Forward>
void col_dif_execute_ws(std::complex<T>* data,
                        std::size_t axis_extent,
                        std::size_t axis_stride,
                        std::size_t batch_count,
                        T* cc0re, T* cc0im, T* cc1re, T* cc1im,
                        const dif_twiddle_set<T>& dtw,
                        T scale_val = T(1),
                        const std::complex<T>* first_src = nullptr,
                        std::size_t first_src_stride = 0) {
    const std::size_t N = axis_extent;
    if (N <= 1) return;
    // A stride-0 read would transform the first row N times. Throw, not assert: under
    // `NDEBUG` an assert is nothing, exactly the builds where a wrong answer would ship.
    if (first_src != nullptr && first_src_stride == 0)
        throw internal_error("col dif: first_src copy-in requires its own stride");

    const std::size_t B = batch_count;
    const std::size_t n_passes = dtw.radices.size();

    if (n_passes == 1) {
        // Single radix: AoS in-place (ido==1), so the copy-in runs up front. Reachable
        // only at a forced route at tiny N; `select_route` never sends `four_step_large` here.
        if (first_src != nullptr)
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < B; ++j)
                    data[i * axis_stride + j] = first_src[i * first_src_stride + j];
        const std::size_t ip = dtw.radices[0];
        const std::size_t ido = N / ip;
        poet::dispatch(poet::throw_on_no_match, dif_col_pass_fused_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       data, axis_stride, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data(), scale_val);
        return;
    }

    // --- First pass: AoS (fused copy-in source, else `data`) -> SoA (`cc0`) ---
    {
        const std::complex<T>* rd = first_src ? first_src : data;
        const std::size_t rd_stride = first_src ? first_src_stride : axis_stride;
        const std::size_t ip = dtw.radices[0];
        const std::size_t ido = N / ip;  // l1 == 1
        poet::dispatch(poet::throw_on_no_match, dif_col_pass_first_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       rd, rd_stride, cc0re, cc0im, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data());
    }

    // --- Intermediate passes: SoA ping-pong (`cc0` <-> `cc1`) ---
    std::size_t l1 = dtw.radices[0];
    bool ping = false;  // first pass wrote `cc0`; next pass reads `cc0`

    for (std::size_t p = 1; p + 1 < n_passes; ++p) {
        const std::size_t ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * ip);
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;
        T* dst_re = ping ? cc0re : cc1re;
        T* dst_im = ping ? cc0im : cc1im;

        poet::dispatch(poet::throw_on_no_match, dif_col_pass_invoke<T>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       src_re, src_im, dst_re, dst_im, l1, ido, B,
                       dtw.passes[p].first.data(), dtw.passes[p].second.data());

        l1 *= ip;
        ping = !ping;
    }

    // --- Last pass: SoA -> AoS ---
    {
        const std::size_t p = n_passes - 1;
        const std::size_t ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * ip);
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;

        poet::dispatch(poet::throw_on_no_match, dif_col_pass_last_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       src_re, src_im, data, axis_stride, l1, ido, B,
                       dtw.passes[p].first.data(), dtw.passes[p].second.data(), scale_val);
    }
}

// `forward` -> `<Forward>` trampoline and instantiation boundary: the leaves are named
// only here and declared extern below, so the col tree stays out of routing TUs.
// Definitions: `src/inst_col_{f,d}_{fwd,inv}.cpp`.
template<typename T>
void col_dif_dispatch(bool forward, std::complex<T>* data,
                      std::size_t axis_extent, std::size_t axis_stride,
                      std::size_t batch_count, T* cc0re, T* cc0im, T* cc1re, T* cc1im,
                      const dif_twiddle_set<T>& dtw, T scale_val = T(1),
                      const std::complex<T>* first_src = nullptr,
                      std::size_t first_src_stride = 0) {
    if (forward)
        col_dif_execute_ws<T, true>(data, axis_extent, axis_stride, batch_count, cc0re, cc0im,
                                    cc1re, cc1im, dtw, scale_val, first_src, first_src_stride);
    else
        col_dif_execute_ws<T, false>(data, axis_extent, axis_stride, batch_count, cc0re, cc0im,
                                     cc1re, cc1im, dtw, scale_val, first_src, first_src_stride);
}

// One TU per `<Forward>` leaf: `src/inst_col_{f,d}_{fwd,inv}.cpp`.
extern template void col_dif_execute_ws<float, true>(
    std::complex<float>*, std::size_t, std::size_t, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, const std::complex<float>*, std::size_t);
extern template void col_dif_execute_ws<float, false>(
    std::complex<float>*, std::size_t, std::size_t, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, const std::complex<float>*, std::size_t);
extern template void col_dif_execute_ws<double, true>(
    std::complex<double>*, std::size_t, std::size_t, std::size_t, double*, double*, double*,
    double*, const dif_twiddle_set<double>&, double, const std::complex<double>*, std::size_t);
extern template void col_dif_execute_ws<double, false>(
    std::complex<double>*, std::size_t, std::size_t, std::size_t, double*, double*, double*,
    double*, const dif_twiddle_set<double>&, double, const std::complex<double>*, std::size_t);

} // namespace detail
} // namespace admiral
