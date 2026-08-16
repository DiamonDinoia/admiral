#pragma once

// ============================================================================
// Batched column DIF pass-chain driver — strided/batched analogue of
// iterative_dif_execute_ws (dif_driver.hpp). One call transforms a slab of
// shape [axis_extent][batch_count] over batch_count contiguous SIMD-lane columns:
//
//   data[p * axis_stride + c]   p in [0,axis_extent), c in [0,batch_count)
//
// axis_stride == batch_count for a contiguous row-major slab. dtw is the
// ordinary 1D dif_twiddle_set for axis_extent; each axis has its own 1D twiddles.
// Scratch cc0/cc1 (re/im) >= axis_extent*batch_count, owned by the caller.
// scale_val folds 1/L into dif_col_pass_last's stores; it is 1 for the
// un-normalized direction, and x*1 is bitwise identity.
// ============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include <poet/poet.hpp>
#include "simd.hpp"     // batch<T>::size (SIMD-lane block alignment)

#include "cache.hpp"         // cache_bytes, cpu_cache
#include "dif_col_pass.hpp"  // dif_col_pass[_first/_last/_fused] + invokers
#include "twiddles.hpp"      // dif_twiddle_set, dif_radix_set

namespace admiral {
namespace detail {

// Bytes one column tile may plan against. A Bt-wide block (len*Bt data + 4 SoA
// scratch) runs its whole DIF chain as a unit, so the slab is re-strided once:
//   * serial, a fat tile (~L2) amortises the column-stride TLB cost;
//   * threaded, the tile must shrink with the thread count or the aggregate
//     footprint thrashes L3 -- /6 leaves room for the other axes and the scratch.
// Geometry comes from cpu_cache, not from baked-in constants. choose_line_route
// compares the whole strided slab against this same number, so the tile width and the
// route agree on what "stays cached" means.
[[nodiscard]] inline std::size_t col_cache_budget(std::size_t nthreads) {
    const cache_bytes& cc = cpu_cache();
    // l3 is one sharing-group's L3, so the per-worker share divides by the threads
    // sharing THAT group, not by a machine-wide count, which badly underbuys the tile
    // budget on many-group machines. Unknown group size (l3_cores==0) falls back to
    // nthreads.
    const std::size_t nt = std::max<std::size_t>(1, nthreads);
    const std::size_t sharing = cc.l3_cores ? std::min(nt, cc.l3_cores) : nt;
    return std::min(cc.l2, (cc.l3 / 6) / sharing);
}

// Tiles per worker the balance term aims for. Reachable only by SPLITTING a run, so it
// buys nothing once the runs alone outnumber the workers -- see nd_col_block.
inline constexpr std::size_t kTilesPerWorker = 4;

// Column-tile width Bt: widest W-multiple fitting the thread-adjusted budget, clamped
// to [W, run_len] for `nruns` runs of `run_len` columns. Returns run_len when the whole
// run fits one untiled pass.
//
// Widest-that-fits: tile time per column falls with the width and flattens at the
// budget, and an uneven cut is not penalised -- nothing to gain by balancing the tiles
// against each other.
//
// The balance term counts UNITS = nruns*ntiles, not ntiles: splitting a run is only one
// of the two ways work reaches a worker. Sized off the run alone the cap collapses to
// the SIMD floor on every axis with run_len < kTilesPerWorker*nthreads*W.
// Widest column block whose (len x Bt) slab fits the thread-adjusted cache budget.
// Unlike nd_col_block this depends only on the axis and the thread count, never on a
// call's band geometry, so it is the only column width knowable at plan time.
template<typename T>
[[nodiscard]] inline std::size_t col_budget_block(std::size_t len, std::size_t nthreads) {
    // Per column: one complex<T> of data (2*sizeof(T)) + 4 planar scratch reals.
    constexpr std::size_t per_col = 6 * sizeof(T);
    return len == 0 ? 0 : col_cache_budget(nthreads) / (len * per_col);
}

// `nruns` is NOT defaulted on purpose: it changes the balance cap by a factor of
// kTilesPerWorker*nthreads, and a caller that omits it silently gets the narrowest
// possible tile. Callers that cannot know it want col_budget_block, not a stand-in value
// here.
template<typename T>
[[nodiscard]] inline std::size_t nd_col_block(std::size_t len, std::size_t run_len,
                                              std::size_t nthreads, std::size_t nruns) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (len == 0) return run_len;
    std::size_t bt = col_budget_block<T>(len, nthreads);
    if (nthreads > 1) {
        // Tiles each run must yield for the units to reach kTilesPerWorker per worker;
        // one (a no-op cap) as soon as the runs alone get there.
        const std::size_t tiles = (kTilesPerWorker * nthreads + nruns - 1) / nruns;
        bt = std::min(bt, run_len / tiles);
    }
    // Threaded, tiles get line-sized granules: adjacent (run, tile) units handed
    // to different workers then never share a coherence line. At
    // W*2*sizeof(T) >= line bytes (AVX-512, both precisions) gran == W: no-op.
    constexpr std::size_t line_elems = kCacheLine / sizeof(std::complex<T>);
    const std::size_t gran =
        nthreads > 1 && W < line_elems ? line_elems : W;  // both pow2
    if (bt < gran) return std::min(run_len, gran);  // never below one SIMD batch
    bt -= bt % gran;                                // align to the granule
    return std::min(bt, run_len);
}

// first_src: when set, the first pass reads AoS from first_src instead of data
// (fused copy-in; four_step_large uses this to skip a full-array memmove).
// ND callers use the default (nullptr). first_src shares data's extent but
// carries its own stride, so the destination may be padded.
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
    // Defaulted stride is only a placeholder for the no-copy-in case; reading the
    // source at stride 0 would silently transform the first row N times. A throw, not an
    // assert: an assert is nothing under NDEBUG -- exactly the builds where a wrong answer
    // would ship. One entry branch against N*B element work.
    if (first_src != nullptr && first_src_stride == 0)
        throw std::invalid_argument("col dif: first_src copy-in requires its own stride");

    const std::size_t B = batch_count;
    const std::size_t n_passes = dtw.radices.size();

    if (n_passes == 1) {
        // Single radix: the pass is AoS in-place (ido==1), so the copy-in cannot ride
        // along inside it -- do it up front. Only reachable for a forced route at tiny
        // N (a single radix caps at 32); select_route never sends four_step_large here.
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

    // --- First pass: AoS (fused copy-in source, else data) -> SoA (cc0) ---
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

    // --- Intermediate passes: SoA ping-pong (cc0 <-> cc1) ---
    std::size_t l1 = dtw.radices[0];
    bool ping = false;  // first pass wrote cc0; next pass reads cc0

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

// forward -> compile-time <Forward> trampoline, and the column engine's instantiation
// boundary: the two leaves are named nowhere but here and are declared extern below, so
// the whole col tree stays out of every TU that merely uses an N-D plan. Spelled out
// rather than variadic-forwarding so there is a signature to declare.
// Definitions: src/inst_col_{f,d}_{fwd,inv}.cpp, one per leaf.
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

// Instantiation boundary is the <Forward> leaf, one TU per direction:
// src/inst_col_{f,d}_{fwd,inv}.cpp. Measurements: src/CMakeLists.txt.
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
