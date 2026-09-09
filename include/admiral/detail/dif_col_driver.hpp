#pragma once

// Pass schedule for the column DIF chain over dif_col_pass.hpp.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <utility>

#ifndef ADM_COLDIF_GEO
#define ADM_COLDIF_GEO 0
#endif

#if ADM_COLDIF_GEO
#include <cstdlib>
#endif

#include <admiral/errors.hpp>

#include <poet/poet.hpp>
#include "simd.hpp"

#include "cache.hpp"
#include "dif_col_pass.hpp"
#include "twiddles.hpp"

namespace admiral {
namespace detail {

[[nodiscard]] inline std::size_t col_cache_budget(std::size_t nthreads) {
    const cache_bytes& cc = cpu_cache();
    const std::size_t nt = std::max<std::size_t>(1, nthreads);
    const std::size_t sharing = cc.l3_cores ? std::min(nt, cc.l3_cores) : nt;
    return std::min(cc.l2, (cc.l3 / 6) / sharing);
}

inline constexpr std::size_t kTilesPerWorker = 4;
// Floor on a col_dif tile row (bt elements contiguous at the axis stride). Below it the per-row
// cost (prologue, prefetch ramp, DRAM page open) is not amortized. The wider tile spills the
// scratch out of L2, which pays back only while the rows come from DRAM: hence the L3 test.
inline constexpr std::size_t kColDifMinRowBytes = 2048;

template<typename T>
[[nodiscard]] inline std::size_t col_budget_block(std::size_t len, std::size_t nthreads) {
    constexpr std::size_t per_col = 6 * sizeof(T);
    return len == 0 ? 0 : col_cache_budget(nthreads) / (len * per_col);
}

template<typename T>
[[nodiscard]] inline std::size_t nd_col_block(std::size_t len, std::size_t run_len,
                                              std::size_t nthreads, std::size_t nruns) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (len == 0) return run_len;
    std::size_t bt = col_budget_block<T>(len, nthreads);
    constexpr std::size_t elem = sizeof(std::complex<T>);
    if (len * run_len * nruns * elem > cpu_cache().l3) bt = std::max(bt, kColDifMinRowBytes / elem);
    if (nthreads > 1) {
        const std::size_t tiles = (kTilesPerWorker * nthreads + nruns - 1) / nruns;
        bt = std::min(bt, run_len / tiles);
    }
    constexpr std::size_t line_elems = kCacheLine / sizeof(std::complex<T>);
    const std::size_t gran =
        nthreads > 1 && W < line_elems ? line_elems : W;
    if (bt < gran) return std::min(run_len, gran);
    bt -= bt % gran;
    return std::min(bt, run_len);
}

#if ADM_COLDIF_GEO
// L6 geometry probe arm (default off). Hypothesis: team-r5-shared evidence/l3/memo.md. The
// flat floor under-amortizes the per-row cost it exists for (prologue, prefetch ramp, DRAM
// page open) when the floored row covers < half of its write period (rome 512^3 axis1: a
// 2 KiB row at an 8 KiB period), so when the floor fires the arm widens it toward
// min(period/2, one 4 KiB page). Probe-only override: ADM_COLDIF_BT=<elements> replaces the
// floor outright for the Bt sweep. Plan-time pricing only: routes and bit patterns never
// read this (Bt moves tile boundaries; inst_col_*'s pin keeps per-clone bits fixed).
[[nodiscard]] inline std::size_t coldif_geo_floor_bytes(std::size_t row_period_bytes,
                                                        std::size_t elem) {
    if (const char* e = std::getenv("ADM_COLDIF_BT")) {
        const unsigned long v = std::strtoul(e, nullptr, 10);
        if (v > 0) return v * elem;
    }
    if (row_period_bytes > 2 * kColDifMinRowBytes)
        return std::max(kColDifMinRowBytes,
                        std::min(row_period_bytes / 2, std::size_t{4096}));
    return kColDifMinRowBytes;
}

// nd_col_block with the geo floor substituted; the rest of the schedule is master's.
template<typename T>
[[nodiscard]] inline std::size_t nd_col_block_geo(std::size_t len, std::size_t run_len,
                                                  std::size_t row_period_bytes,
                                                  std::size_t nthreads, std::size_t nruns) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (len == 0) return run_len;
    std::size_t bt = col_budget_block<T>(len, nthreads);
    constexpr std::size_t elem = sizeof(std::complex<T>);
    if (len * run_len * nruns * elem > cpu_cache().l3)
        bt = std::max(bt, coldif_geo_floor_bytes(row_period_bytes, elem) / elem);
    if (nthreads > 1) {
        const std::size_t tiles = (kTilesPerWorker * nthreads + nruns - 1) / nruns;
        bt = std::min(bt, run_len / tiles);
    }
    constexpr std::size_t line_elems = kCacheLine / sizeof(std::complex<T>);
    const std::size_t gran =
        nthreads > 1 && W < line_elems ? line_elems : W;
    if (bt < gran) return std::min(run_len, gran);
    bt -= bt % gran;
    return std::min(bt, run_len);
}
#endif

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
    if (first_src != nullptr && first_src_stride == 0)
        throw internal_error("col dif: first_src copy-in requires its own stride");

    const std::size_t B = batch_count;
    const std::size_t n_passes = dtw.radices.size();

    if (n_passes == 1) {
        if (first_src != nullptr)
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t j = 0; j < B; ++j)
                    data[i * axis_stride + j] = first_src[i * first_src_stride + j];
        const std::size_t ip = dtw.radices[0];
        const std::size_t ido = N / ip;
#if ADM_COLDIF_FIRST
        poet::dispatch(poet::throw_on_no_match, dif_col_pass_fused_staged_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       data, axis_stride, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data(), scale_val);
#else
        poet::dispatch(poet::throw_on_no_match, dif_col_pass_fused_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       data, axis_stride, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data(), scale_val);
#endif
        return;
    }

    {
        const std::complex<T>* rd = first_src ? first_src : data;
        const std::size_t rd_stride = first_src ? first_src_stride : axis_stride;
        const std::size_t ip = dtw.radices[0];
        const std::size_t ido = N / ip;
#if ADM_COLDIF_FIRST
        if (N >= kColdifFirstMinLen) {
            poet::dispatch(poet::throw_on_no_match,
                           dif_col_pass_first_staged_invoke<T, Forward>,
                           poet::dispatch_param<dif_radix_set>{ip},
                           rd, rd_stride, cc0re, cc0im, std::size_t{1}, ido, B,
                           dtw.passes[0].first.data(), dtw.passes[0].second.data());
        } else
#endif
        {
            poet::dispatch(poet::throw_on_no_match, dif_col_pass_first_invoke<T, Forward>,
                           poet::dispatch_param<dif_radix_set>{ip},
                           rd, rd_stride, cc0re, cc0im, std::size_t{1}, ido, B,
                           dtw.passes[0].first.data(), dtw.passes[0].second.data());
        }
    }

    std::size_t l1 = dtw.radices[0];
    bool ping = false;

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

    {
        const std::size_t p = n_passes - 1;
        const std::size_t ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * ip);
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;

#if ADM_COLDIF_DIET
        if (N >= kColdifDietMinLen) {
            poet::dispatch(poet::throw_on_no_match, dif_col_pass_last_staged_invoke<T, Forward>,
                           poet::dispatch_param<dif_radix_set>{ip},
                           src_re, src_im, data, axis_stride, l1, ido, B,
                           dtw.passes[p].first.data(), dtw.passes[p].second.data(), scale_val);
            return;
        }
#endif
        poet::dispatch(poet::throw_on_no_match, dif_col_pass_last_invoke<T, Forward>,
                       poet::dispatch_param<dif_radix_set>{ip},
                       src_re, src_im, data, axis_stride, l1, ido, B,
                       dtw.passes[p].first.data(), dtw.passes[p].second.data(), scale_val);
    }
}

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

}
}
