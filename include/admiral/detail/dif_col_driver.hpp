#pragma once

// ============================================================================
// Batched column DIF pass-chain driver — the strided/batched analogue of
// iterative_dif_execute_ws (dif_driver.hpp). Drives a length-`axis_extent`
// Gentleman-Sande transform simultaneously over `batch_count` columns, where the
// columns are the contiguous SIMD-lane dimension.
//
// Used by the N-D row-column driver for every transform axis that is not the
// contiguous innermost axis. One call transforms a single contiguous slab of
// shape [axis_extent][batch_count]:
//
//   data[p * axis_stride + c]   p in [0,axis_extent), c in [0,batch_count)
//
// For a contiguous row-major slab axis_stride == batch_count (consecutive axis
// positions are batch_count apart, and the batch_count columns are contiguous);
// both are passed explicitly to mirror the 1D signature and document intent.
//
// The twiddle set `dtw` is the ordinary 1D dif_twiddle_set built for
// `axis_extent` (build_dif_twiddle_set) — there is no 2D twiddle tensor; each
// axis reuses its own 1D twiddles.
//
// Scratch (cc0/cc1, re/im) is owned by the caller (the N-D plan) and must each
// be >= axis_extent * batch_count elements. Allocation-free. When Scale=false
// (default) the output is un-normalized; the caller applies 1/L per axis. When
// Scale=true the 1/L scale (scale_val) is folded into dif_col_pass_last's stores.
// ============================================================================

#include <algorithm>
#include <complex>
#include <cstddef>

#if defined(__linux__)
#include <unistd.h>            // sysconf(_SC_LEVEL*_CACHE_SIZE)
#elif defined(__APPLE__)
#include <sys/sysctl.h>        // sysctlbyname("hw.l*cachesize")
#endif

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>     // batch<T>::size (SIMD-lane block alignment)

#include "dif_col_pass.hpp"  // dif_col_pass[_first/_last/_fused] + invokers
#include "dif_driver.hpp"    // dif_radix_set
#include "twiddles.hpp"      // dif_twiddle_set

namespace admiral {
namespace detail {

// Byte budget for the strided column DIF tile. A Bt-wide column block's working
// set (len*Bt data + 4 SoA scratch buffers of len*Bt) is processed as a unit so
// its whole DIF pass chain runs without re-striding the full inner-wide slab.
//
// The optimum is a TENSION between two regimes. See docs/mt-performance.md.
//   * Serial: a big tile (~2 MiB, past L2) minimises the strided-access / TLB cost
//     of the 16 KiB column stride — the win is stride amortisation, not L2
//     residency.
//   * Threaded: every worker holds its OWN tile and they all share L3, so the tile
//     must SHRINK with thread count or the aggregate footprint thrashes L3.
// Hence budget = min(serial cap, L3 share / nthreads): fat tiles when serial,
// L3-safe tiles when many threads contend. Both bounds are derived from the
// machine's real cache geometry (cpu_cache), not baked-in constants:
//   * serial cap = L2   — the fat tile that amortises the column stride;
//   * L3 share   = L3/6 — the aggregate column-tile footprint we allow, so the
//     per-thread slice (share/nthreads) stays clear of the shared L3 as workers
//     grow; other machines scale automatically.

// Detected data-cache sizes (bytes), probed ONCE via a function-local static and
// reused: cache geometry is a property of the machine, not the transform. Uses the
// OS-portable cache query for each target (sysconf on Linux, sysctl on macOS/BSD);
// anything else keeps the measured fallbacks. This ONLY sizes cache tiles — the
// SIMD width stays fixed at compile time by -march (no runtime ISA dispatch).
struct cache_bytes { std::size_t l1d, l2, l3; };
[[nodiscard]] inline const cache_bytes& cpu_cache() {
    static const cache_bytes c = [] {
        cache_bytes d{std::size_t{48} << 10, std::size_t{2} << 20, std::size_t{45} << 20};
#if defined(__linux__)
        const auto pos = [](long v, std::size_t fb) { return v > 0 ? std::size_t(v) : fb; };
        d.l1d = pos(::sysconf(_SC_LEVEL1_DCACHE_SIZE), d.l1d);
        d.l2  = pos(::sysconf(_SC_LEVEL2_CACHE_SIZE),  d.l2);
        d.l3  = pos(::sysconf(_SC_LEVEL3_CACHE_SIZE),  d.l3);
#elif defined(__APPLE__)
        const auto rd = [](const char* key, std::size_t fb) {
            std::uint64_t v = 0; std::size_t n = sizeof(v);
            return ::sysctlbyname(key, &v, &n, nullptr, 0) == 0 && v ? std::size_t(v) : fb;
        };
        d.l1d = rd("hw.l1dcachesize", d.l1d);
        d.l2  = rd("hw.l2cachesize",  d.l2);
        d.l3  = rd("hw.l3cachesize",  d.l3);
#endif
        return d;
    }();
    return c;
}

// Column-tile width Bt: largest multiple of the SIMD width whose per-block working
// set fits the (thread-count-adjusted) byte budget, clamped to [W, inner]. When the
// whole inner-wide slab already fits, this returns `inner` — a single untiled pass.
template<typename T>
[[nodiscard]] inline std::size_t nd_col_block(std::size_t len, std::size_t inner,
                                              std::size_t nthreads = 1) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    if (len == 0) return inner;
    const cache_bytes& cc = cpu_cache();
    const std::size_t budget = std::min(cc.l2,
                                        (cc.l3 / 6) / std::max<std::size_t>(1, nthreads));
    // Per column: one complex<T> of data (2*sizeof(T)) + 4 planar scratch reals.
    constexpr std::size_t per_col = 6 * sizeof(T);
    std::size_t bt = budget / (len * per_col);
    // Parallel granularity cap: with workers, keep >= ~4 tiles per worker so
    // the (slab, tile) unit list stays balanced. On small shapes this binds
    // before the byte budget and dominates over fat tiles; large shapes stay
    // budget-bound (the cap only binds when inner/(4*nt) < budget tiles)
    // and 1T is untouched.
    if (nthreads > 1) bt = std::min(bt, inner / (4 * nthreads));
    if (bt < W) return std::min(inner, W);   // never below one SIMD batch
    bt -= bt % W;                             // align to the SIMD lane width
    return std::min(bt, inner);
}

// first_src (default nullptr): when set, the FIRST pass reads its strided AoS
// input from first_src instead of data, while every later pass and the AoS
// write-back still target data. This lets a caller fuse a copy-in into the first
// gather (four_step_large reads the const transform input directly, dropping a
// full-array memmove). Requires n_passes > 1 (multi-radix); all ND callers use
// the default and are unaffected. first_src must share data's [extent][stride]
// layout.
template<typename T, bool Forward, bool Scale = false>
void col_dif_execute_ws(std::complex<T>* data,
                        std::size_t axis_extent,
                        std::size_t axis_stride,
                        std::size_t batch_count,
                        T* cc0re, T* cc0im, T* cc1re, T* cc1im,
                        const dif_twiddle_set<T>& dtw,
                        [[maybe_unused]] T scale_val = T(1),
                        const std::complex<T>* first_src = nullptr) {
    const std::size_t N = axis_extent;
    if (N <= 1) return;

    const std::size_t B = batch_count;
    const std::size_t n_passes = dtw.radices.size();
    // First-pass AoS read source (fused copy-in); nullptr -> in-place on data.
    const std::complex<T>* rd = first_src ? first_src : data;

    if (n_passes == 1) {
        // Single radix: read AoS, write AoS (ido == 1, twiddle trivial). The
        // fused kernel is in-place; first_src (which needs a distinct read/write)
        // is only offered on the multi-pass path, so read from data here.
        const unsigned ip = dtw.radices[0];
        const std::size_t ido = N / static_cast<std::size_t>(ip);
        poet::dispatch(dif_col_pass_fused_invoke<T, Forward, Scale>{scale_val},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       data, axis_stride, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data());
        return;
    }

    // --- First pass: AoS (rd) -> SoA (cc0) ---
    {
        const unsigned ip = dtw.radices[0];
        const std::size_t ido = N / static_cast<std::size_t>(ip);  // l1 == 1
        poet::dispatch(dif_col_pass_first_invoke<T, Forward>{},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       rd, axis_stride, cc0re, cc0im, std::size_t{1}, ido, B,
                       dtw.passes[0].first.data(), dtw.passes[0].second.data());
    }

    // --- Intermediate passes: SoA ping-pong (cc0 <-> cc1) ---
    std::size_t l1 = static_cast<std::size_t>(dtw.radices[0]);
    bool ping = false;  // first pass wrote cc0; next pass reads cc0

    for (std::size_t p = 1; p + 1 < n_passes; ++p) {
        const unsigned ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;
        T* dst_re = ping ? cc0re : cc1re;
        T* dst_im = ping ? cc0im : cc1im;

        poet::dispatch(dif_col_pass_invoke<T, Forward>{},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       src_re, src_im, dst_re, dst_im, l1, ido, B,
                       dtw.passes[p].first.data(), dtw.passes[p].second.data());

        l1 *= static_cast<std::size_t>(ip);
        ping = !ping;
    }

    // --- Last pass: SoA -> AoS ---
    {
        const std::size_t p = n_passes - 1;
        const unsigned ip = dtw.radices[p];
        const std::size_t ido = N / (l1 * static_cast<std::size_t>(ip));
        const T* src_re = ping ? cc1re : cc0re;
        const T* src_im = ping ? cc1im : cc0im;

        poet::dispatch(dif_col_pass_last_invoke<T, Forward, Scale>{scale_val},
                       poet::dispatch_param<dif_radix_set>{static_cast<int>(ip)},
                       src_re, src_im, data, axis_stride, l1, ido, B,
                       dtw.passes[p].first.data(), dtw.passes[p].second.data());
    }
}

} // namespace detail
} // namespace admiral
