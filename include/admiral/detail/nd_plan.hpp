#pragma once

// ============================================================================
// N-D FFT plan: row-column algorithm, a batched 1D transform per axis. Axes run
// innermost-first. The innermost axis (stride 1) runs `plan_impl::execute` verbatim.
// An outer axis runs the SIMD DIF column pass (`col_dif_execute_ws`) on smooth
// lengths, else scalar gather -> `plan_impl` -> scatter. Each axis applies its own
// 1/len on inverse; forward is unscaled.
// ============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "cxx_compat.hpp"  // `ADM_UNLIKELY`, span, `detail::has_single_bit`

#include <admiral/errors.hpp>  // `size_error`

#include "simd.hpp"     // `batch<T>::size` (SIMD-lane block alignment)

#include "dif_col_driver.hpp"  // `col_dif_execute_ws`, `col_dif_dispatch`, `nd_col_block`
#include "cache.hpp"           // `cpu_cache` (the E2 per-core-L3 gate)
#include "math.hpp"            // `is_codelet_supported`
#include "plan.hpp"           // `plan_impl`
#include "scratch.hpp"        // `soa_scratch`
#include "thread_pool.hpp"    // `thread_pool`, `parallel_for` (plan-owned multithreading)
#include "twiddles.hpp"       // `dif_twiddle_set`, `build_dif_twiddle_set`, `dif_factor_plan`
#include "macros.hpp"         // `ADM_ALWAYS_INLINE`

namespace admiral {
namespace detail {

// Product of extents; nullopt on a zero extent or overflow, because a wrapped total
// would reach an array bound downstream.
[[nodiscard]] inline std::optional<std::size_t> extent_product(
    span<const std::size_t> shape) noexcept {
    std::size_t total = 1;
    for (const std::size_t e : shape) {
        if (e == 0 || total > std::numeric_limits<std::size_t>::max() / e) return std::nullopt;
        total *= e;
    }
    return total;
}

// Per-axis state. Exactly one of {dtw, plan} is active: dtw for the batched SIMD DIF
// column pass, plan for the innermost row or scalar fallback. `col_codelet` marks a
// dif-available length in the codelet catalog: the col route then runs the one-call
// column codelet (len <= 64, no scratch) instead of the dif chain.
template<typename T>
struct nd_axis_state {
    std::size_t length = 0;
    bool dif = false;
    bool col_codelet = false;
    dif_twiddle_set<T> dtw;             // active iff dif
    std::optional<plan_impl<T>> plan;   // active iff !dif (row pass or scalar fallback)
};

// `cpu_cache()` and `nd_col_block<T>()` live in `dif_col_driver.hpp` to avoid circular
// `nd_plan` <-> plan includes and to enable reuse by `four_step_large`.

// Radix-4-only factorization (one trailing radix-2 for odd powers): the fallback
// for small-inner `pow2` f32 axes, which avoids register spills.
[[nodiscard]] inline dif_factor_plan build_radix4_plan(std::size_t n) {
    dif_factor_plan p;
    while (n % 4 == 0) { p.push(4); n /= 4; }
    if (n == 2) p.push(2);   // odd power of two: one trailing radix-2 pass
    return p;
}

// E2 col-batch len cap by per-core probed L3 (knob A/B on the three hosts,
// 2026-08-31; Measurer's knob tables fi/mt/out/knobab-*). len 64 pays on rome
// (4 MiB/core; ties everything else) and genoa (5.3 MiB/core; +28% at 2^7 without
// it), and loses on ice (1.5 MiB/core; -18..-20% at 64^2 both modes). Discriminator:
// the col codelet carries no shareable scratch, so at len 64 the tile chain's
// twiddle sets must fit each core's L3 slice. Probe reads, per-class sysfs views:
// ice 48 MiB/32c -> 1.5, rome 16 MiB/4c -> 4, genoa 32 MiB/6c -> 5.3 MiB.
inline constexpr std::size_t kE2Len64MinL3PerCoreBytes = std::size_t{2} << 20;

[[nodiscard]] constexpr std::size_t e2_len_cap_by_l3(std::size_t l3_per_core_bytes) {
    return l3_per_core_bytes >= kE2Len64MinL3PerCoreBytes ? std::size_t{64} : std::size_t{32};
}

// Host-side arm. Without a split count the cache layers can't be relativized; the
// probe classes run the AMD hosts, so enable: an unknown arm keeps len 64.
[[nodiscard]] inline std::size_t e2_len_cap() {
    const cache_bytes& cc = cpu_cache();
    return e2_len_cap_by_l3(cc.l3_cores != 0 ? cc.l3 / cc.l3_cores : cc.l3);
}

// Per-axis state: innermost takes the `plan_impl` row path, outer smooth axes the
// `col_dif_execute_ws` column path, outer non-smooth axes gather -> `plan_impl` -> scatter.
// Small-inner `pow2` f32 exception: radix-8 DIF spills on AVX2's 16 YMM and radix-4's
// extra pass costs less than the spills at small B; f64 (W=4) fits radix-8.
template<typename T>
[[nodiscard]] inline nd_axis_state<T> make_nd_axis_state(std::size_t length, std::size_t inner,
                                                         bool is_forward, bool innermost,
                                                         std::size_t nthreads = 1,
                                                         admiral::effort eff =
                                                             admiral::effort::estimate) {
    nd_axis_state<T> st;
    st.length = length;
    if (length <= 1) {
        // Degenerate axis: identity (size-1 `plan_impl` is a no-op).
        st.plan.emplace(length, is_forward, nthreads, nullptr, eff);
        return st;
    }
    if (!innermost && is_codelet_supported(length)) {
        st.dif = true;
        // 8..cap catalog, small inner: one batched column codelet call per tile
        // replaces the dif chain's log(len) sweeps and SoA scratch (WS-1 E2). Len
        // <= 4 keeps the chain's trivial butterflies; past inner 64 the chain's
        // per-tile slab reuse wins (len x inner A/B sweep, SP-class AVX-512,
        // 2026-08-31); at len 64 the arm is per-core-L3 gated.
        const std::size_t e2_cap = std::min(e2_len_cap(), kFourStepLeafMax);
        st.col_codelet = length >= 8 && length <= e2_cap &&
                         is_codelet_catalog(length) && inner <= 64;
        dif_factor_plan r4;
        const dif_factor_plan* ov = nullptr;
        // f32 only: on f64 the r16 passes spill, but the pass-count saving outweighs
        // the spills at every band width, so forcing radix 4 there loses.
        if constexpr (sizeof(T) == 4) {
            constexpr std::size_t W = xsimd::batch<T>::size;
            // Plan-time proxy for the executed tile, not a bound: nruns pinned to 1
            // (narrower tile) with stride inner as the run length (usually wider);
            // the two errors have opposite signs.
            const bool small_inner =
                (nd_col_block<T>(length, inner, nthreads, /*nruns=*/1) / W) < 4
                || (inner % W) != 0;
            const bool pow2 = detail::has_single_bit(length);
            if (small_inner && pow2) { r4 = build_radix4_plan(length); ov = &r4; }
        }
        // Col form (`fuse_packed`=false): feeds `col_dif_execute_ws` with plain per-pass tables.
        st.dtw = build_dif_twiddle_set<T>(length, ov, /*fuse_packed=*/false);
    }
    // Every strided axis keeps the 1D plan: non-dif axes have no other route, and dif
    // axes fall back per call when `choose_line_route` picks transposed. Run width and
    // thread count arrive at execute time, so both forms stay resident.
    st.plan.emplace(length, is_forward, nthreads, nullptr, eff);
    return st;
}

// Per-axis executors, shared by `nd_apply_axis` and the single-axis `axis_plan`.
// `line_base`(i) is the flat offset of line i. Selection order: `choose_band_form` ranks
// the band pair, then `choose_line_route` ranks one run per call, because the two bands
// of a split differ in width.

// Non-zero `row_stride` means `line_base`(r) == `line_base`(0) + r*`row_stride`, so the whole
// chunk takes one `execute_many`: route and index decode resolve once per chunk, not
// per line.
template<typename T, typename LineBase>
ADM_ALWAYS_INLINE void apply_lines_contiguous(std::complex<T>* data, std::size_t len,
                                              const nd_axis_state<T>& st, std::optional<T> fct,
                                              thread_pool* pool, std::size_t nrows,
                                              std::size_t total_elems, LineBase line_base,
                                              std::size_t row_stride = 0) {
    // The axis plan gets no pool here. Sub-plans that must thread internally own
    // their pool by construction (see `make_nd_axis_state`).
    const exec_options<T> opts{fct};
    parallel_for(pool, nrows, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        if (row_stride) {
            st.plan->execute_many(data + line_base(b), e - b, row_stride, opts);
            return;
        }
        for (std::size_t r = b; r < e; ++r)
            st.plan->execute(span<std::complex<T>>(data + line_base(r), len), opts);
    });
}

// How a single strided run of columns is transformed.
enum class line_route : std::uint8_t {
    col_dif,     // batched SIMD DIF straight down the columns; needs st.dtw
    transposed,  // move the run to contiguous, 1D-plan each column, move back
};

// Columns per transposed sweep: two cache lines (8 columns f64, 16 f32), never more
// than the run. Past that the Zen gather/scatter move loops cliff (each strided line
// stays open for 4 f64 touches and the open-line-stream tracking saturates); ice is
// flat to grp 512 and pays nothing at 8. The cap is the ws2_transprobe grp-sweep
// optimum (fi/ws2-profiler-r1.md r2, 2026-09-01); it is ducc's n_bunch operating
// point class, so the substitute stays a ducc-class blocked copy.
template<typename T>
[[nodiscard]] inline std::size_t transpose_group([[maybe_unused]] std::size_t len,
                                                 std::size_t run_len) {
    return std::min<std::size_t>(run_len, 2 * kCacheLine / sizeof(std::complex<T>));
}

// The transposed form pays when register fill is low: the column chain vectorizes
// over columns, so a run below half a batch leaves lanes idle. A footprint term (slab
// out of cache) reads the thread-scaled tile budget. Some threaded cells are known
// losses, kept because a one-thread gate would forfeit the threaded wins.
template<typename T>
[[nodiscard]] inline line_route choose_line_route(const nd_axis_state<T>& st, std::size_t len,
                                                  std::size_t inner, std::size_t run_len,
                                                  std::size_t nthreads) {
    // Availability first: without a column twiddle set the transposed form is the only
    // route (the non-smooth-length fallback; a necessity, not a preference).
    if (!st.dif) return line_route::transposed;
    // Tile-collapse gate, ST only (WS-2 r2): with the col budget block under two SIMD
    // batches the dif chain's strided boundary passes starve the memory system
    // (IPC 0.61 at the standing cells), and the transposed arm's bulk copies win at
    // the capped group. The gate disarms with nthreads > 1: threaded transposed arms
    // blew up on Zen at grp ~12 (standings-2, genoa 2d 2^11 +101%); threaded
    // transposed economics are a follow-up. Reads (st, len, inner, run_len, nthreads)
    // only — no dst layout (the strides bit-identity contract).
    if (nthreads <= 1 && col_budget_block<T>(len, 1) < 2 * xsimd::batch<T>::size)
        return line_route::transposed;
    if (2 * run_len <= xsimd::batch<T>::size
        && len * inner * sizeof(std::complex<T>) > col_cache_budget(nthreads))
        return line_route::transposed;
    return line_route::col_dif;
}

// Move gw columns at stride inner into buf as gw contiguous runs of len (Gather), or
// back (!Gather). Deliberately scalar: the move is memory-bound.
template<bool Gather, typename T>
void move_run(std::complex<T>* line, std::size_t inner, std::size_t len, std::size_t gw,
              std::complex<T>* buf) {
    for (std::size_t p = 0; p < len; ++p)
        for (std::size_t g = 0; g < gw; ++g) {
            if constexpr (Gather) buf[g * len + p] = line[p * inner + g];
            else line[p * inner + g] = buf[g * len + p];
        }
}

template<typename T, typename LineBase>
ADM_ALWAYS_INLINE void apply_lines_strided(std::complex<T>* data, std::size_t len,
                                           std::size_t inner, bool forward,
                                           const nd_axis_state<T>& st, std::optional<T> fct,
                                           thread_pool* pool, std::size_t nruns,
                                           std::size_t run_len, std::size_t total_elems,
                                           LineBase line_base) {
    // Work is a flat [0,nunits) range of (run, tile) units chunked across threads:
    // an odometer advances run/tile instead of two 64-bit divisions per unit, and the
    // run base recomputes only when run steps.
    const std::size_t nthreads = pool_size(pool);
    if (choose_line_route<T>(st, len, inner, run_len, nthreads) == line_route::col_dif) {
        const std::size_t Bt = nd_col_block<T>(len, run_len, nthreads, nruns);
        const std::size_t ntiles = (run_len + Bt - 1) / Bt;
        const std::size_t nunits = nruns * ntiles;
        const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
        parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
            std::size_t run = b / ntiles, tile = b % ntiles;
            auto* line = data + line_base(run);
            if (st.col_codelet) {
                for (std::size_t u = b; u < e; ++u) {
                    const std::size_t c0 = tile * Bt;
                    const std::size_t bc = std::min(Bt, run_len - c0);
                    col_codelet_dispatch<T>(forward, line + c0, inner, line + c0, inner, bc,
                                            len, scale);
                    if (++tile == ntiles) { tile = 0; line = data + line_base(++run); }
                }
                return;
            }
            soa_scratch<T, 4> sc(len * Bt);
            for (std::size_t u = b; u < e; ++u) {
                const std::size_t c0 = tile * Bt;
                const std::size_t bc = std::min(Bt, run_len - c0);
                col_dif_dispatch<T>(forward, line + c0, len, inner, bc,
                                    sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, scale);
                if (++tile == ntiles) { tile = 0; line = data + line_base(++run); }
            }
        });
        return;
    }
    // Transpose the run into contiguous columns, 1D-plan each, transpose back. One
    // column at a time would re-read each cache line per column; batching reads each
    // line once, and the group cap keeps the scratch L2-resident.
    std::size_t group = transpose_group<T>(len, run_len);
    // Keep ~2 units per thread or a pool runs the axis serially; shrink the move
    // width toward one cache line of complex, below which a group re-reads its lines.
    if (pool && nruns * ((run_len + group - 1) / group) < 2 * nthreads) {
        constexpr std::size_t kLine = kCacheLine / sizeof(std::complex<T>);
        const std::size_t target =
            ((run_len + 2 * nthreads - 1) / (2 * nthreads) + kLine - 1) / kLine * kLine;
        group = std::min(group, std::max(kLine, target));
    }
    const std::size_t ngroups = (run_len + group - 1) / group;
    const std::size_t nunits = nruns * ngroups;
    const exec_options<T> opts{fct};
    parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        // Uninitialized: the gather below fills all gw*len entries before every use.
        soa_scratch<T, 1> scratch(2 * len * group);
        auto* const buf = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t u = b; u < e; ++u) {
            const std::size_t c0 = (u % ngroups) * group;
            const std::size_t gw = std::min(group, run_len - c0);
            auto* const line = data + line_base(u / ngroups) + c0;
            move_run<true>(line, inner, len, gw, buf);
            // The move lands the group as gw contiguous runs of len, so the whole
            // group resolves its route once under uniform stride.
            st.plan->execute_many(buf, gw, len, opts);
            move_run<false>(line, inner, len, gw, buf);
        }
    });
}

// Out-of-place twin of `apply_lines_strided`, with independent src/dst strides. The col
// route needs both batch strides 1: the pass kernels walk columns contiguously, and the
// first pass reads the source straight (the `first_src` of `col_dif`). Every other
// pattern
// takes the transposed route. The route reads the source layout only, because the route
// picks the numbers. Pricing the destination in would make a transform's bits depend
// on where the result lands. dst == src is legal only when the layouts match. A route
// reads a tile fully before it writes that tile. A differing stride pair then makes
// one tile's writes land in the next tile's reads.
template<typename T, typename SrcBase, typename DstBase>
ADM_ALWAYS_INLINE void
apply_lines_strided_oop(const std::complex<T>* src, std::size_t src_line,
                        std::size_t src_batch, std::complex<T>* dst,
                        std::size_t dst_line, std::size_t dst_batch, std::size_t len,
                        bool forward, const nd_axis_state<T>& st, std::optional<T> fct,
                        thread_pool* pool, std::size_t nruns, std::size_t run_len,
                        std::size_t total_elems, SrcBase src_base, DstBase dst_base) {
    const std::size_t nthreads = pool_size(pool);
    if (src_batch == 1 && dst_batch == 1 &&
        choose_line_route<T>(st, len, src_line, run_len, nthreads) ==
            line_route::col_dif) {
        const std::size_t Bt = nd_col_block<T>(len, run_len, nthreads, nruns);
        const std::size_t ntiles = (run_len + Bt - 1) / Bt;
        const std::size_t nunits = nruns * ntiles;
        const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
        parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
            std::size_t run = b / ntiles, tile = b % ntiles;
            const std::complex<T>* sline = src + src_base(run);
            std::complex<T>* dline = dst + dst_base(run);
            if (st.col_codelet) {
                for (std::size_t u = b; u < e; ++u) {
                    const std::size_t c0 = tile * Bt;
                    const std::size_t bc = std::min(Bt, run_len - c0);
                    col_codelet_dispatch<T>(forward, sline + c0, src_line, dline + c0,
                                            dst_line, bc, len, scale);
                    if (++tile == ntiles) {
                        tile = 0;
                        sline = src + src_base(++run);
                        dline = dst + dst_base(run);
                    }
                }
                return;
            }
            soa_scratch<T, 4> sc(len * Bt);
            for (std::size_t u = b; u < e; ++u) {
                const std::size_t c0 = tile * Bt;
                const std::size_t bc = std::min(Bt, run_len - c0);
                col_dif_dispatch<T>(forward, dline + c0, len, dst_line, bc, sc.buf(0),
                                    sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, scale,
                                    sline + c0, src_line);
                if (++tile == ntiles) {
                    tile = 0;
                    sline = src + src_base(++run);
                    dline = dst + dst_base(run);
                }
            }
        });
        return;
    }
    // Transposed route (any strides; the only one without a column chain): gather a
    // cache-resident group, 1D-plan each column, scatter. Same grouping rule as in place.
    std::size_t group = transpose_group<T>(len, run_len);
    if (pool && nruns * ((run_len + group - 1) / group) < 2 * nthreads) {
        constexpr std::size_t kLine = kCacheLine / sizeof(std::complex<T>);
        const std::size_t target =
            ((run_len + 2 * nthreads - 1) / (2 * nthreads) + kLine - 1) / kLine * kLine;
        group = std::min(group, std::max(kLine, target));
    }
    const std::size_t ngroups = (run_len + group - 1) / group;
    const std::size_t nunits = nruns * ngroups;
    const exec_options<T> opts{fct};
    parallel_for(pool, nunits, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        // Uninitialized: the gather fills all gw*len entries before every use.
        soa_scratch<T, 1> scratch(2 * len * group);
        auto* const buf = reinterpret_cast<std::complex<T>*>(scratch.buf(0));
        for (std::size_t u = b; u < e; ++u) {
            const std::size_t c0 = (u % ngroups) * group;
            const std::size_t gw = std::min(group, run_len - c0);
            const std::size_t r = u / ngroups;
            const std::complex<T>* const sline = src + src_base(r) + c0 * src_batch;
            std::complex<T>* const dline = dst + dst_base(r) + c0 * dst_batch;
            for (std::size_t p = 0; p < len; ++p)
                for (std::size_t g = 0; g < gw; ++g)
                    buf[g * len + p] = sline[p * src_line + g * src_batch];
            st.plan->execute_many(buf, gw, len, opts);
            for (std::size_t p = 0; p < len; ++p)
                for (std::size_t g = 0; g < gw; ++g)
                    dline[p * dst_line + g * dst_batch] = buf[g * len + p];
        }
    });
}

// How a pair of disjoint column bands on the same lines is issued.
enum class band_form : std::uint8_t {
    packed,  // both bands gathered into one <= W slab: one pass chain instead of two
    merged,  // equal widths, so 2*nruns runs of a single width fit in one call
    split,   // one call per band (the second is skipped when there is no second band)
};

// Packing costs a gather and a scatter (2 passes over the slab) and saves one whole
// pass chain. The trade pays only for a chain that is long enough. `dif` tests
// availability: with no column chain there is nothing to save.
inline constexpr std::size_t kPackMinPasses = 5;

[[nodiscard]] constexpr band_form choose_band_form(bool dif, std::size_t n_passes,
                                                  std::size_t w0, std::size_t w1,
                                                  std::size_t simd_width) {
    if (w1 == 0) return band_form::split;  // one band; nothing to pair it with
    if (dif && w0 + w1 <= simd_width && n_passes >= kPackMinPasses) return band_form::packed;
    // Equal widths are 2*nruns runs of one `run_len`, so they fit in one call; split,
    // a band of <= W columns is one tile (Bt >= one batch), and two single-unit calls
    // would run serially. Unequal widths cannot merge: `run_len` is one value per call.
    if (w0 == w1) return band_form::merged;
    return band_form::split;
}

// Two disjoint column bands transformed as one packed run: a sub-register band pays a
// whole pass chain regardless of width, so packing into one Bp = w0 + w1 <= W slab
// removes a chain for the cost of a gather/scatter pair (2 slab passes against the
// chain's log(len)). Caller guarantees Bp <= W, st.dif, and disjoint bands.
template<typename T, typename LineBases>
void apply_bands_strided_packed(std::complex<T>* data, std::size_t len, std::size_t inner,
                                bool forward, const nd_axis_state<T>& st, std::optional<T> fct,
                                thread_pool* pool, std::size_t nruns, std::size_t w0,
                                std::size_t w1, std::size_t total_elems, LineBases line_bases) {
    const std::size_t Bp = w0 + w1;
    const T scale = fct.value_or(forward ? T(1) : T(1) / static_cast<T>(len));
    // Bands are complex, the copies are real: 2*w <= 2*W reals, so one `real_run_copy`
    // each, mask built once here rather than per row.
    const auto cp0 = real_run_copy<T>::make(2 * w0);
    const auto cp1 = real_run_copy<T>::make(2 * w1);
    parallel_for(pool, nruns, total_elems, [&](std::size_t b, std::size_t e, std::size_t) {
        soa_scratch<T, 4> sc(len * Bp);
        // Uninitialized: the gather fills all len*Bp entries before every use.
        soa_scratch<T, 1> slab_re(2 * len * Bp);
        auto* const slab = reinterpret_cast<std::complex<T>*>(slab_re.buf(0));
        for (std::size_t r = b; r < e; ++r) {
            const auto [o0, o1] = line_bases(r);
            auto* const l0 = reinterpret_cast<T*>(data + o0);
            auto* const l1 = reinterpret_cast<T*>(data + o1);
            auto* const sl = slab_re.buf(0);
            for (std::size_t p = 0; p < len; ++p) {
                cp0(l0 + 2 * p * inner, sl + 2 * p * Bp);
                cp1(l1 + 2 * p * inner, sl + 2 * p * Bp + 2 * w0);
            }
            // The slab is its own contiguous [len][Bp] tensor: axis stride == Bp.
            col_dif_dispatch<T>(forward, slab, len, Bp, Bp, sc.buf(0), sc.buf(1),
                                sc.buf(2), sc.buf(3), st.dtw, scale);
            for (std::size_t p = 0; p < len; ++p) {
                cp0(sl + 2 * p * Bp, l0 + 2 * p * inner);
                cp1(sl + 2 * p * Bp + 2 * w0, l1 + 2 * p * inner);
            }
        }
    });
}

// One full axis transform in place: total/(len*inner) contiguous slabs of len*inner,
// so a slab's whole inner block is one contiguous column run. inner is the product of
// the faster extents, i.e. the axis stride (1 for innermost).
template<typename T>
void nd_apply_axis(std::complex<T>* data, std::size_t total, std::size_t len,
                   std::size_t inner, bool innermost, bool is_forward,
                   const nd_axis_state<T>& st, std::optional<T> axis_fct,
                   thread_pool* pool = nullptr) {
    if (len <= 1) return;  // identity axis
    const std::size_t outer = total / (len * inner);
    if (innermost)
        apply_lines_contiguous<T>(data, len, st, axis_fct, pool, outer, total,
                                  [len](std::size_t r) { return r * len; }, len);
    else
        apply_lines_strided<T>(data, len, inner, is_forward, st, axis_fct, pool, outer, inner,
                               total, [len, inner](std::size_t r) { return r * (len * inner); });
}

// N-D plan engine. Rank is runtime; per-axis state is built once and reused. The
// per-axis loop is not the hot path, so there is no Dim template.
template<typename T>
class nd_runtime_plan {
    struct M {
        std::vector<std::size_t> shape;
        bool is_forward;
        std::size_t total;
        std::vector<nd_axis_state<T>> axes;
        // Plan-owned workers for the batch loops, built iff nthreads > 1 and some
        // axis' batch loop can thread (see the ctor); single-line shapes instead carry
        // the pool in the axis sub-plan, so a plan owns at most one active pool.
        std::unique_ptr<thread_pool> pool;
    } m;

public:
    // Out-of-line (extern-template): an inline body re-instantiates the route tree in
    // every consumer TU. nthreads drives `execute()`; only a long innermost axis' route
    // choice depends on it. nthreads > 1 builds threading state here and/or inside the
    // axis sub-plans; eff flows to each axis's 1-D engine. nthreads == 0 resolves the
    // auto count from the shape: the wake law when a batch loop can thread, else the
    // single threading-capable axis' own route-aware count.
    nd_runtime_plan(span<const std::size_t> shape, bool is_forward,
                    std::size_t nthreads = 1,
                    admiral::effort eff = admiral::effort::estimate);
    void execute(std::complex<T>* data, const exec_options<T>& opts = {}) const;
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 const exec_options<T>& opts = {}) const;

    [[nodiscard]] std::size_t size() const noexcept { return m.total; }

private:
    // `exec_options::debug` >= `dbg_route` traces here; rank >= 2 only, since rank 1 hands
    // its line to the axis plan, and the axis plans get no debug (a batch loop would
    // print per line). Cold, out of line, split out so the rank-1 arm keeps a leaf frame.
    ADM_NOINLINE void execute_nd(std::complex<T>* data, const exec_options<T>& opts) const;
    ADM_NOINLINE void execute_nd(const std::complex<T>* src, std::complex<T>* dst,
                                 const exec_options<T>& opts) const;

    ADM_NOINLINE ADM_COLD void trace(unsigned level, const char* how) const {
        dbg_print("rank=", m.shape.size(), m.is_forward ? " fwd " : " inv ", how, " total=",
                  m.total, m.pool ? " threaded" : " serial");
        if (level < dbg_shape) return;
        dbg_print_seq("  shape", m.shape);
        for (std::size_t d = 0; d < m.shape.size(); ++d)
            dbg_print("  axis ", d, " len=", m.shape[d], " ",
                      m.axes[d].dif ? "col_dif" : m.axes[d].plan->route_name());
    }

    // Distribute opts.fct across axes. Default: every axis uses nullopt. Custom fct:
    // folded into one axis (innermost with extent > 1); all others use T(1).
    struct scale_plan {
        bool custom;
        T fct;
        std::size_t scale_axis;   // == ndim when the tensor has no extent > 1
    };
    [[nodiscard]] scale_plan make_scale_plan(std::optional<T> fct) const {
        const T def = m.is_forward ? T(1) : T(1) / static_cast<T>(m.total);
        const T f = fct.value_or(def);
        std::size_t axis = m.shape.size();
        if (f != def)
            for (std::size_t di = 0; di < m.shape.size(); ++di) {
                const std::size_t d = m.shape.size() - 1 - di;
                if (m.shape[d] > 1) { axis = d; break; }
            }
        return {f != def, f, axis};
    }
    [[nodiscard]] std::optional<T> axis_fct(const scale_plan& sp, std::size_t d) const {
        if (!sp.custom) return std::nullopt;                 // natural per-axis scale
        return d == sp.scale_axis ? sp.fct : T(1);
    }
};

template<typename T>
nd_runtime_plan<T>::nd_runtime_plan(span<const std::size_t> shape, bool is_forward,
                                    std::size_t nthreads, admiral::effort eff) {
    m.shape.assign(shape.begin(), shape.end());
    m.is_forward = is_forward;
    const auto total = extent_product(m.shape);
    if (!total) ADM_UNLIKELY throw size_error("Plan size must be greater than 0");
    m.total = *total;
    // Every partial product below is <= m.total, so the strides cannot overflow.
    // inner = product of faster extents = this axis' stride (suffix product).
    bool batch_threadable = false;
    for (const std::size_t d : m.shape) {
        if (d == 0) continue;
        // An axis' batch loop threads when it has >= 2 lines over a big-enough tensor.
        const std::size_t units = m.total / d;
        batch_threadable |= units >= 2 && m.total >= kThreadMinElems && d > 1;
    }
    if (nthreads == 0 && batch_threadable) {
        // Wake law (fi/mt t0-modeler-r2.md): K = rank dispatches per execute (one
        // `parallel_for` per non-trivial axis), W = the summed per-axis line work.
        std::size_t dispatches = 0;
        double work_cyc = 0.0;
        for (const std::size_t d : m.shape) {
            if (d <= 1) continue;
            ++dispatches;
            work_cyc += double(m.total / d) * line_work_cyc<T>(d);
        }
        const unsigned cls = m.shape.size() >= 3 ? 2 : 1;
        nthreads = resolve_nthreads(0, m.total, dispatches, work_cyc / core_cyc_per_ns(), cls);
    }
    // !batch_threadable with nthreads == 0: the count stays 0 and the one
    // threading-capable axis' sub-plan resolves its own route-aware count below.
    m.axes.resize(m.shape.size());
    std::size_t inner = 1;
    for (std::size_t di = 0; di < m.shape.size(); ++di) {
        const std::size_t d = m.shape.size() - 1 - di;
        // Plan-time threading split: an axis sub-plan threads internally only when
        // the batch loop above it cannot thread; otherwise it is a 1-thread plan
        // running serially inside `parallel_for`. No per-call pool exists, so this ctor
        // decision is final.
        const std::size_t units = m.total / m.shape[d];
        const bool threads_above = units >= 2 && m.total >= kThreadMinElems;
        const std::size_t axis_threads = threads_above ? 1 : nthreads;
        m.axes[d] = make_nd_axis_state<T>(m.shape[d], inner, is_forward,
                                          /*innermost=*/d == m.shape.size() - 1, axis_threads,
                                          eff);
        inner *= m.shape[d];
    }
    if (nthreads > 1 && batch_threadable)
        m.pool = std::make_unique<thread_pool>(nthreads);
}

// Threads the batch loops on the plan-owned pool (null for serial plans); axis
// sub-plans that thread internally own their own pool.
template<typename T>
void nd_runtime_plan<T>::execute(std::complex<T>* data, const exec_options<T>& opts) const {
    // rank-0 (m.total==1): the empty axis loop plus the degenerate-tensor branch below
    // already cover this path, custom fct included.
    const std::size_t ndim = m.shape.size();
    if (ndim == 1) {
        // Rank-1 goes straight to the axis plan; the generic path adds zero-work
        // layers at nrows==1. A custom fct always lands on the only axis, shape{1} included.
        const scale_plan sp = make_scale_plan(opts.fct);
        m.axes[0].plan->execute(span<std::complex<T>>(data, m.total),
                                {sp.custom ? std::optional<T>(sp.fct) : std::nullopt,
                                 opts.debug});
        return;
    }
    execute_nd(data, opts);
}

template<typename T>
void nd_runtime_plan<T>::execute_nd(std::complex<T>* data, const exec_options<T>& opts) const {
    const std::size_t ndim = m.shape.size();
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "in-place");
    const scale_plan sp = make_scale_plan(opts.fct);
    std::size_t inner = 1;
    for (std::size_t di = 0; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(data, m.total, m.shape[d], inner,
                         /*innermost=*/d == ndim - 1, m.is_forward, m.axes[d],
                         axis_fct(sp, d), m.pool.get());
        inner *= m.shape[d];
    }
    // Degenerate tensor (no extent > 1) with a custom fct: no axis carried it.
    if (sp.custom && sp.scale_axis == ndim) scale_inplace(data, m.total, sp.fct);
}

// src == dst: in-place (same contract as `plan_impl::execute`(p, p)). src != dst: the
// innermost row pass reads src and writes dst (the input copy folds into the threaded
// first pass); later axes run in place on dst. Partial overlap is UB.
template<typename T>
void nd_runtime_plan<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                                 const exec_options<T>& opts) const {
    if (src == dst) { execute(dst, opts); return; }
    if (m.shape.empty()) {   // rank-0: one element, identity; shape[ndim-1] below would be OOB
        const scale_plan sp = make_scale_plan(opts.fct);
        *dst = sp.custom ? *src * sp.fct : *src;
        return;
    }
    if (m.shape.size() == 1) {
        // Rank-1 out-of-place: straight to the axis plan (see the in-place arm).
        const scale_plan sp = make_scale_plan(opts.fct);
        m.axes[0].plan->execute(src, dst,
                                {sp.custom ? std::optional<T>(sp.fct) : std::nullopt,
                                 opts.debug});
        return;
    }
    execute_nd(src, dst, opts);
}

template<typename T>
void nd_runtime_plan<T>::execute_nd(const std::complex<T>* src, std::complex<T>* dst,
                                    const exec_options<T>& opts) const {
    const std::size_t ndim = m.shape.size();
    const std::size_t len = m.shape[ndim - 1];   // innermost extent
    if (opts.debug >= dbg_route) ADM_UNLIKELY trace(opts.debug, "oop");
    const std::size_t rows = m.total / len;
    const nd_axis_state<T>& in_st = m.axes[ndim - 1];
    const scale_plan sp = make_scale_plan(opts.fct);
    const exec_options<T> row_opts{axis_fct(sp, ndim - 1)};
    // Innermost pass src -> dst: a batched lanes-as-lines call per chunk when the row
    // is a catalog size (the per-line loop pays an out-of-line dispatch per row, several
    // times the transform cost at small len). `iterative_dif` writes dst directly; other
    // routes copy the row and transform in place (row hot from copy). A single row runs
    // the batch loop serial-inline, so the axis plan threads internally on its own pool.
    parallel_for(m.pool.get(), rows, m.total, [&](std::size_t b, std::size_t e, std::size_t) {
        // Batched rows pay where the per-line dispatch dominates (len <= 32); at len
        // 64 the per-line codelet_apply out-runs the unzip/batch/zip tile (measured
        // (rows, len) sweep, SP-class AVX-512, 2026-08-31 campaign).
        if (len <= 32 && is_codelet_catalog(len)) {
            const T fct = row_opts.fct.value_or(m.is_forward ? T(1) : T(1) / static_cast<T>(len));
            if (m.is_forward)
                codelet_dispatch_many_oop<T, true >(src + b * len, dst + b * len, e - b,
                                                    len, len, len, fct);
            else
                codelet_dispatch_many_oop<T, false>(src + b * len, dst + b * len, e - b,
                                                    len, len, len, fct);
            return;
        }
        for (std::size_t r = b; r < e; ++r)
            in_st.plan->execute(src + r * len, dst + r * len, row_opts);
    });
    // Remaining (outer) axes: in place on dst.
    std::size_t inner = len;
    for (std::size_t di = 1; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(dst, m.total, m.shape[d], inner,
                         /*innermost=*/false, m.is_forward, m.axes[d],
                         axis_fct(sp, d), m.pool.get());
        inner *= m.shape[d];
    }
    if (sp.custom && sp.scale_axis == ndim) scale_inplace(dst, m.total, sp.fct);
}

extern template class nd_runtime_plan<float>;
extern template class nd_runtime_plan<double>;

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
