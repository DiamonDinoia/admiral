#pragma once

// ============================================================================
// N-D FFT plan (row-column algorithm).
//
// An N-D complex FFT on a contiguous row-major tensor is a sequence of batched
// 1D transforms, one per axis — NOT an N-D butterfly. The 1D chiplet/codelet
// layer is reused unchanged; only the per-axis *addressing* is new.
//
//   * Innermost axis (last, contiguous, stride 1): each line is a contiguous
//     complex<T>[len], satisfying plan_impl<T>::execute(span) verbatim. Pure
//     reuse, full 1D SIMD, and plan_impl applies the 1/len inverse scale.
//
//   * Every outer axis (stride > 1): a batched/strided column transform where
//     the contiguous trailing block is the SIMD-lane batch. Smooth lengths
//     (every prime factor <= 11) take the batched SIMD DIF path
//     (col_dif_execute_ws); all other lengths fall back to a scalar
//     per-column gather -> plan_impl::execute -> scatter (correctness first;
//     SIMD batching of the codelet/direct/bluestein routes is a follow-on).
//
// Normalization: every axis applies its own 1/len on the inverse — plan_impl
// does so internally for the row pass and the scalar fallback, and the batched
// DIF path is scaled explicitly here — so the product over axes is the correct
// 1/(prod len) = 1/Ntot N-D inverse scale. The forward transform is unscaled.
//
// Axes are processed innermost-first; the transform is separable so the order
// does not affect the result.
// ============================================================================

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <optional>
#include <span>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <xsimd/xsimd.hpp>     // batch<T>::size (SIMD-lane block alignment)

#include "dif_col_driver.hpp"  // col_dif_execute_ws
#include "math.hpp"            // is_codelet_supported
#include "plan.hpp"           // plan_impl
#include "scratch.hpp"        // soa_scratch
#include "thread_pool.hpp"    // thread_pool, parallel_for (opt-in multithreading)
#include "twiddles.hpp"       // dif_twiddle_set, build_dif_twiddle_set, dif_factor_plan

namespace admiral {
namespace detail {

// Precomputed per-axis execution state. Exactly one of {dtw, plan} is active:
//   dif == true  -> batched SIMD DIF column pass driven by `dtw`.
//   dif == false -> plan_impl path (innermost row pass, or scalar-fallback
//                    column pass for codelet/direct/bluestein lengths).
template<typename T>
struct nd_axis_state {
    std::size_t length = 0;
    bool dif = false;
    dif_twiddle_set<T> dtw;             // active iff dif
    std::optional<plan_impl<T>> plan;   // active iff !dif (row pass or scalar fallback)
};

// cache_bytes, cpu_cache() and nd_col_block<T>() now live in dif_col_driver.hpp
// (next to col_dif_execute_ws, which the tile serves) so the large-N four-step
// route can reuse them without a circular nd_plan <-> plan include.

// Radix-4-only factorization of a power-of-two length (a single leftover radix-2
// for odd powers). Kept as a fallback lever for small-inner pow2 f32 axes.
[[nodiscard]] inline dif_factor_plan build_radix4_plan(std::size_t n) {
    dif_factor_plan p;
    while (n % 4 == 0) { p.push(4); n /= 4; }
    if (n == 2) p.push(2);   // odd power of two: one trailing radix-2 pass
    return p;
}

// Build the per-axis state for an axis of the given length and inner stride.
// `innermost` axes always take the plan_impl row path (contiguous, full 1D SIMD).
// Outer axes with a smooth length take the batched DIF column path; non-smooth
// lengths fall back to the per-column scalar plan_impl path.
//
// Small-inner pow2 f32 exception: when the contiguous `inner` batch is too narrow
// (few column-block SIMD batches) or non-W-aligned, the default radix-8 column DIF
// spills the register file — B-vectorized radix-8 keeps ~30 YMM live (8 arms + 7
// broadcast twiddles) on AVX2's 16, forcing spills. A radix-4-only factorization is spill-free (radix-4 holds only
// ~10 YMM); at small B the extra pass costs less than the radix-8 spill traffic.
// f64 (W=4) already fits radix-8 in 16 YMM, so it keeps the default plan.
//
// (The ido-vectorized column driver is a NO-GO on AVX2 pow2 — ido shrinks by
// radix each pass so only the first pass vectorizes; every later pass runs
// scalar per-column. See docs/nd-perf-frontier.md.)
template<typename T>
[[nodiscard]] inline nd_axis_state<T> make_nd_axis_state(std::size_t length, std::size_t inner,
                                                         bool is_forward, bool innermost) {
    nd_axis_state<T> st;
    st.length = length;
    if (length <= 1) {
        // Degenerate axis: identity. A size-1 plan_impl is a no-op execute().
        st.plan.emplace(length == 0 ? std::size_t{1} : length, is_forward);
        return st;
    }
    if (!innermost && is_codelet_supported(length)) {
        st.dif = true;
        dif_factor_plan r4;
        const dif_factor_plan* ov = nullptr;
        if constexpr (sizeof(T) == 4) {
            constexpr std::size_t W = xsimd::batch<T>::size;
            const bool small_inner = (nd_col_block<T>(length, inner) / W) < 4
                                     || (inner % W) != 0;
            const bool pow2 = (length & (length - 1)) == 0;
            if (small_inner && pow2) { r4 = build_radix4_plan(length); ov = &r4; }
        }
        // Col form: this set feeds col_dif_execute_ws (strided axes), which
        // reads plain per-pass tables and never fuses.
        st.dtw = is_forward ? build_dif_twiddle_set<T, true>(length, ov, /*fuse_packed=*/false)
                            : build_dif_twiddle_set<T, false>(length, ov, /*fuse_packed=*/false);
    }
    // Non-dif axes (innermost row path + non-smooth-length scalar fallback) run
    // the per-column 1D plan_impl; dif axes never touch st.plan, so skip it there.
    if (!st.dif && !st.plan) st.plan.emplace(length, is_forward);
    return st;
}

// Apply one axis transform in place over the whole tensor.
//   total       = product of all extents
//   len         = this axis' extent
//   inner       = product of extents of all *inner* (faster) axes (stride of
//                 this axis). inner == 1 for the innermost axis.
//   innermost   = whether this is the contiguous last axis.
// The tensor decomposes into `total/(len*inner)` contiguous slabs of
// `len*inner` elements; within a slab, axis position p and inner offset c map to
// data[p*inner + c].
template<typename T>
void nd_apply_axis(std::complex<T>* data, std::size_t total, std::size_t len,
                   std::size_t inner, bool innermost, bool is_forward,
                   const nd_axis_state<T>& st, std::optional<T> axis_fct,
                   thread_pool* pool = nullptr) {
    if (len <= 1) return;  // identity axis

    const std::size_t outer = total / (len * inner);
    // Scale this axis applies to its output. nullopt = the natural per-axis
    // default (forward = 1, inverse = 1/len), which keeps the tuned default N-D
    // path byte-identical; a caller distributing a custom fct passes one axis the
    // whole factor and the rest T(1) (unscaled).
    const exec_options<T> axis_opts{.fct = axis_fct};

    if (innermost) {
        // Contiguous row pass: `outer` (== total/len) lines of length len. Each
        // execute() allocates its scratch call-locally -> rows are independent, a
        // drop-in parallel-for with no shared state and no per-row/per-chunk setup.
        parallel_for(pool, outer, total, [&](std::size_t b, std::size_t e, std::size_t) {
            for (std::size_t r = b; r < e; ++r)
                st.plan->execute(std::span<std::complex<T>>(data + r * len, len), axis_opts);
        });
        return;
    }

    if (st.dif) {
        // Batched SIMD column pass, cache-blocked over the contiguous `inner`
        // dimension. Each Bt-wide column tile runs its whole DIF pass chain while
        // L2-resident; axis_stride stays `inner` (columns keep their tensor
        // stride), only batch_count and the slab base shrink per tile.
        //
        // Parallel unit = one (slab, column-tile) pair, flattened so 2D (one slab,
        // many tiles) threads over tiles and 3D+ over slabs*tiles. Scratch is
        // len*Bt, allocated ONCE per chunk inside the body (serial = one chunk =
        // one alloc, as before). Unit order matches the old nested s/c0 loops, so
        // the per-tile math — and the result — is byte-identical.
        // Tile budget shrinks with worker count so the per-thread tiles don't blow
        // the shared L3 (see nd_col_block). pool==nullptr -> serial -> fat tile.
        const std::size_t Bt = nd_col_block<T>(len, inner, pool ? pool->size() : 1);
        const std::size_t ntiles = (inner + Bt - 1) / Bt;
        const std::size_t nunits = outer * ntiles;
        // Scale folded into dif_col_pass_last's store loop (Scale=true) —
        // eliminates a separate full-array *= sweep after the tiles. The default
        // per-axis 1/len (inverse) keeps the tuned instantiation; a T(1) scale
        // takes the unscaled path (default forward, or a non-scaling axis under a
        // distributed custom fct).
        const T col_scale = axis_fct.value_or(is_forward ? T(1) : T(1) / static_cast<T>(len));
        const bool col_scaled = col_scale != T(1);
        parallel_for(pool, nunits, total, [&](std::size_t b, std::size_t e, std::size_t) {
            soa_scratch<T, 4> sc(len * Bt);
            for (std::size_t u = b; u < e; ++u) {
                std::complex<T>* slab = data + (u / ntiles) * (len * inner);
                const std::size_t c0 = (u % ntiles) * Bt;
                const std::size_t bc = std::min(Bt, inner - c0);
                if (is_forward && !col_scaled)
                    col_dif_execute_ws<T, true, false>(slab + c0, len, inner, bc,
                                                       sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw);
                else if (is_forward)
                    col_dif_execute_ws<T, true, true>(slab + c0, len, inner, bc,
                                                      sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, col_scale);
                else if (col_scaled)
                    col_dif_execute_ws<T, false, true>(slab + c0, len, inner, bc,
                                                       sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw, col_scale);
                else
                    col_dif_execute_ws<T, false, false>(slab + c0, len, inner, bc,
                                                        sc.buf(0), sc.buf(1), sc.buf(2), sc.buf(3), st.dtw);
            }
        });
        return;
    }

    // Scalar fallback column pass: gather each column, run the 1D plan (which
    // applies its own 1/len inverse scale), scatter back. Parallel unit = one
    // (slab, column) pair; per-chunk gather buffer (serial = one chunk = one
    // alloc). Unit order matches the old nested s/c loops.
    const std::size_t nunits = outer * inner;
    parallel_for(pool, nunits, total, [&](std::size_t b, std::size_t e, std::size_t) {
        std::vector<std::complex<T>> col(len);
        for (std::size_t u = b; u < e; ++u) {
            std::complex<T>* slab = data + (u / inner) * (len * inner);
            const std::size_t c = u % inner;
            for (std::size_t p = 0; p < len; ++p) col[p] = slab[p * inner + c];
            st.plan->execute(std::span<std::complex<T>>(col.data(), len), axis_opts);
            for (std::size_t p = 0; p < len; ++p) slab[p * inner + c] = col[p];
        }
    });
}

// N-D plan engine. The rank is a runtime property of the shape: per-axis
// twiddles/sub-plans are precomputed once into a vector of axis states and
// reused across execute() calls. The per-axis loop is not the hot path (the 1D
// kernels are), so a runtime rank costs nothing measurable over a compile-time
// one — hence a single runtime engine for every rank, with no Dim template.
template<typename T>
class nd_runtime_plan {
    struct M {
        std::vector<std::size_t> shape;
        bool is_forward;
        std::size_t total;
        std::vector<nd_axis_state<T>> axes;
    } m;

public:
    // Members are defined out-of-line (below) so a single explicit instantiation
    // in c_api.cpp emits them; every other TU sees the extern-template decl and
    // does not re-instantiate the route tree these pull in (measured ~3.1 GiB ->
    // ~0.3 GiB per consumer TU). size() stays inline (trivial).
    nd_runtime_plan(std::span<const std::size_t> shape, bool is_forward);
    void execute(std::complex<T>* data, exec_options<T> opts = {}) const;
    void execute(const std::complex<T>* src, std::complex<T>* dst,
                 exec_options<T> opts = {}) const;

    [[nodiscard]] std::size_t size() const noexcept { return m.total; }

private:
    // Per-axis output scale for a whole-tensor factor `opts.fct`. On the direction
    // default (nullopt, or explicitly forward=1 / inverse=1/Ntot) every axis takes
    // its natural scale (nullopt) — byte-identical to the tuned path. A custom fct
    // is folded into a single running axis (`scale_axis`, the innermost extent > 1)
    // so there is no separate scaling sweep; all other axes run unscaled (T(1)).
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
nd_runtime_plan<T>::nd_runtime_plan(std::span<const std::size_t> shape, bool is_forward) {
    m.shape.assign(shape.begin(), shape.end());
    m.is_forward = is_forward;
    m.total = 1;
    for (auto e : m.shape) {
        if (e == 0) [[unlikely]] {
            throw std::invalid_argument("Plan size must be greater than 0");
        }
        m.total *= e;
    }
    // Per-axis `inner` = product of all faster (inner) extents = this axis'
    // stride. Built innermost-first via a suffix product, matching execute().
    m.axes.resize(m.shape.size());
    std::size_t inner = 1;
    for (std::size_t di = 0; di < m.shape.size(); ++di) {
        const std::size_t d = m.shape.size() - 1 - di;
        m.axes[d] = make_nd_axis_state<T>(m.shape[d], inner, is_forward,
                                          /*innermost=*/d == m.shape.size() - 1);
        inner *= m.shape[d];
    }
}

// pool == nullptr keeps the serial path; a plan-owned pool threads the batch
// loops (see nd_apply_axis).
template<typename T>
void nd_runtime_plan<T>::execute(std::complex<T>* data, exec_options<T> opts) const {
    if (m.total == 0 || m.shape.empty()) return;
    const std::size_t ndim = m.shape.size();
    const scale_plan sp = make_scale_plan(opts.fct);
    std::size_t inner = 1;
    for (std::size_t di = 0; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(data, m.total, m.shape[d], inner,
                         /*innermost=*/d == ndim - 1, m.is_forward, m.axes[d],
                         axis_fct(sp, d), opts.pool);
        inner *= m.shape[d];
    }
    // Degenerate tensor (no extent > 1) with a custom fct: no axis carried it.
    if (sp.custom && sp.scale_axis == ndim)
        for (std::size_t i = 0; i < m.total; ++i) data[i] *= sp.fct;
}

// Out-of-place: read `src` (preserved), write the result to `dst`. The
// innermost row pass reads src and writes dst — folding what would otherwise be
// a separate, serial full-tensor input-preserving copy INTO the (threaded,
// cache-hot) first pass; every later axis runs in place on dst. A caller that
// must keep its input pays no extra RAM sweep. src and dst must not alias.
template<typename T>
void nd_runtime_plan<T>::execute(const std::complex<T>* src, std::complex<T>* dst,
                                 exec_options<T> opts) const {
    if (m.total == 0 || m.shape.empty()) return;
    const std::size_t ndim = m.shape.size();
    const std::size_t len = m.shape[ndim - 1];   // innermost extent
    const std::size_t rows = m.total / len;
    const nd_axis_state<T>& in_st = m.axes[ndim - 1];
    const scale_plan sp = make_scale_plan(opts.fct);
    const exec_options<T> row_opts{.fct = axis_fct(sp, ndim - 1)};
    // Innermost pass, src -> dst, via plan_impl's out-of-place execute: the
    // iterative_dif route reads src and writes dst directly (no input copy at
    // all); other routes copy the row then transform in place on dst (the row
    // is hot from the copy). Same parallel unit as the in-place row pass.
    parallel_for(opts.pool, rows, m.total, [&](std::size_t b, std::size_t e, std::size_t) {
        for (std::size_t r = b; r < e; ++r)
            in_st.plan->execute(src + r * len, dst + r * len, row_opts);
    });
    // Remaining (outer) axes: in place on dst.
    std::size_t inner = len;
    for (std::size_t di = 1; di < ndim; ++di) {
        const std::size_t d = ndim - 1 - di;
        nd_apply_axis<T>(dst, m.total, m.shape[d], inner,
                         /*innermost=*/false, m.is_forward, m.axes[d],
                         axis_fct(sp, d), opts.pool);
        inner *= m.shape[d];
    }
    if (sp.custom && sp.scale_axis == ndim)
        for (std::size_t i = 0; i < m.total; ++i) dst[i] *= sp.fct;
}

#ifndef ADM_INSTANTIATE_ENGINE
extern template class nd_runtime_plan<float>;
extern template class nd_runtime_plan<double>;
#endif

} // namespace detail
} // namespace admiral
