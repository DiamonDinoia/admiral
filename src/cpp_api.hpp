// Definitions for <admiral/admiral.hpp>. The public header only forward-declares
// the *_state structs below, so every member that touches one is declared there
// and defined here.
//
// A header, not a TU: inst_api_f.cpp and inst_api_d.cpp each instantiate one
// precision, same as the rest of the engine.
#pragma once

#include "admiral/admiral.hpp"

#include <limits>
#include <vector>
#include "admiral/detail/cxx_compat.hpp"  // ADM_UNLIKELY, span, detail::type_identity_t

#include "admiral/detail/nd_plan.hpp"      // nd_runtime_plan, nd_axis_state, apply_lines_*
#include "admiral/detail/plan.hpp"         // plan_impl, exec_options
#include "admiral/detail/r2r.hpp"          // r2r_plan
#include "admiral/detail/real_fft.hpp"     // nd_real_plan
#include "admiral/detail/thread_pool.hpp"  // thread_pool, resolve_nthreads

namespace admiral {

namespace detail {

// Each state takes the public options aggregate whole rather than one parameter
// per field: it is the only thing a plan is built from, and resolve_nthreads runs
// here because routing depends on the real worker count, not on the 0-means-auto
// sentinel. Threading then lives INSIDE the engine plans
// (plan-owned pools); no state here. options::debug is the one field the engine
// plan does not keep, so each state holds it and replays it per execute.

// nthreads resolution for the auto heuristic needs the transform's element
// count; on overflow the plan constructor reports the bad shape itself, so a
// saturated max (full auto count) is fine here.
[[nodiscard]] inline std::size_t resolve_auto(const admiral::options& opts,
                                              span<const std::size_t> shape) {
    return resolve_nthreads(opts.nthreads, extent_product(shape).value_or(
                                               std::numeric_limits<std::size_t>::max()));
}

template<typename T>
struct plan_state {
    nd_runtime_plan<T> fwd;
    nd_runtime_plan<T> inv;
    unsigned debug;

    plan_state(span<const std::size_t> shape, const admiral::options& opts)
        : fwd{shape, /*is_forward=*/true, resolve_auto(opts, shape), opts.eff},
          inv{shape, /*is_forward=*/false, resolve_auto(opts, shape), opts.eff},
          debug(opts.debug) {}
};

template<typename T>
struct axis_state {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> stride;   // suffix product: stride[d] = prod(shape[>d])
    // The dimensions execute() iterates over: every dimension except the axis,
    // and except the last one too when the axis is strided (there the last
    // dimension is the contiguous run inside one line). Ordered so the last
    // varies fastest, which makes base_of() walk memory forwards.
    std::vector<std::size_t> bd;
    std::size_t axis;
    bool forward;
    bool innermost;                    // axis == ndim-1, so lines are contiguous
    nd_axis_state<T> st;
    std::unique_ptr<thread_pool> pool;
    unsigned debug;

    axis_state(span<const std::size_t> extents, std::size_t ax, bool fwd,
               const admiral::options& opts)
        : shape(extents.begin(), extents.end()), stride(extents.size()), axis(ax), forward(fwd),
          innermost(ax + 1 == extents.size()), debug(opts.debug) {
        if (shape.empty() || axis >= shape.size())
            throw size_error("axis_plan: axis out of range");
        // Validate the full product once; every stride below is a factor of it.
        const auto total = extent_product(shape);
        if (!total)
            throw size_error("axis_plan: extents must be > 0 and their product must fit");
        const std::size_t nthreads = resolve_nthreads(opts.nthreads, *total);
        std::size_t s = 1;
        for (std::size_t di = 0; di < shape.size(); ++di) {
            const std::size_t d = shape.size() - 1 - di;
            stride[d] = s;
            s *= shape[d];
        }
        for (std::size_t d = 0; d < shape.size(); ++d)
            if (d != axis && (innermost || d + 1 != shape.size())) bd.push_back(d);
        // Plan-time threading split, identical to the one in nd_runtime_plan:
        // the sub-plan routes as (and owns the pool of) a threaded plan only
        // when the batch loop in execute() can never thread, since it then runs
        // serially inside/instead of it. prod(shape[d != axis]) bounds that
        // loop's unit count for both the contiguous and the strided form, so a
        // single-line box (finufft with ntrans == 1) hands the axis the real
        // thread count: four_step_large is the only route that reads nthreads
        // and the only one that threads its own passes.
        const std::size_t units = *total / shape[axis];
        const bool threads_above = units >= 2 && *total >= kThreadMinElems;
        // Exactly one live pool per axis plan: the batch loops' when they can
        // thread (>= 2 lines exist), else the sub-plan's (single-line boxes).
        st = make_nd_axis_state<T>(shape[axis], stride[axis], forward, innermost,
                                   threads_above ? 1 : nthreads, opts.eff);
        if (nthreads > 1 && threads_above) pool = std::make_unique<thread_pool>(nthreads);
    }
};

template<typename T>
struct real_state {
    nd_real_plan<T> plan;   // owns its pool when nthreads > 1
    unsigned debug;

    real_state(span<const std::size_t> shape, const admiral::options& opts)
        : plan{shape, resolve_auto(opts, shape), opts.eff}, debug(opts.debug) {}
};

template<typename T>
struct r2r_state {
    r2r_plan<T> plan;   // owns its pool when nthreads > 1 and rows > 1

    r2r_state(std::size_t N, r2r_kind kind, std::size_t rows, const admiral::options& opts)
        : plan{N, kind, rows, opts.eff, resolve_nthreads(opts.nthreads, sat_elems(N, rows))} {}
};

template<typename T>
[[nodiscard]] std::optional<T> as_optional(const T* p) {
    return p ? std::optional<T>{*p} : std::nullopt;
}

}  // namespace detail

// ============================================================================
// One-shot transforms
//
// The plan is discarded after the call, so a measuring effort can never repay
// its own plan-time race: every one-shot routes with effort::estimate and
// opts.eff is ignored here.
// ============================================================================

namespace {

template<typename T>
void one_shot_1d(span<const std::complex<T>> input, span<std::complex<T>> output,
                 bool is_forward, const options& opts, std::optional<T> fct) {
    if (input.size() != output.size()) ADM_UNLIKELY
        throw size_error("Input and output sizes must match");
    if (input.empty()) ADM_UNLIKELY return;
    detail::plan_impl<T>(output.size(), is_forward,
                         detail::resolve_nthreads(opts.nthreads, output.size()), nullptr,
                         effort::estimate)
        .execute(input.data(), output.data(), {fct, opts.debug});
}

template<typename T>
void one_shot_nd(std::complex<T>* data, span<const std::size_t> shape, bool is_forward,
                 const options& opts, std::optional<T> fct) {
    detail::nd_runtime_plan<T>(shape, is_forward, detail::resolve_auto(opts, shape),
                               effort::estimate)
        .execute(data, {fct, opts.debug});
}

}  // namespace

// The type_identity_t wrapper is part of the signature; see admiral.hpp.
#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(detail::type_identity_t<span<const std::complex<T>>> input,
        span<std::complex<T>> output, const options& opts, std::optional<T> fct) {
    one_shot_1d<T>(input, output, /*is_forward=*/true, opts, fct);
}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
inverse(detail::type_identity_t<span<const std::complex<T>>> input,
        span<std::complex<T>> output, const options& opts, std::optional<T> fct) {
    one_shot_1d<T>(input, output, /*is_forward=*/false, opts, fct);
}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(std::complex<T>* data, span<const std::size_t> shape, const options& opts,
        std::optional<T> fct) {
    one_shot_nd<T>(data, shape, /*is_forward=*/true, opts, fct);
}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
inverse(std::complex<T>* data, span<const std::size_t> shape, const options& opts,
        std::optional<T> fct) {
    one_shot_nd<T>(data, shape, /*is_forward=*/false, opts, fct);
}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(const T* in, std::complex<T>* out, span<const std::size_t> shape,
        const options& opts, std::optional<T> fct) {
    detail::nd_real_plan<T>(shape, detail::resolve_auto(opts, shape), effort::estimate)
        .forward(in, out, {fct, opts.debug});
}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
inverse(std::complex<T>* spec, T* out, span<const std::size_t> shape,
        const options& opts, std::optional<T> fct) {
    detail::nd_real_plan<T>(shape, detail::resolve_auto(opts, shape), effort::estimate)
        .inverse(spec, out, {fct, opts.debug});
}

// ============================================================================
// plan
// ============================================================================

template<typename T>
plan<T>::plan(span<const std::size_t> shape, const options& opts)
    : m{std::make_unique<detail::plan_state<T>>(shape, opts)} {}

template<typename T>
plan<T>::~plan() = default;
template<typename T>
plan<T>::plan(plan&&) noexcept = default;
template<typename T>
plan<T>& plan<T>::operator=(plan&&) noexcept = default;

template<typename T>
std::size_t plan<T>::size() const noexcept {
    return m->fwd.size();
}

template<typename T>
void plan<T>::run(bool is_forward, std::complex<T>* data, const T* fct) const {
    const auto& p = is_forward ? m->fwd : m->inv;
    p.execute(data, {detail::as_optional(fct), m->debug});
}

template<typename T>
void plan<T>::run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
                  const T* fct) const {
    const auto& p = is_forward ? m->fwd : m->inv;
    p.execute(src, dst, {detail::as_optional(fct), m->debug});
}

// ============================================================================
// axis_plan
// ============================================================================

template<typename T>
axis_plan<T>::axis_plan(span<const std::size_t> shape, std::size_t axis, bool forward,
                        const options& opts)
    : m{std::make_unique<detail::axis_state<T>>(shape, axis, forward, opts)} {}

template<typename T>
axis_plan<T>::~axis_plan() = default;
template<typename T>
axis_plan<T>::axis_plan(axis_plan&&) noexcept = default;
template<typename T>
axis_plan<T>& axis_plan<T>::operator=(axis_plan&&) noexcept = default;

template<typename T>
void axis_plan<T>::execute(std::complex<T>* data, span<const std::size_t> lo,
                           span<const std::size_t> hi, std::optional<T> fct) const {
    execute_bands(data, lo, hi, 0, 0, fct);
}

template<typename T>
void axis_plan<T>::execute_bands(std::complex<T>* data, span<const std::size_t> lo,
                                 span<const std::size_t> hi, std::size_t lo2_last,
                                 std::size_t hi2_last, std::optional<T> fct) const {
    const std::size_t ndim = m->shape.size();
    const std::size_t len = m->shape[m->axis];
    const std::size_t inner = m->stride[m->axis];

    // Resolve the box (empty span => full extent) and validate. Read through
    // accessors rather than materializing two vectors: this runs per call.
    if ((!lo.empty() && lo.size() != ndim) || (!hi.empty() && hi.size() != ndim))
        throw size_error("axis_plan: box rank must match shape");
    const auto blo = [&](std::size_t d) { return lo.empty() ? std::size_t{0} : lo[d]; };
    const auto bhi = [&](std::size_t d) { return hi.empty() ? m->shape[d] : hi[d]; };
    bool empty_box = false;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (bhi(d) > m->shape[d] || blo(d) > bhi(d))
            throw size_error("axis_plan: box out of range");
        empty_box |= (blo(d) == bhi(d));   // validate every dim before bailing out
    }
    if (blo(m->axis) != 0 || bhi(m->axis) != len)
        throw size_error("axis_plan: transformed axis must be full");
    // A second band rides the same lines as the box, so it differs only on the last
    // dim: non-empty, in range, disjoint from the first band, and not on the axis
    // itself (which must stay whole).
    if (lo2_last != hi2_last) {
        const std::size_t last = ndim - 1;
        if (hi2_last > m->shape[last] || lo2_last > hi2_last || m->axis == last || empty_box)
            throw size_error("axis_plan: second band out of range");
        if (lo2_last < bhi(last) && blo(last) < hi2_last)
            throw size_error("axis_plan: bands overlap");
    }
    if (empty_box || len <= 1) return;

    detail::thread_pool* pool = m->pool.get();

    std::size_t nbatch = 1;
    for (const std::size_t d : m->bd) nbatch *= bhi(d) - blo(d);
    const auto base_of = [&](std::size_t i) {
        std::size_t off = 0;
        for (std::size_t k = m->bd.size(); k-- > 0;) {
            const std::size_t d = m->bd[k];
            const std::size_t e = bhi(d) - blo(d);
            off += (blo(d) + i % e) * m->stride[d];
            i /= e;
        }
        return off;
    };

    if (m->innermost) {   // contiguous rows of length len
        // base_of(i) is base_of(0) + i*len exactly when every batch dim but the slowest
        // is spanned whole: the mixed-radix digits then enumerate consecutive lines,
        // so the run needs neither the per-line index decode nor a per-line dispatch.
        bool dense = true;
        for (std::size_t k = 1; k < m->bd.size(); ++k)
            dense &= blo(m->bd[k]) == 0 && bhi(m->bd[k]) == m->shape[m->bd[k]];
        detail::apply_lines_contiguous<T>(data, len, m->st, fct, pool, nbatch, nbatch * len,
                                          base_of, dense ? len : 0);
        return;
    }
    // Strided columns: the last dim's band is the contiguous run per line.
    const std::size_t last = ndim - 1;
    const std::size_t band = bhi(last) - blo(last);
    const std::size_t c_lo = blo(last);
    const std::size_t w1 = hi2_last - lo2_last;

    // Band form is decided first (see nd_plan.hpp: choose_band_form); each
    // apply_lines_strided call then picks its own line_route from its own run width.
    switch (detail::choose_band_form(m->st.dif, m->st.dtw.radices.size(), band, w1,
                                     xsimd::batch<T>::size)) {
    case detail::band_form::packed:
        detail::apply_bands_strided_packed<T>(
            data, len, inner, m->forward, m->st, fct, pool, nbatch, band, w1,
            nbatch * len * (band + w1), [&](std::size_t r) {
                const std::size_t off = base_of(r);
                return std::array<std::size_t, 2>{off + c_lo, off + lo2_last};
            });
        return;

    case detail::band_form::merged:
        detail::apply_lines_strided<T>(
            data, len, inner, m->forward, m->st, fct, pool, 2 * nbatch, band,
            2 * nbatch * len * band,
            [&](std::size_t r) { return base_of(r / 2) + (r % 2 == 0 ? c_lo : lo2_last); });
        return;

    case detail::band_form::split:
        detail::apply_lines_strided<T>(data, len, inner, m->forward, m->st, fct, pool, nbatch,
                                       band, nbatch * len * band,
                                       [&](std::size_t r) { return base_of(r) + c_lo; });
        if (w1 != 0)
            detail::apply_lines_strided<T>(data, len, inner, m->forward, m->st, fct, pool, nbatch,
                                           w1, nbatch * len * w1,
                                           [&](std::size_t r) { return base_of(r) + lo2_last; });
        return;
    }
}

// ============================================================================
// plan_r2c
// ============================================================================

template<typename T>
plan_r2c<T>::plan_r2c(span<const std::size_t> shape, const options& opts)
    : m{std::make_unique<detail::real_state<T>>(shape, opts)} {}

template<typename T>
plan_r2c<T>::~plan_r2c() = default;
template<typename T>
plan_r2c<T>::plan_r2c(plan_r2c&&) noexcept = default;
template<typename T>
plan_r2c<T>& plan_r2c<T>::operator=(plan_r2c&&) noexcept = default;

template<typename T>
void plan_r2c<T>::forward(const T* in, std::complex<T>* out, std::optional<T> fct) const {
    m->plan.forward(in, out, {fct, m->debug});
}

template<typename T>
void plan_r2c<T>::inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const {
    m->plan.inverse(spec, out, {fct, m->debug});
}

template<typename T>
std::size_t plan_r2c<T>::real_size() const noexcept {
    return m->plan.real_size();
}

template<typename T>
std::size_t plan_r2c<T>::cplx_size() const noexcept {
    return m->plan.cplx_size();
}

// ============================================================================
// plan_r2r
// ============================================================================

template<typename T>
plan_r2r<T>::plan_r2r(std::size_t size, r2r_kind kind, std::size_t rows, const options& opts)
    : m{std::make_unique<detail::r2r_state<T>>(size, kind, rows, opts)} {}

template<typename T>
plan_r2r<T>::~plan_r2r() = default;
template<typename T>
plan_r2r<T>::plan_r2r(plan_r2r&&) noexcept = default;
template<typename T>
plan_r2r<T>& plan_r2r<T>::operator=(plan_r2r&&) noexcept = default;

template<typename T>
void plan_r2r<T>::forward(const T* in, T* out, std::optional<T> fct) const {
    m->plan.forward(in, out, fct);
}

template<typename T>
void plan_r2r<T>::inverse(const T* in, T* out, std::optional<T> fct) const {
    m->plan.inverse(in, out, fct);
}

template<typename T>
std::size_t plan_r2r<T>::size() const noexcept {
    return m->plan.size();
}

}  // namespace admiral
