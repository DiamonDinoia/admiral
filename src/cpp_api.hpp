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

#include "admiral/detail/cxx_compat.hpp"   // ADM_UNLIKELY, span, detail::type_identity_t
#include "admiral/detail/nd_plan.hpp"      // nd_runtime_plan, nd_axis_state, apply_lines_*
#include "admiral/detail/scratch.hpp"      // make_aligned_buffer (strides_plan slab)
#include "admiral/detail/plan.hpp"         // plan_impl, exec_options
#include "admiral/detail/r2r.hpp"          // r2r_plan
#include "admiral/detail/real_fft.hpp"     // nd_real_plan
#include "admiral/detail/scalar_fft.hpp"   // long double backend states
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
[[nodiscard]] std::optional<T> as_optional(const T* p) {
    return p ? std::optional<T>{*p} : std::nullopt;
}

// Plan-time threading split for the single-axis plans (nd_runtime_plan folds the
// same rule over every axis): the batch loop owns the pool when it can ever
// thread (`lines` >= 2 and enough work), and the axis sub-plan then routes
// 1-threaded; a single-line plan hands the axis the real thread count instead,
// four_step_large being the only route that reads it.
inline std::size_t split_batch_threads(std::size_t requested, std::size_t total,
                                       std::size_t lines,
                                       std::unique_ptr<thread_pool>& pool) {
    const std::size_t nthreads = resolve_nthreads(requested, total);
    if (lines >= 2 && total >= kThreadMinElems) {
        if (nthreads > 1) pool = std::make_unique<thread_pool>(nthreads);
        return 1;
    }
    return nthreads;
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

    [[nodiscard]] std::size_t size() const noexcept { return fwd.size(); }
    void run(bool is_forward, std::complex<T>* data, const T* fct) const {
        (is_forward ? fwd : inv).execute(data, {as_optional(fct), debug});
    }
    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             const T* fct) const {
        (is_forward ? fwd : inv).execute(src, dst, {as_optional(fct), debug});
    }
};

// long double runs the scalar backend, which has no route to choose and no
// trace to print, so opts.eff and opts.debug do not apply. An auto thread count
// fans out the line loops and the 1-D first level, as it does for the engine.
template<>
struct plan_state<long double> : scalar_plan_state<long double> {
    plan_state(span<const std::size_t> shape, const admiral::options& opts)
        : scalar_plan_state(shape, resolve_auto(opts, shape)) {}
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
        std::size_t s = 1;
        for (std::size_t di = 0; di < shape.size(); ++di) {
            const std::size_t d = shape.size() - 1 - di;
            stride[d] = s;
            s *= shape[d];
        }
        for (std::size_t d = 0; d < shape.size(); ++d)
            if (d != axis && (innermost || d + 1 != shape.size())) bd.push_back(d);
        // prod(shape[d != axis]) bounds the batch loop's unit count for both the
        // contiguous and the strided form, so a single-line box (finufft with
        // ntrans == 1) hands the axis the real thread count.
        st = make_nd_axis_state<T>(shape[axis], stride[axis], forward, innermost,
                                   split_batch_threads(opts.nthreads, *total,
                                                       *total / shape[axis], pool),
                                   opts.eff);
    }
};

// Geometry fixed at construction; only the base pointers arrive per call. One
// nd_axis_state per direction (the col twiddles are direction-free, the 1-D
// sub-plan is not), a pool for the batch loop, and the route rules from the
// public docs. The slab below is
// allocated once and overwritten by every call, so a plan serves one call at a
// time; concurrent transforms need one plan each.
template<typename T>
struct strides_state {
    std::size_t len, nbatch;
    std::size_t in_stride, in_dist, out_stride, out_dist;
    nd_axis_state<T> fwd, inv;
    std::unique_ptr<thread_pool> pool;
    // Only the in_dist == 1, out_dist != 1 geometry with the col route uses this:
    // the col chain writes len*nbatch contiguous, then one pass scatters into the
    // strided destination.
    detail::aligned_buffer<std::complex<T>> slab;

    strides_state(std::size_t len_, std::size_t n, std::size_t istride, std::size_t idist,
                  std::size_t ostride, std::size_t odist, const admiral::options& opts)
        : len(len_), nbatch(n), in_stride(istride), in_dist(idist), out_stride(ostride),
          out_dist(odist) {
        if (istride == 0 || ostride == 0 || (n > 1 && (idist == 0 || odist == 0)))
            throw size_error("strides_plan: strides must be nonzero");
        const std::size_t dims[2] = {len_, n};
        const auto total = extent_product(span<const std::size_t>(dims, 2));
        if (!total) throw size_error("strides_plan: len and nbatch must be > 0 and their"
                                     " product must fit");
        const std::size_t axis_threads = split_batch_threads(opts.nthreads, *total, n, pool);
        // The INPUT stride, not max over both sides: the axis state's factoring
        // proxy picks the numbers, and the route rule below promises bits that do
        // not depend on the output layout.
        fwd = make_nd_axis_state<T>(len_, istride, /*is_forward=*/true, /*innermost=*/false,
                                    axis_threads, opts.eff);
        inv = make_nd_axis_state<T>(len_, istride, /*is_forward=*/false, /*innermost=*/false,
                                    axis_threads, opts.eff);
        // Allocate where the route is decided, so a call never allocates and a
        // failure lands at plan time with every other resource error. The gate is
        // the same chooser call run() makes, so the slab exists iff run() reads it.
        if (len_ > 1 && istride != 1 && idist == 1 && odist != 1 &&
            (slab_route(fwd) || slab_route(inv)))
            slab = detail::make_aligned_buffer<std::complex<T>>(*total);
    }

    // The col chain into the slab is worth its extra scatter pass only when the
    // chooser actually picks it; otherwise run() transposes straight into dst.
    [[nodiscard]] bool slab_route(const nd_axis_state<T>& st) const {
        return choose_line_route<T>(st, len, in_stride, nbatch, pool_size(pool.get())) ==
               line_route::col_dif;
    }

    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             std::optional<T> fct) const {
        const nd_axis_state<T>& st = is_forward ? fwd : inv;
        // In place is one layout, not two: the routes below read a tile before they
        // write it, which holds only when the two layouts coincide.
        if (src == dst && (in_stride != out_stride || in_dist != out_dist))
            throw size_error("strides_plan: in place requires matching in/out strides");
        if (len <= 1) {
            // Identity axis: a strided copy, scaled. The default is 1 in both
            // directions, because the inverse's 1/len is 1 at this length.
            const T scale = fct.value_or(T(1));
            for (std::size_t l = 0; l < nbatch; ++l)
                *dst = *src * scale, src += in_dist, dst += out_dist;
            return;
        }
        // Route by INPUT geometry: contiguous input lines run the per-line contiguous
        // engine; unit-dist input columns run the batched SIMD column chain. A given
        // length then produces the same numbers for a given direction regardless of the
        // OUTPUT layout, which is what lets callers compare results across layouts.
        if (in_stride == 1) {
            const exec_options<T> opts{fct};
            if (out_stride == 1) {
                // Contiguous rows on both sides: transform into dst directly.
                parallel_for(pool.get(), nbatch, len * nbatch,
                             [&](std::size_t b, std::size_t e, std::size_t) {
                                 for (std::size_t r = b; r < e; ++r)
                                     st.plan->execute(src + r * in_dist, dst + r * out_dist,
                                                      opts);
                             });
                return;
            }
            // Contiguous input lines, strided output lines: per-line engine into a
            // line-long scratch, then scatter. The scratch is one line, not a slab:
            // this route is not the fast path, it is the numerics-consistency one.
            parallel_for(pool.get(), nbatch, len * nbatch,
                         [&](std::size_t b, std::size_t e, std::size_t) {
                             detail::soa_scratch<std::complex<T>, 1> scratch(len);
                             std::complex<T>* const line = scratch.buf(0);
                             for (std::size_t r = b; r < e; ++r) {
                                 st.plan->execute(src + r * in_dist, line, opts);
                                 for (std::size_t p = 0; p < len; ++p)
                                     dst[p * out_stride + r * out_dist] = line[p];
                             }
                         });
            return;
        }
        // The batched SIMD column pass needs unit dist on BOTH sides. When only the
        // source is dist-contiguous (a transposed output view, say), run the same col
        // chain into the contiguous slab and scatter: one extra strided-write pass
        // buys the col engine's numerics, which callers cross-check against the
        // contiguous run's output. When the chooser prefers the transposed route the
        // slab buys nothing, so fall through and transpose straight into dst.
        if (in_dist == 1 && out_dist != 1 && slab_route(st)) {
            apply_lines_strided_oop<T>(src, in_stride, in_dist, slab.get(), nbatch,
                                       /*dst_batch=*/1, len, is_forward, st, fct,
                                       pool.get(), /*nruns=*/1, nbatch, len * nbatch,
                                       [](std::size_t) { return std::size_t{0}; },
                                       [](std::size_t) { return std::size_t{0}; });
            parallel_for(pool.get(), nbatch, len * nbatch,
                         [&](std::size_t b, std::size_t e, std::size_t) {
                             for (std::size_t l = b; l < e; ++l)
                                 for (std::size_t p = 0; p < len; ++p)
                                     dst[p * out_stride + l * out_dist] = slab[p * nbatch + l];
                         });
            return;
        }
        apply_lines_strided_oop<T>(src, in_stride, in_dist, dst, out_stride, out_dist, len,
                                   is_forward, st, fct, pool.get(), /*nruns=*/1, nbatch,
                                   len * nbatch, [](std::size_t) { return std::size_t{0}; },
                                   [](std::size_t) { return std::size_t{0}; });
    }
};

template<typename T>
struct real_state {
    nd_real_plan<T> plan;   // owns its pool when nthreads > 1
    unsigned debug;

    real_state(span<const std::size_t> shape, const admiral::options& opts)
        : plan{shape, resolve_auto(opts, shape), opts.eff}, debug(opts.debug) {}

    void forward(const T* in, std::complex<T>* out, std::optional<T> fct) const {
        plan.forward(in, out, {fct, debug});
    }
    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const {
        plan.inverse(spec, out, {fct, debug});
    }
    [[nodiscard]] std::size_t real_size() const noexcept { return plan.real_size(); }
    [[nodiscard]] std::size_t cplx_size() const noexcept { return plan.cplx_size(); }
};

// long double: same as plan_state above.
template<>
struct real_state<long double> : scalar_real_state<long double> {
    real_state(span<const std::size_t> shape, const admiral::options& opts)
        : scalar_real_state(shape, resolve_auto(opts, shape)) {}
};

template<typename T>
struct r2r_state {
    r2r_plan<T> plan;   // owns its pool when nthreads > 1 and rows > 1

    r2r_state(std::size_t N, r2r_kind kind, std::size_t rows, const admiral::options& opts)
        : plan{N, kind, rows, opts.eff, resolve_nthreads(opts.nthreads, sat_elems(N, rows))} {}
};

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
    if constexpr (std::is_same_v<T, long double>) {
        // The scalar backend has no plan_impl, so a one-shot builds the state
        // plan<long double> would build and runs it once.
        const std::size_t n = output.size();
        detail::plan_state<T>(span<const std::size_t>(&n, 1), opts)
            .run(is_forward, input.data(), output.data(), fct ? &*fct : nullptr);
    } else {
        detail::plan_impl<T>(output.size(), is_forward,
                             detail::resolve_nthreads(opts.nthreads, output.size()), nullptr,
                             effort::estimate)
            .execute(input.data(), output.data(), {fct, opts.debug});
    }
}

template<typename T>
void one_shot_nd(std::complex<T>* data, span<const std::size_t> shape, bool is_forward,
                 const options& opts, std::optional<T> fct) {
    if constexpr (std::is_same_v<T, long double>) {
        detail::plan_state<T>(shape, opts).run(is_forward, data, fct ? &*fct : nullptr);
    } else {
        detail::nd_runtime_plan<T>(shape, is_forward, detail::resolve_auto(opts, shape),
                                   effort::estimate)
            .execute(data, {fct, opts.debug});
    }
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
    if constexpr (std::is_same_v<T, long double>) {
        detail::real_state<T>(shape, opts).forward(in, out, fct);
    } else {
        detail::nd_real_plan<T>(shape, detail::resolve_auto(opts, shape), effort::estimate)
            .forward(in, out, {fct, opts.debug});
    }
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
    if constexpr (std::is_same_v<T, long double>) {
        detail::real_state<T>(shape, opts).inverse(spec, out, fct);
    } else {
        detail::nd_real_plan<T>(shape, detail::resolve_auto(opts, shape), effort::estimate)
            .inverse(spec, out, {fct, opts.debug});
    }
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
    return m->size();
}

template<typename T>
void plan<T>::run(bool is_forward, std::complex<T>* data, const T* fct) const {
    m->run(is_forward, data, fct);
}

template<typename T>
void plan<T>::run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
                  const T* fct) const {
    m->run(is_forward, src, dst, fct);
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
// strides_plan
// ============================================================================

template<typename T>
strides_plan<T>::strides_plan(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                              std::size_t in_dist, std::size_t out_stride,
                              std::size_t out_dist, const options& opts)
    : m{std::make_unique<detail::strides_state<T>>(len, nbatch, in_stride, in_dist,
                                                   out_stride, out_dist, opts)} {}

template<typename T>
strides_plan<T>::~strides_plan() = default;
template<typename T>
strides_plan<T>::strides_plan(strides_plan&&) noexcept = default;
template<typename T>
strides_plan<T>& strides_plan<T>::operator=(strides_plan&&) noexcept = default;

template<typename T>
void strides_plan<T>::forward(const std::complex<T>* src, std::complex<T>* dst,
                            std::optional<T> fct) const {
    m->run(true, src, dst, fct);
}

template<typename T>
void strides_plan<T>::inverse(const std::complex<T>* src, std::complex<T>* dst,
                            std::optional<T> fct) const {
    m->run(false, src, dst, fct);
}

template<typename T>
std::size_t strides_plan<T>::size() const noexcept {
    return m->len * m->nbatch;
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
    m->forward(in, out, fct);
}

template<typename T>
void plan_r2c<T>::inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const {
    m->inverse(spec, out, fct);
}

template<typename T>
std::size_t plan_r2c<T>::real_size() const noexcept {
    return m->real_size();
}

template<typename T>
std::size_t plan_r2c<T>::cplx_size() const noexcept {
    return m->cplx_size();
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
