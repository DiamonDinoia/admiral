#pragma once

#include "admiral/admiral.hpp"

#include <limits>
#include <vector>

#include "admiral/detail/cxx_compat.hpp"
#include "admiral/detail/nd_plan.hpp"
#include "admiral/detail/scratch.hpp"
#include "admiral/detail/plan.hpp"
#include "admiral/detail/r2r.hpp"
#include "admiral/detail/real_fft.hpp"
#include "admiral/detail/scalar_fft.hpp"
#include "admiral/detail/thread_pool.hpp"

namespace admiral {

namespace detail {

template<typename T>
[[nodiscard]] std::optional<T> as_optional(const T* p) {
    return p ? std::optional<T>{*p} : std::nullopt;
}

template<typename T>
inline std::size_t split_batch_threads(std::size_t requested, std::size_t total,
                                       std::size_t lines, std::size_t len,
                                       std::unique_ptr<thread_pool>& pool) {
    if (lines >= 2 && total >= kThreadMinElems) {
        const std::size_t nthreads =
            resolve_nthreads(requested, total, 1,
                             double(lines) * line_work_cyc<T>(len) / core_cyc_per_ns(), 1);
        if (nthreads > 1) pool = std::make_unique<thread_pool>(nthreads);
        return 1;
    }
    return requested;
}

template<typename T>
struct plan_state {
    nd_runtime_plan<T> fwd;
    nd_runtime_plan<T> inv;
    unsigned debug;

    plan_state(span<const std::size_t> shape, const admiral::options& opts)
        : fwd{shape, true, opts.nthreads, opts.eff},
          inv{shape, false, opts.nthreads, opts.eff},
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

template<>
struct plan_state<long double> : scalar_plan_state<long double> {
    plan_state(span<const std::size_t> shape, const admiral::options& opts)
        : scalar_plan_state(shape, opts.nthreads) {}
};

template<typename T>
struct axis_state {
    std::vector<std::size_t> shape;
    std::vector<std::size_t> stride;
    std::vector<std::size_t> bd;
    std::size_t axis;
    bool forward;
    bool innermost;
    nd_axis_state<T> st;
    std::unique_ptr<thread_pool> pool;
    unsigned debug;

    axis_state(span<const std::size_t> extents, std::size_t ax, bool fwd,
               const admiral::options& opts)
        : shape(extents.begin(), extents.end()), stride(extents.size()), axis(ax), forward(fwd),
          innermost(ax + 1 == extents.size()), debug(opts.debug) {
        if (shape.empty() || axis >= shape.size())
            throw size_error("axis_plan: axis out of range");
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
        st = make_nd_axis_state<T>(shape[axis], stride[axis], forward, innermost,
                                   split_batch_threads<T>(opts.nthreads, *total,
                                                          *total / shape[axis], shape[axis],
                                                          pool),
                                   opts.eff);
    }
};

template<typename T>
struct strides_state {
    std::size_t len, nbatch;
    std::size_t in_stride, in_dist, out_stride, out_dist;
    nd_axis_state<T> fwd, inv;
    std::unique_ptr<thread_pool> pool;

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
        const std::size_t axis_threads =
            split_batch_threads<T>(opts.nthreads, *total, n, len_, pool);
        fwd = make_nd_axis_state<T>(len_, istride, true, false,
                                    axis_threads, opts.eff);
        inv = make_nd_axis_state<T>(len_, istride, false, false,
                                    axis_threads, opts.eff);
        const bool need_slab = len_ > 1 && istride != 1 && idist == 1 && odist != 1 &&
                               (slab_route(fwd) || slab_route(inv));
        if (need_slab && pool && opts.nthreads == 0) {
            const std::size_t n1 =
                resolve_nthreads(0, *total, 2,
                                 double(n) * line_work_cyc<T>(len_) / core_cyc_per_ns(), 1);
            if (n1 > 1 && n1 != pool->size()) pool = std::make_unique<thread_pool>(n1);
        }
    }

    [[nodiscard]] bool slab_route(const nd_axis_state<T>& st) const {
        return choose_line_route<T>(st, len, in_stride, nbatch, pool_size(pool.get())) ==
               line_route::col_dif;
    }

    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             std::optional<T> fct) const {
        const nd_axis_state<T>& st = is_forward ? fwd : inv;
        if (src == dst && (in_stride != out_stride || in_dist != out_dist))
            throw size_error("strides_plan: in place requires matching in/out strides");
        if (len <= 1) {
            const T scale = fct.value_or(T(1));
            for (std::size_t l = 0; l < nbatch; ++l)
                *dst = *src * scale, src += in_dist, dst += out_dist;
            return;
        }
        if (in_stride == 1) {
            const exec_options<T> opts{fct};
            if (out_stride == 1) {
                parallel_for(pool.get(), nbatch, len * nbatch,
                             [&](std::size_t b, std::size_t e, std::size_t) {
                                 for (std::size_t r = b; r < e; ++r)
                                     st.plan->execute(src + r * in_dist, dst + r * out_dist,
                                                      opts);
                             });
                return;
            }
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
        if (in_dist == 1 && out_dist != 1 && slab_route(st)) {
            detail::soa_scratch<std::complex<T>, 1> slab_sc(len * nbatch);
            std::complex<T>* const slab = slab_sc.buf(0);
            apply_lines_strided_oop<T>(src, in_stride, in_dist, slab, nbatch,
                                       1, len, is_forward, st, fct,
                                       pool.get(), 1, nbatch, len * nbatch,
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
                                   is_forward, st, fct, pool.get(), 1, nbatch,
                                   len * nbatch, [](std::size_t) { return std::size_t{0}; },
                                   [](std::size_t) { return std::size_t{0}; });
    }
};

template<typename T>
struct real_state {
    nd_real_plan<T> plan;
    unsigned debug;

    real_state(span<const std::size_t> shape, const admiral::options& opts)
        : plan{shape, opts.nthreads, opts.eff}, debug(opts.debug) {}

    void forward(const T* in, std::complex<T>* out, std::optional<T> fct) const {
        plan.forward(in, out, {fct, debug});
    }
    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct) const {
        plan.inverse(spec, out, {fct, debug});
    }
    [[nodiscard]] std::size_t real_size() const noexcept { return plan.real_size(); }
    [[nodiscard]] std::size_t cplx_size() const noexcept { return plan.cplx_size(); }
};

template<>
struct real_state<long double> : scalar_real_state<long double> {
    real_state(span<const std::size_t> shape, const admiral::options& opts)
        : scalar_real_state(shape, opts.nthreads) {}
};

template<typename T>
struct r2r_state {
    r2r_plan<T> plan;

    r2r_state(std::size_t N, r2r_kind kind, std::size_t rows, const admiral::options& opts)
        : plan{N, kind, rows, opts.eff,
               resolve_nthreads(opts.nthreads, sat_elems(N, rows), 1,
                                double(rows) * line_work_cyc<T>(N) / core_cyc_per_ns(), 1)} {}
};

}

namespace {

template<typename T>
void one_shot_1d(span<const std::complex<T>> input, span<std::complex<T>> output,
                 bool is_forward, const options& opts, std::optional<T> fct) {
    if (input.size() != output.size()) ADM_UNLIKELY
        throw size_error("Input and output sizes must match");
    if (input.empty()) ADM_UNLIKELY return;
    if constexpr (std::is_same_v<T, long double>) {
        const std::size_t n = output.size();
        detail::plan_state<T>(span<const std::size_t>(&n, 1), opts)
            .run(is_forward, input.data(), output.data(), fct ? &*fct : nullptr);
    } else {
        detail::plan_impl<T>(output.size(), is_forward, opts.nthreads, nullptr,
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
        detail::nd_runtime_plan<T>(shape, is_forward, opts.nthreads, effort::estimate)
            .execute(data, {fct, opts.debug});
    }
}

}

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(detail::type_identity_t<span<const std::complex<T>>> input,
        span<std::complex<T>> output, const options& opts, std::optional<T> fct) {
    one_shot_1d<T>(input, output, true, opts, fct);
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
    one_shot_1d<T>(input, output, false, opts, fct);
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
    one_shot_nd<T>(data, shape, true, opts, fct);
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
    one_shot_nd<T>(data, shape, false, opts, fct);
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
        detail::nd_real_plan<T>(shape, opts.nthreads, effort::estimate)
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
        detail::nd_real_plan<T>(shape, opts.nthreads, effort::estimate)
            .inverse(spec, out, {fct, opts.debug});
    }
}

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

    if ((!lo.empty() && lo.size() != ndim) || (!hi.empty() && hi.size() != ndim))
        throw size_error("axis_plan: box rank must match shape");
    const auto blo = [&](std::size_t d) { return lo.empty() ? std::size_t{0} : lo[d]; };
    const auto bhi = [&](std::size_t d) { return hi.empty() ? m->shape[d] : hi[d]; };
    bool empty_box = false;
    for (std::size_t d = 0; d < ndim; ++d) {
        if (bhi(d) > m->shape[d] || blo(d) > bhi(d))
            throw size_error("axis_plan: box out of range");
        empty_box |= (blo(d) == bhi(d));
    }
    if (blo(m->axis) != 0 || bhi(m->axis) != len)
        throw size_error("axis_plan: transformed axis must be full");
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

    if (m->innermost) {
        bool dense = true;
        for (std::size_t k = 1; k < m->bd.size(); ++k)
            dense &= blo(m->bd[k]) == 0 && bhi(m->bd[k]) == m->shape[m->bd[k]];
        detail::apply_lines_contiguous<T>(data, len, m->st, fct, pool, nbatch, nbatch * len,
                                          base_of, dense ? len : 0);
        return;
    }
    const std::size_t last = ndim - 1;
    const std::size_t band = bhi(last) - blo(last);
    const std::size_t c_lo = blo(last);
    const std::size_t w1 = hi2_last - lo2_last;

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

}
