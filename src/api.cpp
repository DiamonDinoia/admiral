// The public API, built once at the floor ISA. Each call reaches the engine tier that
// xsimd::dispatch picked on the first call for its precision; a plan keeps that tier's state.
// The per-arch tier map is src/tiers/<arch>.hpp (tier_archs for this file, ADM_TIER_TABLE
// for the test files); the flags of each tier are in cmake/tiers/<arch>.cmake.

#include "api_tier.hpp"

#include "tiers/tier_map.hpp"

#include <xsimd/xsimd.hpp>

namespace admiral {

namespace {

#if defined(ADM_DISPATCH) && defined(ADM_TIER_MAP)
using detail::tier_archs;
#else
using tier_archs = xsimd::arch_list<xsimd::default_arch>;
#endif

template<typename T>
const detail::tier_ops<T>& ops() {
    return *detail::tier_slot<T>();
}

}

namespace detail {

template<typename T>
const tier_ops<T>*& tier_slot() {
    static const tier_ops<T>* p =
        xsimd::dispatch<tier_archs>(tier_entry{})(static_cast<T*>(nullptr));
    return p;
}

template const tier_ops<float>*& tier_slot<float>();
template const tier_ops<double>*& tier_slot<double>();
template const tier_ops<long double>*& tier_slot<long double>();

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
    ops<T>().one_1d(input, output, true, opts, fct);
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
    ops<T>().one_1d(input, output, false, opts, fct);
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
    ops<T>().one_nd(data, shape, true, opts, fct);
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
    ops<T>().one_nd(data, shape, false, opts, fct);
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
    ops<T>().one_r2c(in, out, shape, opts, fct);
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
    ops<T>().one_c2r(spec, out, shape, opts, fct);
}

template<typename T>
plan<T>::plan(span<const std::size_t> shape, const options& opts)
    : m{ops<T>().plan(shape, opts)} {}

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
    : m{ops<T>().axis(shape, axis, forward, opts)} {}

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
    m->execute_bands(data, lo, hi, lo2_last, hi2_last, fct);
}

template<typename T>
strides_plan<T>::strides_plan(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                              std::size_t in_dist, std::size_t out_stride,
                              std::size_t out_dist, const options& opts)
    : m{ops<T>().strides(len, nbatch, in_stride, in_dist, out_stride, out_dist, opts)} {}

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
    return m->size();
}

template<typename T>
plan_r2c<T>::plan_r2c(span<const std::size_t> shape, const options& opts)
    : m{ops<T>().real(shape, opts)} {}

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
    : m{ops<T>().r2r(size, kind, rows, opts)} {}

template<typename T>
plan_r2r<T>::~plan_r2r() = default;
template<typename T>
plan_r2r<T>::plan_r2r(plan_r2r&&) noexcept = default;
template<typename T>
plan_r2r<T>& plan_r2r<T>::operator=(plan_r2r&&) noexcept = default;

template<typename T>
void plan_r2r<T>::forward(const T* in, T* out, std::optional<T> fct) const {
    m->forward(in, out, fct);
}

template<typename T>
void plan_r2r<T>::inverse(const T* in, T* out, std::optional<T> fct) const {
    m->inverse(in, out, fct);
}

template<typename T>
std::size_t plan_r2r<T>::size() const noexcept {
    return m->size();
}


template class ADM_API_INST plan<float>;
template class ADM_API_INST axis_plan<float>;
template class ADM_API_INST strides_plan<float>;
template class ADM_API_INST plan_r2c<float>;
template class ADM_API_INST plan_r2r<float>;

template ADM_API_INST void forward<float>(span<const std::complex<float>>,
                                          span<std::complex<float>>, const options&,
                                          std::optional<float>);
template ADM_API_INST void inverse<float>(span<const std::complex<float>>,
                                          span<std::complex<float>>, const options&,
                                          std::optional<float>);

template ADM_API_INST void forward<float>(std::complex<float>*, span<const std::size_t>,
                                          const options&, std::optional<float>);
template ADM_API_INST void inverse<float>(std::complex<float>*, span<const std::size_t>,
                                          const options&, std::optional<float>);

template ADM_API_INST void forward<float>(const float*, std::complex<float>*,
                                          span<const std::size_t>, const options&,
                                          std::optional<float>);
template ADM_API_INST void inverse<float>(std::complex<float>*, float*, span<const std::size_t>,
                                          const options&, std::optional<float>);


template class ADM_API_INST plan<double>;
template class ADM_API_INST axis_plan<double>;
template class ADM_API_INST strides_plan<double>;
template class ADM_API_INST plan_r2c<double>;
template class ADM_API_INST plan_r2r<double>;

template ADM_API_INST void forward<double>(span<const std::complex<double>>,
                                          span<std::complex<double>>, const options&,
                                          std::optional<double>);
template ADM_API_INST void inverse<double>(span<const std::complex<double>>,
                                          span<std::complex<double>>, const options&,
                                          std::optional<double>);

template ADM_API_INST void forward<double>(std::complex<double>*, span<const std::size_t>,
                                          const options&, std::optional<double>);
template ADM_API_INST void inverse<double>(std::complex<double>*, span<const std::size_t>,
                                          const options&, std::optional<double>);

template ADM_API_INST void forward<double>(const double*, std::complex<double>*,
                                          span<const std::size_t>, const options&,
                                          std::optional<double>);
template ADM_API_INST void inverse<double>(std::complex<double>*, double*, span<const std::size_t>,
                                          const options&, std::optional<double>);


template class ADM_API_INST plan<long double>;
template class ADM_API_INST plan_r2c<long double>;

template ADM_API_INST void forward<long double>(span<const std::complex<long double>>,
                                                span<std::complex<long double>>, const options&,
                                                std::optional<long double>);
template ADM_API_INST void inverse<long double>(span<const std::complex<long double>>,
                                                span<std::complex<long double>>, const options&,
                                                std::optional<long double>);

template ADM_API_INST void forward<long double>(std::complex<long double>*, span<const std::size_t>,
                                                const options&, std::optional<long double>);
template ADM_API_INST void inverse<long double>(std::complex<long double>*, span<const std::size_t>,
                                                const options&, std::optional<long double>);

template ADM_API_INST void forward<long double>(const long double*, std::complex<long double>*,
                                                span<const std::size_t>, const options&,
                                                std::optional<long double>);
template ADM_API_INST void inverse<long double>(std::complex<long double>*, long double*,
                                                span<const std::size_t>, const options&,
                                                std::optional<long double>);

}
