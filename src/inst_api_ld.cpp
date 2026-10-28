
#include "cpp_api.hpp"

namespace admiral {

namespace detail {

#ifdef ADM_DISPATCH

plan_state_impl<long double>::plan_state_impl(span<const std::size_t> shape,
                                              const admiral::options& opts)
    : s(shape, opts.nthreads, opts.pin_threads) {}

real_state_impl<long double>::real_state_impl(span<const std::size_t> shape,
                                              const admiral::options& opts)
    : s(shape, opts.nthreads, opts.pin_threads) {}

std::size_t plan_state_impl<long double>::size() const noexcept { return s.size(); }

void plan_state_impl<long double>::run(bool is_forward, std::complex<long double>* data,
                                       const long double* fct) const {
    s.run(is_forward, data, fct);
}

void plan_state_impl<long double>::run(bool is_forward, const std::complex<long double>* src,
                                       std::complex<long double>* dst,
                                       const long double* fct) const {
    s.run(is_forward, src, dst, fct);
}

std::size_t real_state_impl<long double>::real_size() const noexcept { return s.real_size(); }

std::size_t real_state_impl<long double>::cplx_size() const noexcept { return s.cplx_size(); }

void real_state_impl<long double>::forward(const long double* in, std::complex<long double>* out,
                                           std::optional<long double> fct) const {
    s.forward(in, out, fct);
}

void real_state_impl<long double>::inverse(std::complex<long double>* spec, long double* out,
                                           std::optional<long double> fct) const {
    s.inverse(spec, out, fct);
}

template const tier_ops<long double>* tier_entry::operator()(xsimd::default_arch,
                                                             long double*) const noexcept;

}  // namespace detail

#else  // !ADM_DISPATCH: master's text, verbatim

plan_state<long double>::plan_state(span<const std::size_t> shape, const admiral::options& opts)
    : scalar_plan_state(shape, opts.nthreads, opts.pin_threads) {}

real_state<long double>::real_state(span<const std::size_t> shape, const admiral::options& opts)
    : scalar_real_state(shape, opts.nthreads, opts.pin_threads) {}

}

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

#endif

}
