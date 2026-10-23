
#include "cpp_api.hpp"

namespace admiral {

namespace detail {

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

}
