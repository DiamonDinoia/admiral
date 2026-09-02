
#include "cpp_api.hpp"

namespace admiral {

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
