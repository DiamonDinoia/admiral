
#include "cpp_api.hpp"
#include "admiral/detail/cxx_compat.hpp"

namespace admiral {

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

}
