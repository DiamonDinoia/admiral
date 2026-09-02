
#include "cpp_api.hpp"
#include "admiral/detail/cxx_compat.hpp"

namespace admiral {

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

}
