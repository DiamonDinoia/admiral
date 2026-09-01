
#include "cpp_api.hpp"
#include "admiral/detail/cxx_compat.hpp"

namespace admiral {

template class plan<float>;
template class axis_plan<float>;
template class strides_plan<float>;
template class plan_r2c<float>;
template class plan_r2r<float>;

template void forward<float>(span<const std::complex<float>>, span<std::complex<float>>,
                             const options&, std::optional<float>);
template void inverse<float>(span<const std::complex<float>>, span<std::complex<float>>,
                             const options&, std::optional<float>);

template void forward<float>(std::complex<float>*, span<const std::size_t>, const options&,
                             std::optional<float>);
template void inverse<float>(std::complex<float>*, span<const std::size_t>, const options&,
                             std::optional<float>);

template void forward<float>(const float*, std::complex<float>*, span<const std::size_t>,
                             const options&, std::optional<float>);
template void inverse<float>(std::complex<float>*, float*, span<const std::size_t>,
                             const options&, std::optional<float>);

}
