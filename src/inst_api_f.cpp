// The float half of the exported ABI; see cpp_api.hpp for the definitions.

#include "cpp_api.hpp"

namespace admiral {

template class plan<float>;
template class axis_plan<float>;
template class plan_r2c<float>;
template class plan_r2r<float>;

template void forward<float>(std::span<const std::complex<float>>, std::span<std::complex<float>>,
                             const options&, std::optional<float>);
template void inverse<float>(std::span<const std::complex<float>>, std::span<std::complex<float>>,
                             const options&, std::optional<float>);

template void forward<float>(std::complex<float>*, std::span<const std::size_t>, const options&,
                             std::optional<float>);
template void inverse<float>(std::complex<float>*, std::span<const std::size_t>, const options&,
                             std::optional<float>);

template void forward<float>(const float*, std::complex<float>*, std::span<const std::size_t>,
                             const options&, std::optional<float>);
template void inverse<float>(std::complex<float>*, float*, std::span<const std::size_t>,
                             const options&, std::optional<float>);

}  // namespace admiral
