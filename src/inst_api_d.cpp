// The double half of the exported ABI; see cpp_api.hpp for the definitions.

#include "cpp_api.hpp"

namespace admiral {

template class plan<double>;
template class axis_plan<double>;
template class plan_r2c<double>;
template class plan_r2r<double>;

template void forward<double>(std::span<const std::complex<double>>,
                              std::span<std::complex<double>>, const options&, std::optional<double>);
template void inverse<double>(std::span<const std::complex<double>>,
                              std::span<std::complex<double>>, const options&, std::optional<double>);

template void forward<double>(std::complex<double>*, std::span<const std::size_t>, const options&,
                              std::optional<double>);
template void inverse<double>(std::complex<double>*, std::span<const std::size_t>, const options&,
                              std::optional<double>);

template void forward<double>(const double*, std::complex<double>*, std::span<const std::size_t>,
                              const options&, std::optional<double>);
template void inverse<double>(std::complex<double>*, double*, std::span<const std::size_t>,
                              const options&, std::optional<double>);

}  // namespace admiral
