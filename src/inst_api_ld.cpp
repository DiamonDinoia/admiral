
#include "cpp_api.hpp"

namespace admiral {

template class plan<long double>;
template class plan_r2c<long double>;

template void forward<long double>(span<const std::complex<long double>>,
                                   span<std::complex<long double>>, const options&,
                                   std::optional<long double>);
template void inverse<long double>(span<const std::complex<long double>>,
                                   span<std::complex<long double>>, const options&,
                                   std::optional<long double>);

template void forward<long double>(std::complex<long double>*, span<const std::size_t>,
                                   const options&, std::optional<long double>);
template void inverse<long double>(std::complex<long double>*, span<const std::size_t>,
                                   const options&, std::optional<long double>);

template void forward<long double>(const long double*, std::complex<long double>*,
                                   span<const std::size_t>, const options&,
                                   std::optional<long double>);
template void inverse<long double>(std::complex<long double>*, long double*,
                                   span<const std::size_t>, const options&,
                                   std::optional<long double>);

}
