// See `inst_gt_f.cpp`.
#include "admiral/detail/good_thomas.hpp"
namespace admiral::detail {

template void good_thomas_run<double, true>(const std::complex<double>*, std::complex<double>*, std::size_t) noexcept;
template void good_thomas_run<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t) noexcept;

}
