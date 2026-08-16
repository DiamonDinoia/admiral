// See inst_dif_f_fwd.cpp.
#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template void iterative_dif_execute_ws<double, true>(
    const std::complex<double>*, std::complex<double>*, std::size_t, double*, double*, double*, double*,
    const dif_twiddle_set<double>&, double, std::size_t);

template void dif_build_tape<double, true>(dif_twiddle_set<double>&, std::size_t);

} // namespace detail
} // namespace admiral
