// See `inst_col_f_fwd.cpp`.
#include "admiral/detail/dif_col_driver.hpp"

namespace admiral {
namespace detail {

template void col_dif_execute_ws<double, true>(
    std::complex<double>*, std::size_t, std::size_t, std::size_t, double*, double*, double*, double*,
    const dif_twiddle_set<double>&, double, const std::complex<double>*, std::size_t);

} // namespace detail
} // namespace admiral
