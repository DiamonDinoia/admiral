// One direction of the 1-D mixed-radix DIF engine. See `inst_col_f_fwd.cpp`.
#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template void iterative_dif_execute_ws<float, true>(
    const std::complex<float>*, std::complex<float>*, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, std::size_t);

template void dif_build_tape<float, true>(dif_twiddle_set<float>&, std::size_t);

} // namespace detail
} // namespace admiral
