// See `inst_dif_f_fwd.cpp`.
#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template void iterative_dif_execute_ws<float, false>(
    const std::complex<float>*, std::complex<float>*, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, std::size_t);

template void dif_build_tape<float, false>(dif_twiddle_set<float>&, std::size_t);

} // namespace detail
} // namespace admiral
