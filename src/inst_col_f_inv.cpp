#include "admiral/detail/dif_col_driver.hpp"

namespace admiral {
namespace detail {

template void col_dif_execute_ws<float, false>(
    std::complex<float>*, std::size_t, std::size_t, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, const std::complex<float>*, std::size_t);

}
}
