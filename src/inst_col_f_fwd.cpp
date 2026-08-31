// One direction of the column DIF engine. The `Forward` leaf is the
// instantiation boundary: a route that references the leaf compiles against a
// declaration only. The direction-free interior instantiates once in
// `inst_dif_thunks_*` (`src/CMakeLists.txt`).
#include "admiral/detail/dif_col_driver.hpp"

namespace admiral {
namespace detail {

template void col_dif_execute_ws<float, true>(
    std::complex<float>*, std::size_t, std::size_t, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, const std::complex<float>*, std::size_t);

} // namespace detail
} // namespace admiral
