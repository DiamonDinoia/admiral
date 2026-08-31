// The direction-free thunk families of the 1-D DIF engine, mid-pass body plain
// and chiplet plus in-place pass and fused pair, carry the pass tree. The
// `Forward` leaf stays out. One instantiation per precision here keeps the
// per-direction tape TUs from compiling the same kernels twice. The numbers
// live in `src/CMakeLists.txt`.
#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template struct dif_thunk<float>;

} // namespace detail
} // namespace admiral
