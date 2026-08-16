// The direction-free thunk families of the 1-D DIF engine (mid-pass body plain and
// chiplet, in-place pass, fused pair). They carry the pass tree but not the <Forward>
// leaf, so instantiating them here — once per precision — keeps the per-direction tape
// TUs from compiling the same kernels twice. Numbers: src/CMakeLists.txt.
#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template struct dif_thunk<float>;

} // namespace detail
} // namespace admiral
