#include "admiral/detail/nd_plan.hpp"

// The fast3d seat's driver instantiates here (weak symbols), where its only non-test
// caller does. Below the granule dialect's ISA the seat is if constexpr-dead and no
// cube symbol is emitted, so this TU is text-identical to a driver-less master there.
#include "granule_cube.hpp"

namespace admiral::detail {
template class nd_runtime_plan<float>;
}
