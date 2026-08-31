// One explicit instantiation per TU: the route trees compile in parallel. The
// extern declarations in the detail headers keep consumers from
// re-instantiating a tree; the definition here supplies the symbol.
#include "admiral/detail/nd_plan.hpp"
namespace admiral::detail {
template class nd_runtime_plan<float>;
}
