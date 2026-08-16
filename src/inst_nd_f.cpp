// One explicit instantiation per TU: the route trees compile in parallel. The extern
// declaration in the detail headers plus the definition here is what makes consumers
// reference these symbols instead of re-instantiating a tree.
#include "admiral/detail/nd_plan.hpp"
namespace admiral::detail {
template class nd_runtime_plan<float>;
}
