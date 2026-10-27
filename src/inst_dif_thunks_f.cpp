#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template auto dif_thunk<float>::body(std::size_t) -> fn_t;
template auto dif_thunk<float>::fused2(std::size_t, std::size_t) -> fn_t;
template auto dif_thunk<float>::fused3() -> fn_t;

}
}
