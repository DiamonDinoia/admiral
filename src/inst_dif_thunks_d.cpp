#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

template auto dif_thunk<double>::body(std::size_t) -> fn_t;
template auto dif_thunk<double>::fused2(std::size_t, std::size_t) -> fn_t;
template auto dif_thunk<double>::fused3() -> fn_t;

}
}
