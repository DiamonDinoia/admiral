#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

// gcc shares one butterfly clone between the Rader passes of a radix group, and the clone
// changes the codegen of its users. So 29, 43, 59 and 71 (radix 7, also used by the in-place
// passes) stay in this TU, and 23, 47, 67, 89 (radix 11) and 53, 79 (radix 13) stay together.
// 73, 83 and 97 share no clone. They are here to balance the two Rader TUs.
template void dif_tape_step_body<double, 29, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 43, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 59, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 71, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 73, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 83, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
template void dif_tape_step_body<double, 97, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);

template auto dif_thunk<double>::in_place(std::size_t) -> fn_t;

}
}
