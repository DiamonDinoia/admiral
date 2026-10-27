#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

// inst_dif_thunks_ip_d.cpp compiles these seven pass bodies.
extern template void dif_tape_step_body<double, 29, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 43, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 59, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 71, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 73, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 83, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);
extern template void dif_tape_step_body<double, 97, true>(const double*, const double*,
    double*, double*, const dif_step<double>&, const dif_rt<double>&);

template auto dif_thunk<double>::chiplet(std::size_t) -> fn_t;

}
}
