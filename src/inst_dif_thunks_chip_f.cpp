#include "admiral/detail/dif_driver.hpp"

namespace admiral {
namespace detail {

// inst_dif_thunks_ip_f.cpp compiles these seven pass bodies.
extern template void dif_tape_step_body<float, 29, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 43, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 59, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 71, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 73, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 83, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);
extern template void dif_tape_step_body<float, 97, true>(const float*, const float*,
    float*, float*, const dif_step<float>&, const dif_rt<float>&);

template auto dif_thunk<float>::chiplet(std::size_t) -> fn_t;

}
}
