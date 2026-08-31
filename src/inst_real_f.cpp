// See `inst_nd_f.cpp`.
#include "admiral/detail/r2r.hpp"
#include "admiral/detail/real_fft.hpp"
namespace admiral::detail {
template class real_adm_plan<float>;
template class nd_real_plan<float>;
template class r2r_plan<float>;
}
