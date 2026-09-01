#include "admiral/detail/r2r.hpp"
#include "admiral/detail/real_fft.hpp"
namespace admiral::detail {
template class real_adm_plan<double>;
template class nd_real_plan<double>;
template class r2r_plan<double>;
}
