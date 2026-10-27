#include "inst_col_radices.hpp"

namespace admiral {
namespace detail {

#define ADM_COL_EXTERN(R)                                                                          \
    extern template ADM_COL_PASS(float, R);                                                        \
    extern template ADM_COL_PASS_FIRST(float, true, R);
ADM_COL_RADICES(ADM_COL_EXTERN)
#define ADM_COL_EXTERN_STAGED(R) extern template ADM_COL_PASS_FIRST_STAGED(float, true, R);
ADM_COL_STAGED_RADICES(ADM_COL_EXTERN_STAGED)

template void col_dif_execute_ws<float, true>(
    std::complex<float>*, std::size_t, std::size_t, std::size_t, float*, float*, float*, float*,
    const dif_twiddle_set<float>&, float, const std::complex<float>*, std::size_t);

}
}
