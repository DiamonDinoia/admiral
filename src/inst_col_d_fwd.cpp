#include "inst_col_radices.hpp"

namespace admiral {
namespace detail {

#define ADM_COL_EXTERN(R)                                                                          \
    extern template ADM_COL_PASS(double, R);                                                       \
    extern template ADM_COL_PASS_FIRST(double, true, R);
ADM_COL_RADICES(ADM_COL_EXTERN)
#define ADM_COL_EXTERN_STAGED(R) extern template ADM_COL_PASS_FIRST_STAGED(double, true, R);
ADM_COL_STAGED_RADICES(ADM_COL_EXTERN_STAGED)

template void col_dif_execute_ws<double, true>(
    std::complex<double>*, std::size_t, std::size_t, std::size_t, double*, double*, double*, double*,
    const dif_twiddle_set<double>&, double, const std::complex<double>*, std::size_t);

}
}
