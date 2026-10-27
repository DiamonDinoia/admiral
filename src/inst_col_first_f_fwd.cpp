#include "inst_col_radices.hpp"

namespace admiral {
namespace detail {

#define ADM_COL_INST(R) template ADM_COL_PASS_FIRST(float, true, R);
ADM_COL_RADICES(ADM_COL_INST)
#define ADM_COL_INST_STAGED(R) template ADM_COL_PASS_FIRST_STAGED(float, true, R);
ADM_COL_STAGED_RADICES(ADM_COL_INST_STAGED)

}
}
