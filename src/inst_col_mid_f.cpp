#include "inst_col_radices.hpp"

namespace admiral {
namespace detail {

#define ADM_COL_INST(R) template ADM_COL_PASS(float, R);
ADM_COL_RADICES(ADM_COL_INST)

}
}
