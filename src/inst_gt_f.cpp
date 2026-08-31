// Good-Thomas PFA kernel tree, float. The catalog expands 9 tuples times two
// aliasing arms per direction, so the tree gets its own TU instead of riding
// in `inst_plan_*`.
#include "admiral/detail/good_thomas.hpp"
namespace admiral::detail {
template void good_thomas_run<float, true>(const std::complex<float>*, std::complex<float>*, std::size_t) noexcept;
template void good_thomas_run<float, false>(const std::complex<float>*, std::complex<float>*, std::size_t) noexcept;
}
