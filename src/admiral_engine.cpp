// Single instantiation site for the heavy complex route engines (plan_impl and
// nd_runtime_plan; nd_real_plan stays inline). Every other TU sees the extern-template
// declarations in the detail headers and references these symbols instead of
// re-instantiating the whole route tree — measured ~3.1 GiB -> ~0.3 GiB peak
// RSS per consumer TU. This is the one TU that pays the full instantiation cost,
// exactly once. Built as an OBJECT library so its (weak) explicit-instantiation
// symbols are always linked into consumers rather than lazily pulled from an
// archive (a weak definition never forces archive-member extraction).
#define ADM_INSTANTIATE_ENGINE
#include "admiral/admiral.hpp"

namespace admiral::detail {
template class plan_impl<float>;
template class plan_impl<double>;
template class nd_runtime_plan<float>;
template class nd_runtime_plan<double>;
// nd_real_plan (r2c/c2r) is intentionally NOT externed — see the note in
// real_fft.hpp: out-of-lining it broke serial-vs-threaded bit-identity under
// -ffast-math, and it lives in few TUs, so it stays inline.
}  // namespace admiral::detail
