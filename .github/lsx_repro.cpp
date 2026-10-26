// Minimal repro: xsimd::transpose on batch<double, lsx> at -O3 -mlsx -Wall -Werror.
// gcc 14 reports -Wmaybe-uninitialized on scratch_buffer in xsimd_lsx.hpp store_aligned.
#include <xsimd/xsimd.hpp>

using B = xsimd::batch<double, xsimd::lsx>;

void f(const double* p, double* q) {
    B r[2] = {B::load_unaligned(p), B::load_unaligned(p + 2)};
    xsimd::transpose(r, r + 2);
    r[0].store_unaligned(q);
    r[1].store_unaligned(q + 2);
}
