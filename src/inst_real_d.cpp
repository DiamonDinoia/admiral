#include "admiral/detail/r2r.hpp"
#include "admiral/detail/real_fft.hpp"
#include "api_tier.hpp"
namespace admiral::detail {
template class real_adm_plan<double>;
template class nd_real_plan<double>;
template class r2r_plan<double>;
}

// Row-batched entries for bindings that cannot include engine detail headers
// (the numpy gufunc loops hold the prototypes): one plan built per call, one
// execute over all rows, no state kept between calls. POD arguments only.
// With ADM_DISPATCH the body runs on the engine tier that the dispatch picked
// (tier_slot); the entry symbol itself always binds the floor copy, so a direct
// body here would run the floor engine on every tier.
extern "C" void admiral_r2c_rows_f64(const double* in, std::complex<double>* out,
                                     std::size_t rows, std::size_t n) {
#ifdef ADM_DISPATCH
    admiral::detail::tier_slot<double>()->r2c_rows(in, out, rows, n);
#else
    admiral::detail::real_adm_plan<double>(n, admiral::effort::estimate)
        .r2c(in, out, rows, nullptr);
#endif
}

extern "C" void admiral_c2r_rows_f64(const std::complex<double>* in, double* out,
                                     std::size_t rows, std::size_t n) {
#ifdef ADM_DISPATCH
    admiral::detail::tier_slot<double>()->c2r_rows(in, out, rows, n);
#else
    admiral::detail::real_adm_plan<double>(n, admiral::effort::estimate)
        .c2r(in, out, rows, nullptr);
#endif
}
