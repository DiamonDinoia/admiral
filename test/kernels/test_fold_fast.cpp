// The one test TU built with the engine's fast-math flags (admiral_fast_math_flags): on the
// narrow long double cells the engine TUs run sincos_turns_fold_dbl under fast math, where a
// plain Kahan sum or the explicit std::fma of dd_two_prod collapses, and the fold lands
// several 2^-53 off. The volatile round trips pin each compensation step, and __FAST_MATH__
// routes the dd product through Dekker's split, so the fold holds its one 2^-53 contract
// under the flag. A revert of the pins must fail the bound below.
// The reference is the volatile-pinned dd series (about 2^-100), not libm: libm sinl/cosl is
// not portable under -ffinite-math-only, and reference.hpp's no-fast-math guard is about
// libm-referenced ratios. This TU does not include it. No lattice sweep at compile time:
// every call below runs at run time.
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/ct_math.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

using namespace admiral::detail;

// Same dd reference as test_ct_math.cpp's turns_ref_sincos: the dd ops are pinned at run
// time, so fast math cannot collapse the reference either.
static ct_sincos_v<ct_dd> turns_ref_sincos(bool conjugate, std::size_t num, std::size_t den) {
    const std::size_t n = num % den;
    const std::size_t m = conjugate && n != 0 ? den - n : n;
    ct_dd x =
        (ct_pi<ct_dd> * ct_dd{2.0 * static_cast<double>(m), 0.0}) / ct_dd{static_cast<double>(den), 0.0};
    if (x.hi > ct_pi<double>) x = x - (ct_pi<ct_dd> * ct_dd{2.0, 0.0});
    const ct_dd x2 = x * x;
    ct_dd ssum{0.0, 0.0}, csum{0.0, 0.0}, ts = x, tc{1.0, 0.0};
    for (unsigned i = 0; i < 60; ++i) {
        ssum = ssum + ts;
        csum = csum + tc;
        ts = ((ts * x2) * ct_dd{-1.0, 0.0}) / ct_dd{(2.0 * i + 2.0) * (2.0 * i + 3.0), 0.0};
        tc = ((tc * x2) * ct_dd{-1.0, 0.0}) / ct_dd{(2.0 * i + 1.0) * (2.0 * i + 2.0), 0.0};
    }
    return {ssum, csum};
}

static double turns_ulp_err_dbl(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<double> w = sincos_turns_fold_dbl<double>(conjugate, num, den);
    const ct_sincos_v<ct_dd> r = turns_ref_sincos(conjugate, num, den);
    const long double rs = static_cast<long double>(r.s.hi) + static_cast<long double>(r.s.lo);
    const long double rc = static_cast<long double>(r.c.hi) + static_cast<long double>(r.c.lo);
    const long double es = std::abs(static_cast<long double>(w.s) - rs);
    const long double ec = std::abs(static_cast<long double>(w.c) - rc);
    return static_cast<double>(std::max(es, ec) /
                               static_cast<long double>(std::numeric_limits<double>::epsilon()));
}

static const std::size_t turns_dens[] = {2, 8, 13, 47, 59, 61, 101, 251, 1024, 3329, 65537};

TEST_CASE("the fallback fold holds its bound under fast math", "[ct_math][numerics][fast_math]") {
    double worst = 0;
    for (const std::size_t den : turns_dens)
        for (std::size_t k = 0; k < den; ++k)
            for (const bool conj : {false, true})
                worst = std::max(worst, turns_ulp_err_dbl(conj, k, den));
    INFO("worst fallback fold error under fast math " << worst << " in 2^-53");
    REQUIRE(worst <= 1.0);
}
