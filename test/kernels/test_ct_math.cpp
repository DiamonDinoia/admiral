
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/math.hpp>
#include <admiral/detail/portable_trig.hpp>

#include "utils/reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using namespace admiral::detail;

static_assert(smallest_radix(4) == 4);
static_assert(smallest_radix(8) == 4);
static_assert(smallest_radix(6) == 2);
static_assert(smallest_radix(9) == 3);
static_assert(smallest_radix(25) == 5);
static_assert(smallest_radix(49) == 7);
static_assert(smallest_radix(121) == 11);
static_assert(smallest_radix(13) == 13);
static_assert(codelet_radix(32) == 8);
static_assert(codelet_radix(60) == 5);
static_assert(codelet_radix(16) == 4);

static_assert(!ct_is_prime(0));
static_assert(!ct_is_prime(1));
static_assert(ct_is_prime(2));
static_assert(ct_is_prime(97));
static_assert(!ct_is_prime(91));
static_assert(ct_is_prime(65537));
static_assert(ct_powmod(2, 10, 1000) == 24);
static_assert(ct_powmod(3, 0, 7) == 1);
static_assert(ct_powmod(7, 4, 5) == 1);
// The residues reach m - 1 = 65536 and 65536^2 = 2^32 wraps a 32-bit size_t to 0.
static_assert(ct_powmod(65536, 2, 65537) == 1);
static_assert(ct_powmod(65536, 65535, 65537) == 65536);

static_assert(!is_rader_prime(11));
static_assert(!is_rader_prime(13));
static_assert(is_rader_prime(17));
static_assert(is_rader_prime(19));

static_assert(is_rader_prime(97));
static_assert(!is_rader_prime(100));

constexpr bool generates_group(std::size_t p) {
    const std::size_t g = ct_primitive_root(p);
    if (g < 2) return false;
    std::size_t x = 1;
    for (std::size_t i = 1; i < p; ++i) {
        x = x * g % p;
        if (x == 1 && i + 1 != p) return false;
    }
    return x == 1;
}
static_assert(generates_group(13));
static_assert(generates_group(17));
static_assert(generates_group(97));
static_assert(generates_group(101));

template<typename F, std::size_t Den>
ADM_CONSTEVAL std::array<ct_sincos_v<F>, Den> folded_turns() {
    std::array<ct_sincos_v<F>, Den> a{};
    for (std::size_t k = 0; k < Den; ++k) a[k] = ct_sincos_turns<F>(false, k, Den);
    return a;
}

// Reference sine and cosine of exp(+2*pi*i*n/den) at double-double precision: a Taylor
// series on the dd angle. The libm long double route is not portable here: MSVC and Apple
// arm64 have a double long double, and clang -mlong-double-64 narrows the type without the
// libm ABI, so sinl returns garbage there. The series is accurate to about 2^-100.
// ponytail: m and den must stay below 2^53 for the ratio to fold exactly; the test dens do.
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

template<typename F, std::size_t Den>
F turns_err() {
    constexpr std::array<ct_sincos_v<F>, Den> tab = folded_turns<F, Den>();
    const F eps = static_cast<F>(digits_eps<F>());
    F worst = 0;
    for (std::size_t k = 0; k < Den; ++k) {
        const ct_sincos_v<ct_dd> r = turns_ref_sincos(false, k, Den);
        const long double rs = static_cast<long double>(r.s.hi) + static_cast<long double>(r.s.lo);
        const long double rc = static_cast<long double>(r.c.hi) + static_cast<long double>(r.c.lo);
        const F es = static_cast<F>(std::abs(static_cast<long double>(tab[k].s) - rs)
                                    / static_cast<long double>(eps));
        const F ec = static_cast<F>(std::abs(static_cast<long double>(tab[k].c) - rc)
                                    / static_cast<long double>(eps));
        worst = es > worst ? es : worst;
        worst = ec > worst ? ec : worst;
    }
    return worst;
}

template<typename F>
F worst_turns_err() {
    F w = 0;
    for (const F e : {turns_err<F, 2>(), turns_err<F, 3>(), turns_err<F, 4>(),
                      turns_err<F, 5>(), turns_err<F, 7>(), turns_err<F, 8>(),
                      turns_err<F, 11>(), turns_err<F, 13>(), turns_err<F, 16>(),
                      turns_err<F, 32>(), turns_err<F, 64>()})
        w = e > w ? e : w;
    return w;
}

TEST_CASE("ct_sincos_turns matches libm at the fold precision", "[ct_math][numerics]") {
    const double wd = worst_turns_err<double>();
    INFO("double fold worst " << wd << " eps");
    REQUIRE(wd <= 8);
}

template<typename T>
static double turns_ulp_err_v(T s, T c, bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<ct_dd> r = turns_ref_sincos(conjugate, num, den);
    const long double rs = static_cast<long double>(r.s.hi) + static_cast<long double>(r.s.lo);
    const long double rc = static_cast<long double>(r.c.hi) + static_cast<long double>(r.c.lo);
    const long double es = std::abs(static_cast<long double>(s) - rs);
    const long double ec = std::abs(static_cast<long double>(c) - rc);
    return static_cast<double>(std::max(es, ec) / static_cast<long double>(digits_eps<T>()));
}

static double turns_ulp_err(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<double> w = sincos_turns_eval<double>(conjugate, num, den);
    return turns_ulp_err_v<double>(w.s, w.c, conjugate, num, den);
}

static double turns_ulp_err_dbl(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<double> w = sincos_turns_fold_dbl<double>(conjugate, num, den);
    return turns_ulp_err_v<double>(w.s, w.c, conjugate, num, den);
}

static double turns_ulp_err_ctdd(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<ct_dd> w = sincos_turns_fold<ct_dd>(conjugate, num, den);
    return turns_ulp_err_v<double>(dd_round<double>(w.s), dd_round<double>(w.c),
                                   conjugate, num, den);
}

static double turns_ulp_err_ctdd_f(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<ct_dd> w = sincos_turns_fold<ct_dd>(conjugate, num, den);
    return turns_ulp_err_v<float>(dd_round<float>(w.s), dd_round<float>(w.c),
                                  conjugate, num, den);
}

static double turns_ulp_err_f(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<float> w = sincos_turns_eval<float>(conjugate, num, den);
    return turns_ulp_err_v<float>(w.s, w.c, conjugate, num, den);
}

static double turns_ulp_err_f_dbl(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<float> w = sincos_turns_fold_dbl<float>(conjugate, num, den);
    return turns_ulp_err_v<float>(w.s, w.c, conjugate, num, den);
}

// Every shipped denominator class, including the power-of-two and prime edges the fold can
// meet: den 2 and 8 (octant degeneracies), 65537 (the widest FFT root of unity), and the
// largest den whose 8*num stays in size_t.
static const std::size_t turns_dens[] = {2, 8, 13, 47, 59, 61, 101, 251, 1024, 3329, 65537};

template<typename Err>
static double turns_worst(Err err) {
    double worst = 0;
    for (const std::size_t den : turns_dens) {
        for (std::size_t k = 0; k < den; ++k) {
            worst = std::max(worst, err(false, k, den));
            worst = std::max(worst, err(true, k, den));
        }
    }
    constexpr std::size_t big_den = SIZE_MAX / 8;
    for (const std::size_t k : {std::size_t{0}, std::size_t{1}, big_den / 8,
                                big_den / 2, big_den - 1}) {
        worst = std::max(worst, err(false, k, big_den));
        worst = std::max(worst, err(true, k, big_den));
    }
    return worst;
}

// The fold lands within one 2^-53 of the true value: one double ulp for a value near 1, the
// quantum the FFT error analysis carries. A double-folded series plus the octant fold loses a
// factor of 2-3 here, and that error sits in every twiddle table and codelet constant. The
// metric is absolute, in units of 2^-53. The dispatcher's contract is one 2^-53 on every
// route; the long double route measures about half of that.
TEST_CASE("trig constants land within one 2^-53", "[ct_math][numerics]") {
    const double worst = turns_worst(turns_ulp_err);
    INFO("worst trig constant error " << worst << " in 2^-53");
    REQUIRE(worst <= 1.0);
}

// The fallback fold runs where long double is double or composite. Its contract is the same
// absolute bound the FFT error analysis needs: within one 2^-53 of the true value. Relative to
// a value's own magnitude the error grows for small results (up to 1000 ulp near 1e-3), so
// the relative check applies only where the value is well scaled.
TEST_CASE("the fallback fold lands within one 2^-53", "[ct_math][numerics]") {
    const double worst = turns_worst(turns_ulp_err_dbl);
    INFO("worst fallback fold error " << worst << " in 2^-53");
    REQUIRE(worst <= 1.0);
    const ct_sincos_v<double> w = sincos_turns_fold_dbl<double>(true, 1450, 2313);
    REQUIRE(w.s == 0x1.6e50cf76e4352p-1);
    REQUIRE(w.c == -0x1.65b5e6319fdd2p-1);
    const ct_sincos_v<double> a = sincos_turns_fold_dbl<double>(false, 1, 3);
    const ct_sincos_v<double> b = sincos_turns_fold_dbl<double>(true, 1, 3);
    REQUIRE(a.s == 0x1.bb67ae8584caap-1);
    REQUIRE(a.c == -0x1p-1);
    REQUIRE(b.s == -0x1.bb67ae8584cabp-1);
    REQUIRE(b.c == -0x1p-1);
}

// The ct_dd fold for F = double and float is the long-double-is-double route's cousin and
// the IBM double-double route's body; it must run here too or it rots like its branch did.
TEST_CASE("the ct_dd fold lands within 0.75 of a 2^-53", "[ct_math][numerics]") {
    const double worst = turns_worst(turns_ulp_err_ctdd);
    const double worstf = turns_worst(turns_ulp_err_ctdd_f);
    INFO("worst ct_dd fold error " << worst << " double, " << worstf << " float");
    REQUIRE(worst <= 0.75);
    REQUIRE(worstf <= 0.75);
}

TEST_CASE("float trig constants land within 0.75 float ulp", "[ct_math][numerics]") {
    const double worst = turns_worst(turns_ulp_err_f);
    const double worstf = turns_worst(turns_ulp_err_f_dbl);
    INFO("worst float constant error " << worst << " ulp route, " << worstf << " ulp fallback");
    REQUIRE(worst <= 0.75);
    REQUIRE(worstf <= 0.75);
}

// 1 + 2^-24 is a float tie and the residual lo pushes past it: the float must round up to the
// next float, not down to the tie-even neighbor. lo stays inside half of hi's double ulp, the
// normalized dd_round input; the old test used lo = 2^-52 and passed vacuously.
TEST_CASE("dd_round rounds a narrow F once", "[ct_math][numerics]") {
    REQUIRE(dd_round<float>(ct_dd{0x1.000001p+0, 0x1p-54}) == 0x1.000002p+0f);
    REQUIRE(dd_round<float>(ct_dd{-0x1.000001p+0, -0x1p-54}) == -0x1.000002p+0f);
    REQUIRE(dd_round<float>(ct_dd{0x1.000001p+0, -0x1p-54}) == 0x1.000000p+0f);
    REQUIRE(dd_round<float>(ct_dd{0x1.8p+0, 0x1p-54}) == 0x1.8p+0f);
}

// A runtime table entry and a folded codelet constant carry the same value: portable_trig is
// the fold, not a second approximation of it.
TEST_CASE("a runtime table entry equals the folded constant", "[ct_math][numerics]") {
    for (const std::size_t den : turns_dens) {
        for (std::size_t k = 0; k < den; ++k) {
            for (const bool forward : {false, true}) {
                const ct_sincos_v<double> w = sincos_turns_eval<double>(forward, k, den);
                const auto p = admiral::detail::portable_trig::sincos_turns(k, den, forward);
                REQUIRE(p.first == w.s);
                REQUIRE(p.second == w.c);
            }
        }
    }
}

TEST_CASE("ct_sincos_turns matches libm at the long double fold precision",
          "[ct_math][numerics][longdouble]") {
    const double wl = static_cast<double>(worst_turns_err<long double>());
    INFO("long double fold worst " << wl << " eps (digits "
                                   << std::numeric_limits<long double>::digits << ")");
    REQUIRE(wl <= 8);
}

TEST_CASE("ct_math helpers evaluate at runtime too", "[coverage][ct_math]") {
    auto rt = [](std::size_t n) { volatile std::size_t v = n; return smallest_radix(v); };
    REQUIRE(rt(8) == 4);
    REQUIRE(rt(121) == 11);

    auto cr = [](std::size_t n) { volatile std::size_t v = n; return codelet_radix(v); };
    REQUIRE(cr(32) == 8);
    REQUIRE(cr(60) == 5);
    REQUIRE(cr(16) == 4);

    auto ip = [](std::size_t n) { volatile std::size_t v = n; return ct_is_prime(v); };
    REQUIRE(ip(65537));
    REQUIRE_FALSE(ip(91));

    auto pm = [](std::size_t b, std::size_t e, std::size_t m) {
        volatile std::size_t vb = b, ve = e, vm = m; return ct_powmod(vb, ve, vm);
    };
    REQUIRE(pm(2, 10, 1000) == 24);
    REQUIRE(pm(65536, 2, 65537) == 1);

    auto rp = [](std::size_t n) { volatile std::size_t v = n; return is_rader_prime(v); };
    REQUIRE(rp(17));
    REQUIRE_FALSE(rp(13));
    REQUIRE_FALSE(rp(11));

    auto pr = [](std::size_t p) { volatile std::size_t v = p; return ct_primitive_root(v); };
    const std::size_t p = 97, g = pr(p);
    std::vector<bool> seen(p, false);
    std::size_t x = 1;
    for (std::size_t i = 1; i < p; ++i) {
        x = x * g % p;
        REQUIRE_FALSE(seen[x]);
        seen[x] = true;
    }
}
