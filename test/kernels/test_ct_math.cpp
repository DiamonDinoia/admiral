
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

template<typename F, std::size_t Den>
F turns_err() {
    constexpr std::array<ct_sincos_v<F>, Den> tab = folded_turns<F, Den>();
    const F eps = static_cast<F>(digits_eps<F>());
    F worst = 0;
    for (std::size_t k = 0; k < Den; ++k) {
        const F ang = F(2) * numbers::pi_v<F> * static_cast<F>(k) / static_cast<F>(Den);
        const F es = std::abs(tab[k].s - std::sin(ang)) / eps;
        const F ec = std::abs(tab[k].c - std::cos(ang)) / eps;
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

// The double fold is a rounded ct_dd value, so it lands within 1 double-ulp of the correctly
// rounded root. A double-folded series plus the octant fold loses a factor of 2-3 here, and
// that error sits in every twiddle table and codelet constant.
static double turns_ulp_err(bool conjugate, std::size_t num, std::size_t den) {
    const ct_sincos_v<double> w = sincos_turns_eval<double>(conjugate, num, den);
    const std::size_t n = num % den;
    const long double ang = 2.0L * numbers::pi_v<long double>
                            * static_cast<long double>(conjugate && n != 0 ? den - n : n)
                            / static_cast<long double>(den);
    const double es = std::abs(w.s - static_cast<double>(std::sin(ang)));
    const double ec = std::abs(w.c - static_cast<double>(std::cos(ang)));
    return std::max(es, ec) / digits_eps<double>();
}

TEST_CASE("trig constants are correctly rounded doubles", "[ct_math][numerics]") {
    double worst = 0;
    for (const std::size_t den : {std::size_t{13}, std::size_t{47}, std::size_t{59},
                                  std::size_t{61}, std::size_t{101}, std::size_t{251},
                                  std::size_t{1024}, std::size_t{3329}}) {
        for (std::size_t k = 0; k < den; ++k) {
            worst = std::max(worst, turns_ulp_err(false, k, den));
            worst = std::max(worst, turns_ulp_err(true, k, den));
            const auto [sn, cs] = admiral::detail::portable_trig::sincos_turns(k, den, true);
            const std::size_t n = k % den;
            const long double ang = -2.0L * numbers::pi_v<long double>
                                    * static_cast<long double>(n)
                                    / static_cast<long double>(den);
            worst = std::max(worst, std::abs(sn - static_cast<double>(std::sin(ang)))
                                        / digits_eps<double>());
            worst = std::max(worst, std::abs(cs - static_cast<double>(std::cos(ang)))
                                        / digits_eps<double>());
        }
    }
    INFO("worst trig constant error " << worst << " ulp");
    REQUIRE(worst <= 1.0);
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
