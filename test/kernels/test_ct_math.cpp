// ct_math.hpp holds the compile-time number-theory helpers behind kernel generation.
//
// Two layers on purpose:
//   * static_assert is the actual contract. These run in the constant evaluator,
//     which is how the generators call them, and a violation is a build failure
//     rather than a red test.
//   * the TEST_CASE calls the same helpers through volatile arguments, which forces
//     the RUNTIME instantiation no generator emits. That is a coverage path, not a
//     second assertion of the same fact, so it only spot-checks.

#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/math.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using namespace admiral::detail;

// --- smallest_radix / codelet_radix peel the expected factor ---
static_assert(smallest_radix(4) == 4);
static_assert(smallest_radix(8) == 4);    // 8 % 4 == 0
static_assert(smallest_radix(6) == 2);    // even, not a multiple of 4
static_assert(smallest_radix(9) == 3);
static_assert(smallest_radix(25) == 5);
static_assert(smallest_radix(49) == 7);
static_assert(smallest_radix(121) == 11);
static_assert(smallest_radix(13) == 13);  // prime to our radix set
static_assert(codelet_radix(32) == 8);    // special-cased radix-8 peel
static_assert(codelet_radix(60) == 5);    // special-cased
static_assert(codelet_radix(16) == 4);

// --- primality / modular arithmetic ---
static_assert(!ct_is_prime(0));
static_assert(!ct_is_prime(1));
static_assert(ct_is_prime(2));
static_assert(ct_is_prime(97));
static_assert(!ct_is_prime(91));          // 7*13
static_assert(ct_is_prime(65537));        // Fermat prime
static_assert(ct_powmod(2, 10, 1000) == 24);   // 1024 mod 1000
static_assert(ct_powmod(3, 0, 7) == 1);        // anything^0
static_assert(ct_powmod(7, 4, 5) == 1);        // 2401 mod 5

// --- Rader gate: up to 13 is a radix butterfly, 17 is the first Rader prime ---
static_assert(!is_rader_prime(11));
static_assert(!is_rader_prime(13));
static_assert(is_rader_prime(17));
static_assert(is_rader_prime(19));

static_assert(is_rader_prime(97));
static_assert(!is_rader_prime(100));      // composite

// g is a primitive root iff its powers hit every residue in [1,p) exactly once.
// The constant evaluator checks it for the primes Rader plans.
constexpr bool generates_group(std::size_t p) {
    const std::size_t g = ct_primitive_root(p);
    if (g < 2) return false;
    std::size_t x = 1;
    for (std::size_t i = 1; i < p; ++i) {
        x = x * g % p;
        if (x == 1 && i + 1 != p) return false;   // order < p-1
    }
    return x == 1;                                 // g^(p-1) == 1
}
static_assert(generates_group(13));
static_assert(generates_group(17));
static_assert(generates_group(97));
static_assert(generates_group(101));

// --- the twiddle fold is accurate at whatever width F carries ---
//
// ct_sincos_turns is the ONLY twiddle source for the butterflies, so its error
// is a phase error the whole transform inherits. Nothing else can catch a fold
// that is short: every transform test compares against a reference folded the
// same way, or against a tolerance scaled to the element type rather than to
// the fold. libm is the independent reference here.
//
// F = long double is the case that bites. It is 64 bits on x86, 113 on aarch64
// and 53 on arm64 macOS, so a fold calibrated to one of those is wrong on the
// others, and only this test says so.
template<typename F, std::size_t Den>
ADM_CONSTEVAL std::array<ct_sincos_v<F>, Den> folded_turns() {
    std::array<ct_sincos_v<F>, Den> a{};
    for (std::size_t k = 0; k < Den; ++k) a[k] = ct_sincos_turns<F>(/*conjugate=*/false, k, Den);
    return a;
}

// Worst |folded - libm| over one den, in eps(F). Den is a template parameter
// because ct_sincos_turns is consteval: the table has to fold at compile time.
template<typename F, std::size_t Den>
F turns_err() {
    constexpr std::array<ct_sincos_v<F>, Den> tab = folded_turns<F, Den>();
    const F eps = std::numeric_limits<F>::epsilon();
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

// den 7 is the sharp one: the octant residual reaches (pi/4)*6/7, where a short
// series has the least room. The catalog reaches every den below.
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

    // The SIMD engines fold at double; only the scalar long double backend asks
    // for more, and it is the width that varies across the CI hosts.
    const double wl = static_cast<double>(worst_turns_err<long double>());
    INFO("long double fold worst " << wl << " eps (digits "
                                   << std::numeric_limits<long double>::digits << ")");
    REQUIRE(wl <= 8);
}

// The helpers are constexpr, not consteval, so nothing above emits a runtime
// definition. Call them through volatile arguments to cover that instantiation.
TEST_CASE("ct_math helpers evaluate at runtime too", "[coverage][ct_math]") {
    auto rt = [](std::size_t n) { volatile std::size_t v = n; return smallest_radix(v); };
    REQUIRE(rt(8) == 4);
    REQUIRE(rt(121) == 11);

    auto cr = [](std::size_t n) { volatile std::size_t v = n; return codelet_radix(v); };
    REQUIRE(cr(32) == 8);
    REQUIRE(cr(60) == 5);
    REQUIRE(cr(16) == 4);   // neither special case: the smallest_radix fall-through

    auto ip = [](std::size_t n) { volatile std::size_t v = n; return ct_is_prime(v); };
    REQUIRE(ip(65537));
    REQUIRE_FALSE(ip(91));

    auto pm = [](std::size_t b, std::size_t e, std::size_t m) {
        volatile std::size_t vb = b, ve = e, vm = m; return ct_powmod(vb, ve, vm);
    };
    REQUIRE(pm(2, 10, 1000) == 24);

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
