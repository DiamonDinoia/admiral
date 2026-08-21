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

#include <cstddef>
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
consteval bool generates_group(std::size_t p) {
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
