#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <admiral/admiral.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <numbers>
#include <span>

using namespace Catch::Matchers;

// Relative error budget for an N-point transform in precision T (see test_fft.cpp).
template<typename T>
T fft_tol(std::size_t N, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(static_cast<T>(N)) + T(1);
    return eps * std::sqrt(static_cast<T>(N)) * log2N * T(64) * scale;
}

template<typename T>
T max_magnitude(const std::vector<std::complex<T>>& v) {
    T m = T(1);
    for (const auto& x : v) m = std::max(m, std::abs(x));
    return m;
}

// Helper to check if two complex vectors are approximately equal
template<typename T>
bool vectors_approx_equal(const std::vector<std::complex<T>>& a,
                          const std::vector<std::complex<T>>& b,
                          T tolerance = T(1e-10)) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

TEST_CASE("Plan creation and basic properties", "[plan]") {
    SECTION("Forward plan") {
        auto plan = admiral::plan<double>(128);
        REQUIRE(plan.size() == 128);
    }

    SECTION("Inverse plan") {
        auto plan = admiral::plan<double>(128);
        REQUIRE(plan.size() == 128);
    }

    SECTION("Both plan") {
        auto plan = admiral::plan<double>(128);
        REQUIRE(plan.size() == 128);
    }

    SECTION("Various sizes") {
        // Power of 2
        auto plan1 = admiral::plan<float>(64);
        REQUIRE(plan1.size() == 64);

        // Prime
        auto plan2 = admiral::plan<float>(13);
        REQUIRE(plan2.size() == 13);

        // Composite
        auto plan3 = admiral::plan<float>(24);
        REQUIRE(plan3.size() == 24);
    }
}

TEST_CASE("Plan move semantics", "[plan]") {
    // Create a plan
    auto plan1 = admiral::plan<double>(64);
    REQUIRE(plan1.size() == 64);

    // Move construct
    auto plan2 = std::move(plan1);
    REQUIRE(plan2.size() == 64);

    // Move assign
    auto plan3 = admiral::plan<double>(32);
    REQUIRE(plan3.size() == 32);
    plan3 = std::move(plan2);
    REQUIRE(plan3.size() == 64);
}

TEST_CASE("Plan execution produces correct results", "[plan]") {
    const std::size_t N = 16;
    const double tolerance = 1e-10;

    // Create test data
    std::vector<std::complex<double>> input(N);
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = std::complex<double>(std::sin(2.0 * std::numbers::pi * double(i) / double(N)),
                                        std::cos(2.0 * std::numbers::pi * double(i) / double(N)));
    }

    // Create plans
    auto fwd_plan = admiral::plan<double>(N);
    auto inv_plan = admiral::plan<double>(N);

    // Execute plan-based transform
    std::vector<std::complex<double>> plan_data = input;
    fwd_plan.forward(std::span(plan_data));

    std::vector<std::complex<double>> plan_recovered(N);
    std::copy(plan_data.begin(), plan_data.end(), plan_recovered.begin());
    inv_plan.inverse(std::span(plan_recovered));

    // Execute non-plan transform for comparison
    std::vector<std::complex<double>> direct_data(N);
    std::vector<std::complex<double>> input_copy = input;
    admiral::forward(std::span<const std::complex<double>>(input_copy), std::span(direct_data));

    std::vector<std::complex<double>> direct_recovered(N);
    admiral::inverse(std::span<const std::complex<double>>(direct_data), std::span(direct_recovered));

    // Plan and direct should give same results
    REQUIRE(vectors_approx_equal(plan_data, direct_data, tolerance));
    REQUIRE(vectors_approx_equal(plan_recovered, direct_recovered, tolerance));

    // Should recover original
    REQUIRE(vectors_approx_equal(input, plan_recovered, tolerance));
}

TEMPLATE_TEST_CASE("Plan round-trip (power-of-2 sizes)", "[plan][roundtrip]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::sin(T(2) * std::numbers::pi_v<T> * T(i) / T(N)),
                                       std::cos(T(2) * std::numbers::pi_v<T> * T(i) / T(N)));
        }

        // Create plans
        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        // Forward then inverse should recover input
        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(std::span(data));
        inv_plan.inverse(std::span(data));

        REQUIRE(vectors_approx_equal(input, data, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 4") { test_roundtrip(4); }
    SECTION("Size 8") { test_roundtrip(8); }
    SECTION("Size 16") { test_roundtrip(16); }
    SECTION("Size 32") { test_roundtrip(32); }
    SECTION("Size 64") { test_roundtrip(64); }
    SECTION("Size 128") { test_roundtrip(128); }
    SECTION("Size 256") { test_roundtrip(256); }
}

TEMPLATE_TEST_CASE("Plan round-trip (prime sizes)", "[plan][roundtrip][prime]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(T(i + 1), T(N - i));
        }

        // Create plans
        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        // Forward then inverse should recover input
        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(std::span(data));
        inv_plan.inverse(std::span(data));

        REQUIRE(vectors_approx_equal(input, data, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 3") { test_roundtrip(3); }
    SECTION("Size 5") { test_roundtrip(5); }
    SECTION("Size 7") { test_roundtrip(7); }
    SECTION("Size 11") { test_roundtrip(11); }
    SECTION("Size 13") { test_roundtrip(13); }
    SECTION("Size 17") { test_roundtrip(17); }
    SECTION("Size 251 (Bluestein)") { test_roundtrip(251); }
}

TEMPLATE_TEST_CASE("Plan round-trip (composite sizes)", "[plan][roundtrip][composite]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::cos(T(i)), std::sin(T(i) * T(0.5)));
        }

        // Create plans
        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        // Forward then inverse should recover input
        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(std::span(data));
        inv_plan.inverse(std::span(data));

        REQUIRE(vectors_approx_equal(input, data, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 6") { test_roundtrip(6); }
    SECTION("Size 10") { test_roundtrip(10); }
    SECTION("Size 12") { test_roundtrip(12); }
    SECTION("Size 15") { test_roundtrip(15); }
    SECTION("Size 20") { test_roundtrip(20); }
    SECTION("Size 24") { test_roundtrip(24); }
    SECTION("Size 121 (11^2)") { test_roundtrip(121); }
}

TEST_CASE("Plan reuse with different data", "[plan][reuse]") {
    const std::size_t N = 32;
    const double tolerance = 1e-10;

    // Create plan once for both directions
    auto plan = admiral::plan<double>(N);

    // Use plan multiple times with different data
    for (int batch = 0; batch < 10; ++batch) {
        std::vector<std::complex<double>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<double>(
                std::sin(double(static_cast<std::size_t>(batch) + i) * 0.1),
                std::cos(double(static_cast<std::size_t>(batch) + i) * 0.2)
            );
        }

        std::vector<std::complex<double>> data = input;
        plan.forward(std::span(data));
        plan.inverse(std::span(data));

        REQUIRE(vectors_approx_equal(input, data, tolerance));
    }
}

TEST_CASE("Plan error handling", "[plan][error]") {
    auto fwd_plan = admiral::plan<double>(32);

    SECTION("Wrong size data") {
        std::vector<std::complex<double>> data(16);  // Wrong size
        REQUIRE_THROWS_AS(fwd_plan.forward(std::span(data)), std::invalid_argument);
    }

    SECTION("Zero size plan") {
        REQUIRE_THROWS_AS(
            admiral::plan<double>(0),
            std::invalid_argument
        );
    }
}

// ============================================================================
// Codelet catalog routing correctness via the public API.
//
// Verifies that admiral::forward / admiral::inverse (and the plan path) route catalog
// sizes through the compile-time codelet correctly. Checks against a direct
// O(N^2) reference DFT for all 35 catalog sizes and four non-catalog sizes.
// Both float and double are exercised.
// ============================================================================

namespace {

// Direct O(N^2) DFT reference. Forward: exp(-2*pi*i*k*n/N). Unnormalized.
template<typename T>
std::vector<std::complex<T>> reference_dft_catalog(const std::vector<std::complex<T>>& x,
                                                    bool forward) {
    const std::size_t N = x.size();
    const T sign = forward ? T(-1) : T(1);
    std::vector<std::complex<T>> out(N);
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<T> acc(0, 0);
        for (std::size_t n = 0; n < N; ++n) {
            const T ang = sign * T(2) * std::numbers::pi_v<T> * T(k) * T(n) / T(N);
            acc += x[n] * std::complex<T>(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

// Catalog sizes: compare forward output against O(N^2) reference DFT, then
// verify round-trip. Accurate because N<=64 keeps reference error negligible.
template<typename T>
void check_catalog_size(std::size_t N) {
    // Deterministic non-trivial input matching test_fft_codelet.cpp convention.
    std::vector<std::complex<T>> input(N);
    for (std::size_t n = 0; n < N; ++n) {
        input[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                                   std::cos(T(1.1) * T(n) - T(0.2)));
    }

    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(T(N) + T(1));  // +1 avoids log2(1)=0 for N=1
    const T tol = eps * std::sqrt(T(N)) * (log2N + T(1)) * T(64);

    // --- Forward via admiral::forward free function ---
    std::vector<std::complex<T>> out(N);
    admiral::forward(std::span<const std::complex<T>>(input), std::span(out));

    const auto ref_fwd = reference_dft_catalog<T>(input, true);
    for (std::size_t k = 0; k < N; ++k) {
        const T abs_ref_re = std::abs(ref_fwd[k].real());
        const T abs_ref_im = std::abs(ref_fwd[k].imag());
        REQUIRE_THAT(out[k].real(),
            WithinAbs(ref_fwd[k].real(), tol * (T(1) + abs_ref_re)));
        REQUIRE_THAT(out[k].imag(),
            WithinAbs(ref_fwd[k].imag(), tol * (T(1) + abs_ref_im)));
    }

    // --- Inverse: apply to forward output, expect scaled input back ---
    std::vector<std::complex<T>> rt(N);
    admiral::inverse(std::span<const std::complex<T>>(out), std::span(rt));

    for (std::size_t n = 0; n < N; ++n) {
        const T abs_in_re = std::abs(input[n].real());
        const T abs_in_im = std::abs(input[n].imag());
        REQUIRE_THAT(rt[n].real(),
            WithinAbs(input[n].real(), tol * (T(1) + abs_in_re)));
        REQUIRE_THAT(rt[n].imag(),
            WithinAbs(input[n].imag(), tol * (T(1) + abs_in_im)));
    }
}

// Non-catalog sizes: the O(N^2) direct DFT is a poor oracle at large N due to
// accumulation error. Use a round-trip (forward then inverse) check instead —
// a valid end-to-end correctness test independent of any external reference.
template<typename T>
void check_noncatalog_size_roundtrip(std::size_t N) {
    std::vector<std::complex<T>> input(N);
    for (std::size_t n = 0; n < N; ++n) {
        input[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                                   std::cos(T(1.1) * T(n) - T(0.2)));
    }

    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(T(N) + T(1));  // +1 avoids log2(1)=0 for N=1
    const T tol = eps * std::sqrt(T(N)) * (log2N + T(1)) * T(64);

    std::vector<std::complex<T>> out(N);
    admiral::forward(std::span<const std::complex<T>>(input), std::span(out));

    std::vector<std::complex<T>> rt(N);
    admiral::inverse(std::span<const std::complex<T>>(out), std::span(rt));

    for (std::size_t n = 0; n < N; ++n) {
        const T abs_in_re = std::abs(input[n].real());
        const T abs_in_im = std::abs(input[n].imag());
        REQUIRE_THAT(rt[n].real(),
            WithinAbs(input[n].real(), tol * (T(1) + abs_in_re)));
        REQUIRE_THAT(rt[n].imag(),
            WithinAbs(input[n].imag(), tol * (T(1) + abs_in_im)));
    }
}

} // namespace

TEST_CASE("Codelet catalog routing — public API correctness (double)", "[codelet][catalog][plan]") {
    // The catalog is now EVERY integer in [2,64] (incl. primes 11..61 and
    // composites with a factor >7 like 22,26,33,55,62). Reference DFT + round-trip
    // for all of them — the O(N^2) oracle is accurate for N<=64.
    for (std::size_t N = 2; N <= 64; ++N) {
        CAPTURE(N);
        check_catalog_size<double>(N);
    }
    // Non-catalog sizes: round-trip only (O(N^2) reference is inaccurate at large N).
    // 100/128/720 are pow2/7-smooth routed through the iterative DIF pass-chain;
    // 121=11*11 (non-7-smooth) and primes 127, 251 take the Bluestein path.
    for (std::size_t N : {100u, 121u, 128u, 127u, 251u, 720u}) {
        CAPTURE(N);
        check_noncatalog_size_roundtrip<double>(N);
    }
}

TEST_CASE("Codelet catalog routing — public API correctness (float)", "[codelet][catalog][plan]") {
    // Every catalog size in [2,64]: reference DFT + round-trip.
    for (std::size_t N = 2; N <= 64; ++N) {
        CAPTURE(N);
        check_catalog_size<float>(N);
    }
    // Non-catalog sizes: round-trip only (float accumulation error at large N makes
    // reference DFT comparison unreliable). 100/121/128/720 iterative DIF; 127/251 Bluestein.
    for (std::size_t N : {100u, 121u, 128u, 127u, 251u, 720u}) {
        CAPTURE(N);
        check_noncatalog_size_roundtrip<float>(N);
    }
}

// Multi-pass pow2 / 7-smooth path: verify forward output against the O(N^2)
// reference DFT (not just round-trip — that can't catch a matching fwd/inv sign
// error). Sizes that factor into two catalog factors and route through the
// iterative DIF pass-chain: 100=10*10, 128=8*16, 144=12*12, 256=16*16. O(N^2)
// reference stays accurate enough for double at these N.
TEST_CASE("Multi-pass path — forward vs reference DFT (double)", "[multipass][plan]") {
    for (std::size_t N : {100u, 128u, 144u, 256u}) {
        CAPTURE(N);
        check_catalog_size<double>(N);
    }
}

TEMPLATE_TEST_CASE("Plan precision (forward+inverse)", "[plan][precision]", float, double) {
    using T = TestType;
    const std::size_t N = 64;

    std::vector<std::complex<T>> input(N);
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.1)));
    }

    auto plan = admiral::plan<T>(N);

    std::vector<std::complex<T>> data = input;
    plan.forward(std::span(data));
    plan.inverse(std::span(data));

    REQUIRE(vectors_approx_equal(input, data, fft_tol<T>(N, max_magnitude(input))));
}

// Out-of-place execute: dst gets the same result as the in-place path and src
// is preserved. Every route is copy-free OOP; sizes cover each route family:
// degenerate (1), codelet (31), good_thomas masked-tail (15, 30),
// good_thomas/dif (60), catalog-extra codelet (120), rader (127),
// four_step[_batched] (256), bluestein (10007), iterative_dif (4096).
TEMPLATE_TEST_CASE("Plan out-of-place execute (src preserved, matches in-place)",
                   "[plan][oop]", float, double) {
    using T = TestType;
    for (const std::size_t N : {std::size_t{1}, std::size_t{15}, std::size_t{30},
                                std::size_t{31}, std::size_t{60}, std::size_t{120},
                                std::size_t{127}, std::size_t{256},
                                std::size_t{10007}, std::size_t{4096}}) {
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i)
            input[i] = std::complex<T>(std::sin(T(i) * T(0.1)), std::cos(T(i) * T(0.13)));

        auto plan = admiral::plan<T>(N);
        const T tol = fft_tol<T>(N, max_magnitude(input));

        std::vector<std::complex<T>> inplace = input;
        plan.forward(std::span(inplace));

        const std::vector<std::complex<T>> src = input;
        std::vector<std::complex<T>> dst(N);
        plan.forward(src.data(), dst.data());

        REQUIRE(src == input);  // exact: src must be untouched
        REQUIRE(vectors_approx_equal(inplace, dst, tol));

        // Inverse OOP round-trips back to the input.
        std::vector<std::complex<T>> back(N);
        plan.inverse(dst.data(), back.data());
        REQUIRE(vectors_approx_equal(input, back, tol));
    }
}
