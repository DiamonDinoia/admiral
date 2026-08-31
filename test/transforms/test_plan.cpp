#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"
#include <admiral/admiral.hpp>
#include <admiral/detail/plan.hpp>  // `measure_batch`, `kMeasureSampleNs` (sample-interval check)
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <chrono>

using namespace Catch::Matchers;
using admiral::span;
namespace num = admiral::detail::numbers;

TEST_CASE("Plan creation and basic properties", "[plan]") {
    // One plan type: `admiral::plan<T>` is bidirectional. No forward-only plan exists.
    SECTION("Reports its size") { REQUIRE(admiral::plan<double>(128).size() == 128); }

    SECTION("Various sizes") {
        auto plan1 = admiral::plan<float>(64);
        REQUIRE(plan1.size() == 64);

        auto plan2 = admiral::plan<float>(13);
        REQUIRE(plan2.size() == 13);

        auto plan3 = admiral::plan<float>(24);
        REQUIRE(plan3.size() == 24);
    }
}

TEST_CASE("Plan move semantics", "[plan]") {
    // `size()` alone cannot tell a correct move from one that dropped a member.
    // Each moved-to plan transforms and matches a plan that was never moved.
    std::vector<std::complex<double>> in(64);
    for (std::size_t i = 0; i < 64; ++i) in[i] = {std::sin(double(i)), std::cos(double(i))};
    auto want = in;
    admiral::plan<double>(64).forward(span(want));

    auto plan1 = admiral::plan<double>(64);
    REQUIRE(plan1.size() == 64);

    auto plan2 = std::move(plan1);
    REQUIRE(plan2.size() == 64);
    auto got = in;
    plan2.forward(span(got));
    REQUIRE(got == want);

    auto plan3 = admiral::plan<double>(32);
    REQUIRE(plan3.size() == 32);
    plan3 = std::move(plan2);
    REQUIRE(plan3.size() == 64);
    auto got2 = in;
    plan3.forward(span(got2));
    REQUIRE(got2 == want);
}

TEST_CASE("Plan execution produces correct results", "[plan]") {
    const std::size_t N = 16;

    std::vector<std::complex<double>> input(N);
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = std::complex<double>(std::sin(2.0 * num::pi * double(i) / double(N)),
                                        std::cos(2.0 * num::pi * double(i) / double(N)));
    }

    auto fwd_plan = admiral::plan<double>(N);
    auto inv_plan = admiral::plan<double>(N);

    std::vector<std::complex<double>> plan_data = input;
    fwd_plan.forward(span(plan_data));

    std::vector<std::complex<double>> plan_recovered(N);
    std::copy(plan_data.begin(), plan_data.end(), plan_recovered.begin());
    inv_plan.inverse(span(plan_recovered));

    std::vector<std::complex<double>> direct_data(N);
    std::vector<std::complex<double>> input_copy = input;
    admiral::forward(span<const std::complex<double>>(input_copy), span(direct_data));

    std::vector<std::complex<double>> direct_recovered(N);
    admiral::inverse(span<const std::complex<double>>(direct_data), span(direct_recovered));

    require_close(plan_data, direct_data, fft_tol<double>());
    require_close(plan_recovered, direct_recovered, fft_tol<double>());

    require_close(input, plan_recovered, fft_tol<double>(2.0));  // round trip compounds
}

TEMPLATE_TEST_CASE("Plan round-trip (power-of-2 sizes)", "[plan][roundtrip]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::sin(T(2) * num::pi_v<T> * T(i) / T(N)),
                                       std::cos(T(2) * num::pi_v<T> * T(i) / T(N)));
        }

        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(span(data));
        inv_plan.inverse(span(data));

        require_close(input, data, fft_tol<T>());
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
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(T(i + 1), T(N - i));
        }

        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(span(data));
        inv_plan.inverse(span(data));

        require_close(input, data, fft_tol<T>());
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
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::cos(T(i)), std::sin(T(i) * T(0.5)));
        }

        auto fwd_plan = admiral::plan<T>(N);
        auto inv_plan = admiral::plan<T>(N);

        std::vector<std::complex<T>> data = input;
        fwd_plan.forward(span(data));
        inv_plan.inverse(span(data));

        require_close(input, data, fft_tol<T>());
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

    auto plan = admiral::plan<double>(N);

    for (int batch = 0; batch < 10; ++batch) {
        std::vector<std::complex<double>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<double>(
                std::sin(double(static_cast<std::size_t>(batch) + i) * 0.1),
                std::cos(double(static_cast<std::size_t>(batch) + i) * 0.2)
            );
        }

        std::vector<std::complex<double>> data = input;
        plan.forward(span(data));
        plan.inverse(span(data));

        require_close(input, data, fft_tol<double>(2.0));  // round trip compounds
    }
}

TEST_CASE("Plan error handling", "[plan][error]") {
    auto fwd_plan = admiral::plan<double>(32);

    SECTION("Wrong size data") {
        std::vector<std::complex<double>> data(16);
        REQUIRE_THROWS_AS(fwd_plan.forward(span(data)), admiral::size_error);
    }

    SECTION("Zero size plan") {
        REQUIRE_THROWS_AS(
            admiral::plan<double>(0),
            admiral::size_error
        );
    }
}

// The r2c span overloads exist for this check. The two buffers of the transform
// differ in length, so a caller can pass the right sizes in the wrong order. The
// pointer overload cannot tell. n=32 -> 32 reals, 17 complex.
TEST_CASE("plan_r2c span overloads validate both extents", "[plan][error]") {
    admiral::plan_r2c<double> r({32});
    REQUIRE(r.real_size() == 32);
    REQUIRE(r.cplx_size() == 17);

    std::vector<double> real(r.real_size()), back(r.real_size());
    std::vector<std::complex<double>> spec(r.cplx_size());
    for (std::size_t i = 0; i < real.size(); ++i) real[i] = std::sin(0.3 * double(i));

    // Against the pointer overload, not a round trip: the check asserts the
    // forwarding itself and stays true under any inverse scaling convention.
    SECTION("forwards identically to the pointer form") {
        std::vector<std::complex<double>> spec_ptr(r.cplx_size());
        r.forward(real.data(), spec_ptr.data());
        r.forward(span<const double>(real), span(spec));
        REQUIRE(spec == spec_ptr);

        std::vector<double> back_ptr(r.real_size());
        auto spec_copy = spec;  // `inverse` overwrites the spectrum argument
        r.inverse(spec_ptr.data(), back_ptr.data());
        r.inverse(span(spec_copy), span(back));
        REQUIRE(back == back_ptr);
    }

    SECTION("wrong extent throws on either buffer") {
        std::vector<std::complex<double>> spec_bad(r.cplx_size() + 1);
        std::vector<double> real_bad(r.real_size() + 1);
        REQUIRE_THROWS_AS(r.forward(span<const double>(real), span(spec_bad)),
                          admiral::size_error);
        REQUIRE_THROWS_AS(r.forward(span<const double>(real_bad), span(spec)),
                          admiral::size_error);
        REQUIRE_THROWS_AS(r.inverse(span(spec_bad), span(back)),
                          admiral::size_error);
        REQUIRE_THROWS_AS(r.inverse(span(spec), span(real_bad)),
                          admiral::size_error);
    }
}

// Codelet catalog routing via the public API: forward vs the O(N^2) reference
// plus a round trip, for every catalog size and a few non-catalog sizes.

namespace {

// `N` <= 64 keeps the O(N^2) reference's own error negligible; the reference stays
// a usable oracle.
template<typename T>
void check_catalog_size(std::size_t N) {
    const auto input = make_input<T>(N);

    std::vector<std::complex<T>> out(N);
    admiral::forward(span<const std::complex<T>>(input), span(out));
    require_close(out, reference_dft<T>(input, true), fft_tol<T>());

    std::vector<std::complex<T>> rt(N);
    admiral::inverse(span<const std::complex<T>>(out), span(rt));
    require_close(rt, input, fft_tol<T>(2.0));  // two transforms compound
}

// Past the catalog, the O(N^2) reference accumulates too much error for oracle use,
// so these sizes run only a round trip.
template<typename T>
void check_noncatalog_size_roundtrip(std::size_t N) {
    const auto input = make_input<T>(N);

    std::vector<std::complex<T>> out(N);
    admiral::forward(span<const std::complex<T>>(input), span(out));

    std::vector<std::complex<T>> rt(N);
    admiral::inverse(span<const std::complex<T>>(out), span(rt));
    require_close(rt, input, fft_tol<T>(2.0));  // two transforms compound
}

} // namespace

TEST_CASE("Codelet catalog routing: public API correctness (double)", "[codelet][catalog][plan]") {
    // The catalog is every integer in [2,64], including primes 11..61 and the
    // composites with a factor > 7 (22, 26, 33, 55, 62). The O(N^2) oracle is
    // accurate on these sizes.
    for (std::size_t N = 2; N <= 64; ++N) {
        CAPTURE(N);
        check_catalog_size<double>(N);
    }
    // Non-catalog sizes: round trip only here. Every size except 720 is <= 512, and
    // `test_sweeps.cpp` already compares those sizes against the long-double
    // O(N^2) DFT in both precisions. The set covers a catalog extra (100 = 4*25),
    // 11^2 (121), pow2 (128), 7-smooth (720) and the Rader primes (127, 251).
    for (std::size_t N : {100u, 121u, 128u, 127u, 251u, 720u}) {
        CAPTURE(N);
        check_noncatalog_size_roundtrip<double>(N);
    }
}

TEST_CASE("Codelet catalog routing: public API correctness (float)", "[codelet][catalog][plan]") {
    // Every catalog size in [2,64]: reference DFT + round-trip.
    for (std::size_t N = 2; N <= 64; ++N) {
        CAPTURE(N);
        check_catalog_size<float>(N);
    }
    // Non-catalog sizes: round trip only, because float accumulation error at large N
    // makes the reference DFT comparison unreliable. Same coverage set as the double
    // case above.
    for (std::size_t N : {100u, 121u, 128u, 127u, 251u, 720u}) {
        CAPTURE(N);
        check_noncatalog_size_roundtrip<float>(N);
    }
}

// Multi-pass pow2 / 7-smooth path: check forward output against the O(N^2)
// reference DFT, because a round trip cannot catch a matching fwd/inv sign error.
// These sizes factor into two catalog factors and route through the `iterative_dif`
// pass chain: 100=10*10, 128=8*16, 144=12*12, 256=16*16. The O(N^2) reference stays
// accurate enough for double at these N.
TEST_CASE("Multi-pass path: forward vs reference DFT (double)", "[multipass][plan]") {
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
    plan.forward(span(data));
    plan.inverse(span(data));

    require_close(input, data, fft_tol<T>());
}

// Out-of-place execute: `dst` matches the in-place result, `src` stays unchanged.
// Every route is copy-free out-of-place. The sizes cover each route family:
// degenerate (1), `codelet` (31), `good_thomas` masked-tail (15, 30),
// `good_thomas`/`iterative_dif` (60), catalog-extra `codelet` (120), `rader` (127),
// `four_step` or `four_step_batched` (256), `bluestein` (10007), `iterative_dif` (4096).
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
        const double tol = fft_tol<T>();

        std::vector<std::complex<T>> inplace = input;
        plan.forward(span(inplace));

        const std::vector<std::complex<T>> src = input;
        std::vector<std::complex<T>> dst(N);
        plan.forward(src.data(), dst.data());

        REQUIRE(src == input);  // exact: `src` must be untouched
        require_close(inplace, dst, tol);

        // Inverse out-of-place round-trips back to the input.
        std::vector<std::complex<T>> back(N);
        plan.inverse(dst.data(), back.data());
        require_close(input, back, tol);
    }
}

TEST_CASE("plan_r2c rejects an empty or overflowing shape", "[plan][error]") {
    // Rank 0: no innermost axis to transform.
    REQUIRE_THROWS_AS(admiral::plan_r2c<double>(span<const std::size_t>{}),
                      admiral::size_error);
    REQUIRE_THROWS_AS(admiral::plan_r2c<float>({0}), admiral::size_error);   // zero extent
    REQUIRE_THROWS_AS(admiral::plan_r2c<double>(
                          std::initializer_list<std::size_t>{std::size_t(-1) / 2, 4}),
                      admiral::size_error);   // overflows `size_t`
}

TEMPLATE_TEST_CASE("plan_r2c of a 1- or 2-point signal is exact", "[plan][r2c][edge]",
                   float, double) {
    using T = TestType;
    // N=1: spectrum == signal. N=2: X[0] = a+b, X[1] = a-b (unnormalized r2c).
    admiral::plan_r2c<T> p1({1});
    std::vector<std::complex<T>> s1(1);
    p1.forward(std::vector<T>{T(3)}, span(s1));
    REQUIRE(s1[0] == std::complex<T>(T(3), T(0)));

    admiral::plan_r2c<T> p2({2});
    std::vector<std::complex<T>> s2(2);
    p2.forward(std::vector<T>{T(4), T(-1)}, span(s2));
    REQUIRE(s2[0] == std::complex<T>(T(3), T(0)));
    REQUIRE(s2[1] == std::complex<T>(T(5), T(0)));

    std::vector<std::complex<T>> spec{T(3), T(5)};
    std::vector<T> back(2);
    p2.inverse(spec, span(back));
    REQUIRE(std::abs(back[0] - T(4)) <= T(8) * std::numeric_limits<T>::epsilon());
    REQUIRE(std::abs(back[1] - T(-1)) <= T(8) * std::numeric_limits<T>::epsilon());
}

TEMPLATE_TEST_CASE("plan with effort::measure picks a working route", "[plan][measure]",
                   float, double) {
    using T = TestType;
    // Modelled-domain mix: smooth, coprime, `four_step` candidates and small primes
    // (`rader`/`bluestein` race). Correctness only; the winning route is machine-local.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<double> u(-1, 1);
    for (const std::size_t n : {16u, 60u, 67u, 105u, 143u, 243u, 256u, 384u, 500u, 512u}) {
        CAPTURE(n);
        admiral::plan<T> p(n, {0, admiral::effort::measure});
        std::vector<std::complex<T>> x(n), y(n);
        for (auto& v : x) v = {T(u(rng)), T(u(rng))};
        p.forward(x.data(), y.data());
        require_close(y, reference_dft(x, /*forward=*/true), fft_tol<T>());
        std::vector<std::complex<T>> back(n);
        p.inverse(y.data(), back.data());
        require_close(back, x, fft_tol<T>());
    }
}

// The chain race elects a radix chain the cost model never ranked first: a
// different factorization, and then a permutation of it (`measure_route` stages 2
// and 3). Both stages are only shape-checked, so a legal-but-wrong chain would
// ship silently. The estimate plan is the cross-check: the estimate chain differs
// by construction and is an O(N log N) oracle where the O(N^2) `reference_dft`
// cannot reach.
TEMPLATE_TEST_CASE("effort::measure elects a chain that agrees with estimate",
                   "[plan][measure]", float, double) {
    using T = TestType;
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> u(-1, 1);
    for (const std::size_t n : {250u, 1000u, 1080u, 2520u, 3720u, 4096u, 8192u, 16384u,
                                25575u, 30000u}) {
        CAPTURE(n);
        std::vector<std::complex<T>> x(n), a(n), b(n);
        for (auto& v : x) v = {T(u(rng)), T(u(rng))};
        admiral::plan<T>(n, {0, admiral::effort::estimate}).forward(x.data(), a.data());
        // Both efforts: each stops the same race at a different point and can elect
        // a different chain. Each elected chain has to be right on its own.
        for (const admiral::effort eff : {admiral::effort::automatic, admiral::effort::measure}) {
            admiral::plan<T>(n, {0, eff}).forward(x.data(), b.data());
            require_close(b, a, fft_tol<T>());
        }
    }
}

// A small-N execution is shorter than `steady_clock` resolves. A race sampling one
// execution per candidate then times the clock, not the plan, and elects on noise.
// Every sample therefore batches enough executions to span `kMeasureSampleNs`.
TEST_CASE("measure_batch spans the sample interval at every execution cost",
          "[plan][measure]") {
    using admiral::detail::kMeasureSampleNs;
    using admiral::detail::measure_batch;

    // The race samples an execution that already spans the interval once. The
    // large-N races must keep the single-execution sampling.
    REQUIRE(measure_batch(kMeasureSampleNs) == 1);
    REQUIRE(measure_batch(kMeasureSampleNs * 100) == 1);

    // A clock that reads zero made the election a coin flip: no cost exists to
    // divide by, so the batch matches the cheapest measurable execution's.
    REQUIRE(measure_batch(0) == measure_batch(1));

    // Below the interval, the batch spans it.
    for (const std::chrono::nanoseconds::rep one : {1, 7, 15, 20, 50, 137, 999, 3999}) {
        CAPTURE(one);
        const std::size_t inner = measure_batch(one);
        REQUIRE(inner > 1);
        REQUIRE(std::size_t(one) * inner >= std::size_t(kMeasureSampleNs));
    }

    // Cheaper executions never sample in smaller batches.
    for (std::chrono::nanoseconds::rep one = 1; one < kMeasureSampleNs; ++one)
        REQUIRE(measure_batch(one) >= measure_batch(one + 1));
}

// With `ADM_MEASURE=OFF`, the API must accept the knob and behave like `estimate`
// (hint semantics, FFTW-style), with the same correct result.
TEMPLATE_TEST_CASE("plan with effort::measure at N=1 and N=2 stays exact", "[plan][measure]",
                   float, double) {
    using T = TestType;
    admiral::plan<T> p1(1, {0, admiral::effort::measure});
    std::vector<std::complex<T>> v{{T(2), T(-1)}};
    p1.forward(span(v));
    REQUIRE(v[0] == std::complex<T>(T(2), T(-1)));
    admiral::plan<T> p2(2, {0, admiral::effort::measure});
    std::vector<std::complex<T>> w{{T(1), T(0)}, {T(3), T(0)}};
    p2.forward(span(w));
    REQUIRE(w[0] == std::complex<T>(T(4), T(0)));
    REQUIRE(w[1] == std::complex<T>(T(-2), T(0)));
}
