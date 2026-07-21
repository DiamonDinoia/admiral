#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <admiral/admiral.hpp>
#include <bit>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <numbers>

using namespace Catch::Matchers;

// Relative error budget for an N-point transform in precision T:
//   eps * sqrt(N) * (log2(N) + 1) * 64
// scaled by the magnitude of the data so it applies to large-amplitude inputs.
// This tracks the O(eps * log N) FFT error growth and is meaningful for both
// float and double (unlike a flat 1e-10/1e-6 that is too loose for float and
// too tight for large double inputs).
template<typename T>
T fft_tol(std::size_t N, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(static_cast<T>(N)) + T(1);
    return eps * std::sqrt(static_cast<T>(N)) * log2N * T(64) * scale;
}

// Largest element magnitude in a complex vector (for scaling the tolerance).
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
                          T tolerance) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

TEST_CASE("Helper functions", "[helpers]") {
    SECTION("std::bit_ceil (formerly next_power_of_2)") {
        REQUIRE(std::bit_ceil(std::size_t{1}) == 1);
        REQUIRE(std::bit_ceil(std::size_t{2}) == 2);
        REQUIRE(std::bit_ceil(std::size_t{3}) == 4);
        REQUIRE(std::bit_ceil(std::size_t{4}) == 4);
        REQUIRE(std::bit_ceil(std::size_t{5}) == 8);
        REQUIRE(std::bit_ceil(std::size_t{7}) == 8);
        REQUIRE(std::bit_ceil(std::size_t{8}) == 8);
        REQUIRE(std::bit_ceil(std::size_t{9}) == 16);
        REQUIRE(std::bit_ceil(std::size_t{100}) == 128);
        REQUIRE(std::bit_ceil(std::size_t{1000}) == 1024);
    }

    SECTION("std::has_single_bit (formerly is_power_of_2)") {
        REQUIRE(std::has_single_bit(std::size_t{1}));
        REQUIRE(std::has_single_bit(std::size_t{2}));
        REQUIRE(std::has_single_bit(std::size_t{4}));
        REQUIRE(std::has_single_bit(std::size_t{8}));
        REQUIRE(std::has_single_bit(std::size_t{1024}));
        REQUIRE_FALSE(std::has_single_bit(std::size_t{0}));
        REQUIRE_FALSE(std::has_single_bit(std::size_t{3}));
        REQUIRE_FALSE(std::has_single_bit(std::size_t{5}));
        REQUIRE_FALSE(std::has_single_bit(std::size_t{100}));
    }
}

TEMPLATE_TEST_CASE("FFT size 1", "[fft][edge]", float, double) {
    using T = TestType;
    std::vector<std::complex<T>> input = {{T(1), T(0)}};
    std::vector<std::complex<T>> output(1);

    admiral::forward(std::span<const std::complex<T>>(input), std::span(output));

    REQUIRE(output.size() == 1);
    REQUIRE_THAT(output[0].real(), WithinAbs(1.0, 1e-6));
    REQUIRE_THAT(output[0].imag(), WithinAbs(0.0, 1e-6));
}

TEMPLATE_TEST_CASE("FFT round-trip (power-of-2 sizes)", "[fft][roundtrip]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::sin(T(2) * std::numbers::pi_v<T> * T(i) / T(N)),
                                       std::cos(T(2) * std::numbers::pi_v<T> * T(i) / T(N)));
        }

        // Forward then inverse should recover input
        std::vector<std::complex<T>> fft_result(N);
        admiral::forward(std::span(input), std::span(fft_result));

        std::vector<std::complex<T>> recovered(N);
        admiral::inverse(std::span(fft_result), std::span(recovered));

        REQUIRE(vectors_approx_equal(input, recovered, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 2") { test_roundtrip(2); }
    SECTION("Size 4") { test_roundtrip(4); }
    SECTION("Size 8") { test_roundtrip(8); }
    SECTION("Size 16") { test_roundtrip(16); }
    SECTION("Size 32") { test_roundtrip(32); }
    SECTION("Size 64") { test_roundtrip(64); }
    SECTION("Size 128") { test_roundtrip(128); }
    SECTION("Size 256") { test_roundtrip(256); }
    SECTION("Size 512") { test_roundtrip(512); }
    SECTION("Size 1024") { test_roundtrip(1024); }
}

TEMPLATE_TEST_CASE("FFT round-trip (prime sizes)", "[fft][roundtrip][prime]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(T(i + 1), T(N - i));
        }

        // Forward then inverse should recover input
        std::vector<std::complex<T>> fft_result(N);
        admiral::forward(std::span(input), std::span(fft_result));

        std::vector<std::complex<T>> recovered(N);
        admiral::inverse(std::span(fft_result), std::span(recovered));

        REQUIRE(vectors_approx_equal(input, recovered, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 3") { test_roundtrip(3); }
    SECTION("Size 5") { test_roundtrip(5); }
    SECTION("Size 7") { test_roundtrip(7); }
    SECTION("Size 11") { test_roundtrip(11); }
    SECTION("Size 13") { test_roundtrip(13); }
    SECTION("Size 17") { test_roundtrip(17); }
    SECTION("Size 19") { test_roundtrip(19); }
    SECTION("Size 23") { test_roundtrip(23); }
    SECTION("Size 29") { test_roundtrip(29); }
    SECTION("Size 31") { test_roundtrip(31); }
}

TEMPLATE_TEST_CASE("FFT round-trip (composite sizes)", "[fft][roundtrip][composite]", float, double) {
    using T = TestType;
    auto test_roundtrip = [&](std::size_t N) {
        // Create test input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(std::cos(T(i)), std::sin(T(i) * T(0.5)));
        }

        // Forward then inverse should recover input
        std::vector<std::complex<T>> fft_result(N);
        admiral::forward(std::span(input), std::span(fft_result));

        std::vector<std::complex<T>> recovered(N);
        admiral::inverse(std::span(fft_result), std::span(recovered));

        REQUIRE(vectors_approx_equal(input, recovered, fft_tol<T>(N, max_magnitude(input))));
    };

    SECTION("Size 6") { test_roundtrip(6); }
    SECTION("Size 10") { test_roundtrip(10); }
    SECTION("Size 12") { test_roundtrip(12); }
    SECTION("Size 15") { test_roundtrip(15); }
    SECTION("Size 20") { test_roundtrip(20); }
    SECTION("Size 24") { test_roundtrip(24); }
    SECTION("Size 30") { test_roundtrip(30); }
    SECTION("Size 100") { test_roundtrip(100); }
}

TEMPLATE_TEST_CASE("FFT known values (size 4)", "[fft][known]", float, double) {
    using T = TestType;
    // Test case: input = [1, 0, 0, 0]
    // FFT should give [1, 1, 1, 1]
    std::vector<std::complex<T>> input = {
        {T(1), T(0)}, {T(0), T(0)}, {T(0), T(0)}, {T(0), T(0)}
    };
    std::vector<std::complex<T>> output(4);
    admiral::forward(std::span(input), std::span(output));

    const auto tol = fft_tol<T>(4);
    REQUIRE(output.size() == 4);
    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE(std::abs(output[i] - std::complex<T>(T(1), T(0))) <= tol);
    }
}

TEMPLATE_TEST_CASE("FFT known values (size 4, impulse)", "[fft][known]", float, double) {
    using T = TestType;
    // Test case: input = [1, 1, 1, 1]
    // FFT should give [4, 0, 0, 0]
    std::vector<std::complex<T>> input = {
        {T(1), T(0)}, {T(1), T(0)}, {T(1), T(0)}, {T(1), T(0)}
    };
    std::vector<std::complex<T>> output(4);
    admiral::forward(std::span(input), std::span(output));

    const auto tol = fft_tol<T>(4, T(4));
    REQUIRE(output.size() == 4);
    REQUIRE(std::abs(output[0] - std::complex<T>(T(4), T(0))) <= tol);
    for (std::size_t i = 1; i < 4; ++i) {
        REQUIRE(std::abs(output[i]) <= tol);
    }
}

TEMPLATE_TEST_CASE("FFT Parseval's theorem", "[fft][properties]", float, double) {
    using T = TestType;
    // Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2)
    // where X = FFT(x)

    auto test_parseval = [](std::size_t N) {
        // Create random-looking input
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(
                std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.3)),
                std::sin(T(i) * T(0.2)) + std::cos(T(i) * T(0.4))
            );
        }

        // Compute energy in time domain
        T time_energy = T(0);
        for (const auto& x : input) {
            time_energy += std::norm(x);  // |x|^2
        }

        // Compute FFT
        std::vector<std::complex<T>> fft_result(N);
        admiral::forward(std::span(input), std::span(fft_result));

        // Compute energy in frequency domain
        T freq_energy = T(0);
        for (const auto& X : fft_result) {
            freq_energy += std::norm(X);  // |X|^2
        }
        freq_energy /= T(N);

        // Check Parseval's theorem (energy ~ N, so scale the tolerance by it)
        REQUIRE_THAT(static_cast<double>(time_energy),
                     WithinAbs(static_cast<double>(freq_energy),
                               static_cast<double>(fft_tol<T>(N, time_energy))));
    };

    SECTION("Size 8") { test_parseval(8); }
    SECTION("Size 16") { test_parseval(16); }
    SECTION("Size 7 (prime)") { test_parseval(7); }
    SECTION("Size 15 (composite)") { test_parseval(15); }
}

TEMPLATE_TEST_CASE("FFT linearity", "[fft][properties]", float, double) {
    using T = TestType;
    // FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)

    const std::size_t N = 16;
    const T a = T(2.5);
    const T b = T(-1.3);

    std::vector<std::complex<T>> x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::complex<T>(std::sin(T(i)), std::cos(T(i)));
        y[i] = std::complex<T>(std::cos(T(i) * T(2)), std::sin(T(i) * T(0.5)));
    }

    // Compute FFT(a*x + b*y)
    std::vector<std::complex<T>> combined(N);
    for (std::size_t i = 0; i < N; ++i) {
        combined[i] = a * x[i] + b * y[i];
    }
    std::vector<std::complex<T>> fft_combined(N);
    admiral::forward(std::span(combined), std::span(fft_combined));

    // Compute a*FFT(x) + b*FFT(y)
    std::vector<std::complex<T>> fft_x(N), fft_y(N);
    admiral::forward(std::span(x), std::span(fft_x));
    admiral::forward(std::span(y), std::span(fft_y));

    std::vector<std::complex<T>> linear_combo(N);
    for (std::size_t i = 0; i < N; ++i) {
        linear_combo[i] = a * fft_x[i] + b * fft_y[i];
    }

    // They should be equal
    REQUIRE(vectors_approx_equal(fft_combined, linear_combo,
                                 fft_tol<T>(N, max_magnitude(fft_combined))));
}

TEMPLATE_TEST_CASE("In-place transform", "[fft][inplace]", float, double) {
    using T = TestType;
    // Test that input=output works (in-place transform)
    const std::size_t N = 16;

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = std::complex<T>(std::sin(T(i)), std::cos(T(i)));
    }

    // Save original for comparison
    std::vector<std::complex<T>> original = data;

    // In-place forward FFT
    admiral::forward(std::span(data), std::span(data));

    // Data should have changed
    REQUIRE_FALSE(vectors_approx_equal(data, original, fft_tol<T>(N)));

    // In-place inverse FFT should recover original
    admiral::inverse(std::span(data), std::span(data));

    REQUIRE(vectors_approx_equal(data, original, fft_tol<T>(N, max_magnitude(original))));
}
