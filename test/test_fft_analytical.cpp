#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <admiral/admiral.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>
#include <numbers>

using namespace Catch::Matchers;

// Compute tolerance based on machine precision and problem size.
// FFT error grows as O(eps * sqrt(N) * log2(N)); the *64 safety factor and the
// +1 on the log keep the bound meaningful for both float and double across the
// whole size range (a flat constant is too tight for float at large N).
template<typename T>
T compute_tolerance(std::size_t N) {
    constexpr T eps = std::numeric_limits<T>::epsilon();
    const T log_factor = std::log2(static_cast<T>(N)) + T(1);
    const T sqrt_factor = std::sqrt(static_cast<T>(N));
    return eps * sqrt_factor * log_factor * T(64);
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

TEMPLATE_TEST_CASE("FFT analytical: delta function", "[fft][analytical]", float, double) {
    using T = TestType;
    // FFT of delta [1, 0, 0, ...] should be constant [1, 1, 1, ...]
    const std::size_t N = 64;
    const T tolerance = compute_tolerance<T>(N);

    std::vector<std::complex<T>> input(N, {T(0), T(0)});
    input[0] = {T(1), T(0)};  // Delta at origin

    std::vector<std::complex<T>> output(N);
    admiral::forward(std::span(input), std::span(output));

    // All bins should be approximately 1
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE(std::abs(output[i] - std::complex<T>(T(1), T(0))) <= tolerance);
    }
}

TEMPLATE_TEST_CASE("FFT analytical: constant function", "[fft][analytical]", float, double) {
    using T = TestType;
    // FFT of constant [1, 1, 1, ...] should be [N, 0, 0, ...]
    const std::size_t N = 64;
    const T tolerance = compute_tolerance<T>(N) * T(N);

    std::vector<std::complex<T>> input(N, {T(1), T(0)});

    std::vector<std::complex<T>> output(N);
    admiral::forward(std::span(input), std::span(output));

    // DC component should be N
    REQUIRE(std::abs(output[0] - std::complex<T>(T(N), T(0))) <= tolerance);

    // All other bins should be ~0
    for (std::size_t i = 1; i < N; ++i) {
        REQUIRE(std::abs(output[i]) <= tolerance);
    }
}

TEMPLATE_TEST_CASE("FFT analytical: complex exponential", "[fft][analytical]", float, double) {
    using T = TestType;
    // FFT of e^(2pi i k n / N) should give impulse at bin k
    const std::size_t N = 64;
    const std::size_t k = 5;  // Frequency bin
    const T tolerance = compute_tolerance<T>(N) * T(N);

    // Generate complex exponential at frequency k
    std::vector<std::complex<T>> input(N);
    for (std::size_t n = 0; n < N; ++n) {
        T phase = T(2) * std::numbers::pi_v<T> * T(k * n) / T(N);
        input[n] = std::complex<T>(std::cos(phase), std::sin(phase));
    }

    std::vector<std::complex<T>> output(N);
    admiral::forward(std::span(input), std::span(output));

    // Bin k should have magnitude N
    REQUIRE_THAT(static_cast<double>(std::abs(output[k])),
                 WithinAbs(double(N), static_cast<double>(tolerance)));

    // All other bins should be ~0
    for (std::size_t i = 0; i < N; ++i) {
        if (i != k) {
            REQUIRE(std::abs(output[i]) <= tolerance);
        }
    }
}

TEMPLATE_TEST_CASE("FFT analytical: Gaussian properties", "[fft][analytical]", float, double) {
    using T = TestType;
    // Fourier transform of Gaussian is Gaussian
    // For DFT, this is approximate but should be close for well-sampled Gaussians
    const std::size_t N = 128;

    // Create discrete Gaussian centered at origin
    // sigma = N/8 gives good sampling
    const T sigma = T(N) / T(8);
    std::vector<std::complex<T>> input(N);

    for (std::size_t n = 0; n < N; ++n) {
        // Shift to center around N/2 for periodicity
        T t = T(n) - T(N) / T(2);
        T gauss = std::exp(-t * t / (T(2) * sigma * sigma));
        input[n] = {gauss, T(0)};
    }

    // FFT
    std::vector<std::complex<T>> output(N);
    admiral::forward(std::span(input), std::span(output));

    // The magnitude should also look like a Gaussian (in frequency domain)
    // Check that it's peaked at DC and falls off
    T dc_magnitude = std::abs(output[0]);

    // Magnitude should decrease as we move away from DC
    for (std::size_t k = 1; k < N/4; ++k) {
        T mag = std::abs(output[k]);
        // Each bin should generally be smaller than the previous (monotonic decrease initially)
        if (k < 8) {  // Check first few bins
            REQUIRE(mag < dc_magnitude);
        }
    }
}

TEMPLATE_TEST_CASE("FFT round-trip with scaled tolerance", "[fft][roundtrip][analytical]", float, double) {
    using T = TestType;
    // Test that forward + inverse recovers input with proper tolerance

    auto test_roundtrip = [](std::size_t N) {
        const T tolerance = compute_tolerance<T>(N);

        // Create test input with various frequencies
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            T t = T(i) / T(N);
            input[i] = std::complex<T>(
                std::sin(T(2) * std::numbers::pi_v<T> * t * T(3)) + T(0.5) * std::cos(T(2) * std::numbers::pi_v<T> * t * T(7)),
                std::cos(T(2) * std::numbers::pi_v<T> * t * T(5)) + T(0.3) * std::sin(T(2) * std::numbers::pi_v<T> * t * T(11))
            );
        }

        std::vector<std::complex<T>> original = input;

        // Forward
        std::vector<std::complex<T>> freq(N);
        admiral::forward(std::span(input), std::span(freq));

        // Inverse
        std::vector<std::complex<T>> recovered(N);
        admiral::inverse(std::span(freq), std::span(recovered));

        // Check recovery
        REQUIRE(vectors_approx_equal(original, recovered, tolerance));
    };

    SECTION("Size 16") { test_roundtrip(16); }
    SECTION("Size 64") { test_roundtrip(64); }
    SECTION("Size 256") { test_roundtrip(256); }
    SECTION("Size 1024") { test_roundtrip(1024); }
}

TEMPLATE_TEST_CASE("FFT convolution theorem", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // Convolution theorem: FFT(x * y) = FFT(x) . FFT(y)
    // where * is circular convolution
    const std::size_t N = 32;
    const T tolerance = compute_tolerance<T>(N) * T(10);

    // Create two test signals
    std::vector<std::complex<T>> x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = {std::sin(T(i) * T(0.2)), T(0)};
        y[i] = {std::cos(T(i) * T(0.3)), T(0)};
    }

    // Compute circular convolution in time domain
    std::vector<std::complex<T>> conv_time(N, {T(0), T(0)});
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t m = 0; m < N; ++m) {
            conv_time[n] += x[m] * y[(n - m + N) % N];
        }
    }

    // Compute via frequency domain: FFT(x) . FFT(y), then IFFT
    std::vector<std::complex<T>> X(N), Y(N);
    admiral::forward(std::span(x), std::span(X));
    admiral::forward(std::span(y), std::span(Y));

    // Pointwise multiply
    std::vector<std::complex<T>> XY(N);
    for (std::size_t i = 0; i < N; ++i) {
        XY[i] = X[i] * Y[i];
    }

    // Inverse FFT
    std::vector<std::complex<T>> conv_freq(N);
    admiral::inverse(std::span(XY), std::span(conv_freq));

    // Should match
    REQUIRE(vectors_approx_equal(conv_time, conv_freq, tolerance));
}

TEMPLATE_TEST_CASE("FFT time shift property", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // Time shift: FFT(x[n-k]) = e^(-2pi i k w / N) . FFT(x[n])
    const std::size_t N = 32;
    const std::size_t shift = 5;
    const T tolerance = compute_tolerance<T>(N) * T(10);

    // Original signal
    std::vector<std::complex<T>> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = {std::sin(T(i) * T(0.3)) + std::cos(T(i) * T(0.7)), T(0)};
    }

    // Shifted signal (circular shift)
    std::vector<std::complex<T>> x_shifted(N);
    for (std::size_t i = 0; i < N; ++i) {
        x_shifted[i] = x[(i + N - shift) % N];
    }

    // FFT of both
    std::vector<std::complex<T>> X(N), X_shifted(N);
    admiral::forward(std::span(x), std::span(X));
    admiral::forward(std::span(x_shifted), std::span(X_shifted));

    // Apply phase shift to X
    std::vector<std::complex<T>> X_phase_shifted(N);
    for (std::size_t k = 0; k < N; ++k) {
        T phase = T(-2) * std::numbers::pi_v<T> * T(shift * k) / T(N);
        std::complex<T> twiddle(std::cos(phase), std::sin(phase));
        X_phase_shifted[k] = X[k] * twiddle;
    }

    // Should match
    REQUIRE(vectors_approx_equal(X_shifted, X_phase_shifted, tolerance));
}

TEMPLATE_TEST_CASE("FFT symmetry for real signals", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // For real input, X[k] = conj(X[N-k]) (Hermitian symmetry)
    const std::size_t N = 64;
    const T tolerance = compute_tolerance<T>(N);

    // Create real signal
    std::vector<std::complex<T>> input(N);
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = {std::sin(T(i) * T(0.2)) + std::cos(T(i) * T(0.5)), T(0)};
    }

    std::vector<std::complex<T>> output(N);
    admiral::forward(std::span(input), std::span(output));

    // Check Hermitian symmetry: X[k] = conj(X[N-k]). Scale tolerance by N so it
    // is meaningful relative to the O(N) bin amplitudes.
    for (std::size_t k = 1; k < N/2; ++k) {
        std::complex<T> Xk = output[k];
        std::complex<T> XNk_conj = std::conj(output[N - k]);
        REQUIRE(std::abs(Xk - XNk_conj) <= tolerance * T(N));
    }

    // DC and Nyquist bins should be real
    REQUIRE_THAT(static_cast<double>(output[0].imag()),
                 WithinAbs(0.0, static_cast<double>(tolerance * T(N))));
    if (N % 2 == 0) {
        REQUIRE_THAT(static_cast<double>(output[N/2].imag()),
                     WithinAbs(0.0, static_cast<double>(tolerance * T(N))));
    }
}
