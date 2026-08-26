#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <vector>
#include <complex>
#include <cmath>
#include <limits>

using namespace Catch::Matchers;

// Three closed forms of the DFT, each checked as a whole spectrum: a delta
// transforms to a constant, a constant to a delta of height N, and a tone at bin k
// to a delta of height N at k.
namespace {

template<typename T>
std::vector<std::complex<T>> spectrum_of(const std::vector<std::complex<T>>& in) {
    std::vector<std::complex<T>> out(in.size());
    admiral::forward(admiral::span(in), admiral::span(out));
    return out;
}

template<typename T>
std::vector<std::complex<T>> delta(std::size_t N, std::size_t at, T height) {
    std::vector<std::complex<T>> v(N, std::complex<T>(T(0), T(0)));
    v[at] = std::complex<T>(height, T(0));
    return v;
}

} // namespace

TEMPLATE_TEST_CASE("FFT analytical: delta, constant, tone", "[fft][analytical]", float, double) {
    using T = TestType;
    constexpr std::size_t N = 64;
    const double tol = fft_tol<T>();

    require_close(spectrum_of(delta<T>(N, 0, T(1))),
                  std::vector<std::complex<T>>(N, std::complex<T>(T(1), T(0))), tol);

    require_close(spectrum_of(std::vector<std::complex<T>>(N, std::complex<T>(T(1), T(0)))),
                  delta<T>(N, 0, T(N)), tol);

    constexpr std::size_t k = 5;
    std::vector<std::complex<T>> tone(N);
    for (std::size_t n = 0; n < N; ++n) tone[n] = unit_phasor<T>(turn_fraction(k, n, N));
    require_close(spectrum_of(tone), delta<T>(N, k, T(N)), tol);
}

TEMPLATE_TEST_CASE("FFT analytical: Gaussian spectrum matches its closed form",
                   "[fft][analytical]", float, double) {
    using T = TestType;
    // The periodized Gaussian's DFT is, up to double-exponentially small images,
    // the sampled Gaussian: X_k = sigma*sqrt(2*pi)*exp(-k'^2/(2 s^2)) with
    // k' = 2*pi*sigma*min(k,N-k)/N. Check values rather than decay alone. A decay-only
    // check passes a DC-dominated transform.
    const std::size_t N = 128;
    // Periodization wraps the tails, so the only approximation is the spectral
    // alias sum, O(exp(-2 pi^2 sigma^2)): e^-1243 at N=128 with sigma = N/16.
    const T sigma = T(N) / T(16);
    std::vector<std::complex<T>> input(N);
    for (std::size_t n = 0; n < N; ++n) {
        const T t = T(n) - T(N) / T(2);
        input[n] = {std::exp(-t * t / (T(2) * sigma * sigma)), T(0)};
    }

    std::vector<std::complex<T>> output(N);
    admiral::forward(admiral::span(input), admiral::span(output));

    const T peak = sigma * std::sqrt(T(2) * admiral::detail::numbers::pi_v<T>);
    // The test measures error against the peak rather than each bin, so the budget
    // covers the whole spectrum's accumulated rounding rather than one bin's.
    constexpr T kPeakRelTol = T(50);
    for (std::size_t k = 0; k <= N / 2; ++k) {
        const T kk = T(2) * admiral::detail::numbers::pi_v<T> * sigma * T(k) / T(N);
        // The Gaussian peaks at INDEX N/2, and a half-period shift is a (-1)^k phase.
        const T want = ((k & 1) ? T(-1) : T(1)) * peak * std::exp(-kk * kk / T(2));
        if (std::abs(want) < peak * T(1e-30)) break;  // both sides have underflowed there
        REQUIRE(std::abs(output[k] - want)
                <= kPeakRelTol * std::numeric_limits<T>::epsilon() * peak);
        // Real signal: the spectrum is conjugate-symmetric.
        REQUIRE(std::abs(output[k] - std::conj(output[(N - k) % N]))
                <= kPeakRelTol * std::numeric_limits<T>::epsilon() * peak);
    }
}

TEMPLATE_TEST_CASE("FFT round-trip with scaled tolerance", "[fft][roundtrip][analytical]", float, double) {
    using T = TestType;
    auto test_roundtrip = [](std::size_t N) {
        const double tolerance = fft_tol<T>();

        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            T t = T(i) / T(N);
            input[i] = std::complex<T>(
                std::sin(T(2) * admiral::detail::numbers::pi_v<T> * t * T(3)) + T(0.5) * std::cos(T(2) * admiral::detail::numbers::pi_v<T> * t * T(7)),
                std::cos(T(2) * admiral::detail::numbers::pi_v<T> * t * T(5)) + T(0.3) * std::sin(T(2) * admiral::detail::numbers::pi_v<T> * t * T(11))
            );
        }

        std::vector<std::complex<T>> original = input;

        std::vector<std::complex<T>> freq(N);
        admiral::forward(admiral::span(input), admiral::span(freq));

        std::vector<std::complex<T>> recovered(N);
        admiral::inverse(admiral::span(freq), admiral::span(recovered));

        require_close(original, recovered, tolerance);
    };

    SECTION("Size 16") { test_roundtrip(16); }
    SECTION("Size 64") { test_roundtrip(64); }
    SECTION("Size 256") { test_roundtrip(256); }
    SECTION("Size 1024") { test_roundtrip(1024); }
}

TEMPLATE_TEST_CASE("FFT convolution theorem", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // Convolution theorem: FFT(x * y) = FFT(x) . FFT(y), * = circular convolution.
    const std::size_t N = 32;
    const double tolerance = fft_tol<T>(10);

    std::vector<std::complex<T>> x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = {std::sin(T(i) * T(0.2)), T(0)};
        y[i] = {std::cos(T(i) * T(0.3)), T(0)};
    }

    // Circular convolution in the time domain.
    std::vector<std::complex<T>> conv_time(N, {T(0), T(0)});
    for (std::size_t n = 0; n < N; ++n) {
        for (std::size_t m = 0; m < N; ++m) {
            conv_time[n] += x[m] * y[(n - m + N) % N];
        }
    }

    std::vector<std::complex<T>> X(N), Y(N);
    admiral::forward(admiral::span(x), admiral::span(X));
    admiral::forward(admiral::span(y), admiral::span(Y));

    std::vector<std::complex<T>> XY(N);
    for (std::size_t i = 0; i < N; ++i) {
        XY[i] = X[i] * Y[i];
    }

    std::vector<std::complex<T>> conv_freq(N);
    admiral::inverse(admiral::span(XY), admiral::span(conv_freq));

    require_close(conv_time, conv_freq, tolerance);
}

TEMPLATE_TEST_CASE("FFT time shift property", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // Time shift: FFT(x[n-k]) = e^(-2pi i k w / N) . FFT(x[n]).
    const std::size_t N = 32;
    const std::size_t shift = 5;
    const double tolerance = fft_tol<T>(10);

    std::vector<std::complex<T>> x(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = {std::sin(T(i) * T(0.3)) + std::cos(T(i) * T(0.7)), T(0)};
    }

    std::vector<std::complex<T>> x_shifted(N);
    for (std::size_t i = 0; i < N; ++i) {
        x_shifted[i] = x[(i + N - shift) % N];
    }

    std::vector<std::complex<T>> X(N), X_shifted(N);
    admiral::forward(admiral::span(x), admiral::span(X));
    admiral::forward(admiral::span(x_shifted), admiral::span(X_shifted));

    std::vector<std::complex<T>> X_phase_shifted(N);
    for (std::size_t k = 0; k < N; ++k) {
        X_phase_shifted[k] = X[k] * unit_phasor<T>(-turn_fraction(shift, k, N));
    }

    require_close(X_shifted, X_phase_shifted, tolerance);
}

TEMPLATE_TEST_CASE("FFT symmetry for real signals", "[fft][analytical][properties]", float, double) {
    using T = TestType;
    // Hermitian symmetry for real input: X[k] = conj(X[N-k]).
    const std::size_t N = 64;

    std::vector<std::complex<T>> input(N);
    for (std::size_t i = 0; i < N; ++i) {
        input[i] = {std::sin(T(i) * T(0.2)) + std::cos(T(i) * T(0.5)), T(0)};
    }

    std::vector<std::complex<T>> output(N);
    admiral::forward(admiral::span(input), admiral::span(output));

    // X[k] == conj(X[N-k]), compared as a whole spectrum. Relative L2 needs no
    // amplitude fudge for the O(N) bins. k=0 and k=N/2 map to themselves, so this
    // check also forces those two bins real and no separate check for them exists.
    std::vector<std::complex<T>> mirrored(N);
    for (std::size_t k = 0; k < N; ++k) mirrored[k] = std::conj(output[(N - k) % N]);
    require_close(output, mirrored, fft_tol<T>());
}
