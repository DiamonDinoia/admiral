#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <array>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("FFT size 1", "[fft][edge]", float, double) {
    using T = TestType;
    std::vector<std::complex<T>> input = {{T(1), T(0)}};
    std::vector<std::complex<T>> output(1);

    admiral::forward(admiral::span<const std::complex<T>>(input), admiral::span(output));

    REQUIRE(output.size() == 1);
    REQUIRE(output[0] == input[0]);
}

TEMPLATE_TEST_CASE("FFT known values (size 4)", "[fft][known]", float, double) {
    using T = TestType;
    std::vector<std::complex<T>> input = {
        {T(1), T(0)}, {T(0), T(0)}, {T(0), T(0)}, {T(0), T(0)}
    };
    std::vector<std::complex<T>> output(4);
    admiral::forward(admiral::span(input), admiral::span(output));

    REQUIRE(output.size() == 4);
    require_close(output, std::vector<std::complex<T>>(4, std::complex<T>(T(1), T(0))), fft_tol<T>());
}

TEMPLATE_TEST_CASE("FFT known values (size 4, impulse)", "[fft][known]", float, double) {
    using T = TestType;
    std::vector<std::complex<T>> input = {
        {T(1), T(0)}, {T(1), T(0)}, {T(1), T(0)}, {T(1), T(0)}
    };
    std::vector<std::complex<T>> output(4);
    admiral::forward(admiral::span(input), admiral::span(output));

    REQUIRE(output.size() == 4);
    std::vector<std::complex<T>> want(4, std::complex<T>(T(0), T(0)));
    want[0] = std::complex<T>(T(4), T(0));
    require_close(output, want, fft_tol<T>());
}

TEMPLATE_TEST_CASE("FFT Parseval's theorem", "[fft][properties]", float, double) {
    using T = TestType;

    auto test_parseval = [](std::size_t N) {
        std::vector<std::complex<T>> input(N);
        for (std::size_t i = 0; i < N; ++i) {
            input[i] = std::complex<T>(
                std::sin(T(i) * T(0.1)) + std::cos(T(i) * T(0.3)),
                std::sin(T(i) * T(0.2)) + std::cos(T(i) * T(0.4))
            );
        }

        std::vector<std::complex<T>> fft_result(N);
        admiral::forward(admiral::span(input), admiral::span(fft_result));

        require_parseval<T>(input, fft_result, N, 1.0);
    };

    SECTION("Size 8") { test_parseval(8); }
    SECTION("Size 16") { test_parseval(16); }
    SECTION("Size 7 (prime)") { test_parseval(7); }
    SECTION("Size 15 (composite)") { test_parseval(15); }
}

TEMPLATE_TEST_CASE("FFT linearity", "[fft][properties]", float, double) {
    using T = TestType;

    const std::size_t N = 16;
    const T a = T(2.5);
    const T b = T(-1.3);

    std::vector<std::complex<T>> x(N), y(N);
    for (std::size_t i = 0; i < N; ++i) {
        x[i] = std::complex<T>(std::sin(T(i)), std::cos(T(i)));
        y[i] = std::complex<T>(std::cos(T(i) * T(2)), std::sin(T(i) * T(0.5)));
    }

    std::vector<std::complex<T>> combined(N);
    for (std::size_t i = 0; i < N; ++i) {
        combined[i] = a * x[i] + b * y[i];
    }
    std::vector<std::complex<T>> fft_combined(N);
    admiral::forward(admiral::span(combined), admiral::span(fft_combined));

    std::vector<std::complex<T>> fft_x(N), fft_y(N);
    admiral::forward(admiral::span(x), admiral::span(fft_x));
    admiral::forward(admiral::span(y), admiral::span(fft_y));

    std::vector<std::complex<T>> linear_combo(N);
    for (std::size_t i = 0; i < N; ++i) {
        linear_combo[i] = a * fft_x[i] + b * fft_y[i];
    }

    require_close(fft_combined, linear_combo, fft_tol<T>());
}

TEMPLATE_TEST_CASE("In-place transform", "[fft][inplace]", float, double) {
    using T = TestType;
    const std::size_t N = 16;

    std::vector<std::complex<T>> data(N);
    for (std::size_t i = 0; i < N; ++i) {
        data[i] = std::complex<T>(std::sin(T(i)), std::cos(T(i)));
    }

    std::vector<std::complex<T>> original = data;

    admiral::forward(admiral::span(data), admiral::span(data));

    REQUIRE(relerrtwonorm(original, data) > fft_tol<T>());

    admiral::inverse(admiral::span(data), admiral::span(data));

    require_close(data, original, fft_tol<T>());
}

TEMPLATE_TEST_CASE("one-shot validation", "[fft][oneshot]", float, double) {
    using T = TestType;
    std::vector<std::complex<T>> in(64, {T(1), T(0)}), out(64), short_out(32);
    const auto cin = admiral::span<const std::complex<T>>(in);

    REQUIRE_THROWS_AS(admiral::forward(cin, admiral::span(short_out)), admiral::size_error);
    REQUIRE_THROWS_AS(admiral::inverse(cin, admiral::span(short_out)), admiral::size_error);

    std::vector<std::complex<T>> ein, eout;
    REQUIRE_NOTHROW(
        admiral::forward(admiral::span<const std::complex<T>>(ein), admiral::span(eout)));
    REQUIRE_NOTHROW(
        admiral::inverse(admiral::span<const std::complex<T>>(ein), admiral::span(eout)));
}

TEMPLATE_TEST_CASE("one-shot with nthreads transforms correctly", "[fft][oneshot][threads]",
                   float, double) {
    using T = TestType;
    const std::size_t N = 1 << 16;
    std::vector<std::complex<T>> x(N), y(N);
    for (std::size_t j = 0; j < N; ++j) x[j] = unit_phasor<T>(turn_fraction(3, j, N));

    admiral::forward(admiral::span<const std::complex<T>>(x), admiral::span(y), {2});

    REQUIRE_THAT(double(std::abs(y[3])), WithinAbsT(double(N), double(N) * fft_tol<T>()));

    std::vector<std::complex<T>> back(N);
    admiral::inverse(admiral::span<const std::complex<T>>(y), admiral::span(back), {2});
    require_close(back, x, fft_tol<T>());
}

TEMPLATE_TEST_CASE("N-D one-shot with nthreads transforms correctly",
                   "[fft][oneshot][threads]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{64u, 64u};
    const std::size_t N = shape[0] * shape[1];
    std::vector<std::complex<T>> x(N);
    for (std::size_t j = 0; j < N; ++j) {
        x[j] = unit_phasor<T>(turn_fraction(5, j / shape[1], shape[0]))
               * unit_phasor<T>(turn_fraction(3, j % shape[1], shape[1]));
    }

    std::vector<std::complex<T>> y = x;
    admiral::forward(y.data(), shape, {3});
    REQUIRE_THAT(double(std::abs(y[5 * shape[1] + 3])),
                 WithinAbsT(double(N), double(N) * fft_tol<T>()));
    admiral::inverse(y.data(), shape, {3});
    require_close(y, x, fft_tol<T>());
}

TEMPLATE_TEST_CASE("options::debug traces without changing the result",
                   "[fft][options][debug]", float, double) {
    using T = TestType;
    constexpr unsigned kVerbose = 3;

    auto ramp = [](std::size_t n) {
        std::vector<std::complex<T>> v(n);
        for (std::size_t j = 0; j < n; ++j) v[j] = {T(j % 7) - T(3), T(j % 5) - T(2)};
        return v;
    };

    SECTION("complex, 1-D and 2-D") {
        for (const std::vector<std::size_t>& shape :
             {std::vector<std::size_t>{1024}, std::vector<std::size_t>{60},
              std::vector<std::size_t>{32, 24}}) {
            CAPTURE(shape);
            std::size_t n = 1;
            for (const std::size_t e : shape) n *= e;
            const auto x = ramp(n);
            const admiral::span<const std::size_t> sp(shape);
            const admiral::options loud{0, admiral::effort::estimate, kVerbose};

            std::vector<std::complex<T>> v = x;
            admiral::plan<T>(sp).forward(v.data(), v.data());
            const auto quiet = v;
            v = x;
            admiral::plan<T>(sp, loud).forward(v.data(), v.data());
            REQUIRE(v == quiet);

            std::vector<std::complex<T>> dst(n);
            admiral::plan<T>(sp).inverse(quiet.data(), dst.data());
            const auto quiet_oop = dst;
            admiral::plan<T>(sp, loud).inverse(quiet.data(), dst.data());
            REQUIRE(dst == quiet_oop);
        }
    }

    SECTION("real, r2c and c2r") {
        const std::array<std::size_t, 2> shape{16, 12};
        admiral::plan_r2c<T> quiet(shape);
        admiral::plan_r2c<T> loud(shape, {0, admiral::effort::estimate, kVerbose});
        std::vector<T> in(quiet.real_size());
        for (std::size_t j = 0; j < in.size(); ++j) in[j] = T(j % 11) - T(5);

        std::vector<std::complex<T>> spec(quiet.cplx_size());
        quiet.forward(in.data(), spec.data());
        const auto spec_quiet = spec;
        loud.forward(in.data(), spec.data());
        REQUIRE(spec == spec_quiet);

        std::vector<T> back(in.size());
        quiet.inverse(spec.data(), back.data());
        const auto back_quiet = back;
        spec = spec_quiet;
        loud.inverse(spec.data(), back.data());
        REQUIRE(back == back_quiet);
    }
}
