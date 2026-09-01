
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/admiral.hpp>
#include <admiral/detail/bluestein.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/rader.hpp>

#include "utils/reference.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("exhaustive integer-N round-trip and forward vs naive DFT",
                   "[coverage][sweep]", float, double) {
    using T = TestType;
    for (std::size_t N = 2; N <= 512; ++N) {
        const auto in = make_signal<T>(N);

        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<T>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<T>>(spec), admiral::span(recov));
        INFO("round-trip N=" << N);
        require_close(in, recov, fft_tol<T>(2));

        const auto ref = reference_dft<T>(in, true);
        INFO("vs-naive N=" << N);
        require_close(spec, ref, fft_tol<T>());
    }
}

TEMPLATE_TEST_CASE("known-answer impulse, DC, and Parseval", "[coverage][known]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 14> sizes = {1, 2, 3, 5, 7, 8, 12, 16, 60, 64, 67, 120, 128, 251};
    for (std::size_t N : sizes) {
        INFO("N=" << N);

        {
            std::vector<std::complex<T>> x(N, std::complex<T>(0, 0));
            x.at(0) = std::complex<T>(1, 0);
            std::vector<std::complex<T>> X(N);
            admiral::forward(admiral::span<const std::complex<T>>(x), admiral::span(X));
            require_close(X, std::vector<std::complex<T>>(N, std::complex<T>(1, 0)), fft_tol<T>());
        }

        {
            std::vector<std::complex<T>> x(N, std::complex<T>(T(2), T(0)));
            std::vector<std::complex<T>> X(N);
            admiral::forward(admiral::span<const std::complex<T>>(x), admiral::span(X));
            std::vector<std::complex<T>> want(N, std::complex<T>(0, 0));
            want.at(0) = std::complex<T>(T(2) * T(N), T(0));
            require_close(X, want, fft_tol<T>());
        }

        {
            const auto x = make_signal<T>(N);
            std::vector<std::complex<T>> X(N);
            admiral::forward(admiral::span<const std::complex<T>>(x), admiral::span(X));
            require_parseval<T>(x, X, N);
        }
    }
}

TEST_CASE("large-N round-trip (terminal DIF and deep Bluestein)", "[coverage][large]") {
    using T = double;
    for (std::size_t N : {std::size_t{8192}, std::size_t{16384}, std::size_t{32768},
                          std::size_t{65536}, std::size_t{8191}, std::size_t{65537},
                          std::size_t{16807}, std::size_t{880}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<T>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<T>>(spec), admiral::span(recov));
        require_close(in, recov, fft_tol<T>(4));

        require_parseval<T>(in, spec, N);
    }
}

TEST_CASE("Bluestein pad on the six-step delegate", "[coverage][large][bluestein]") {
    using T = double;
    using admiral::detail::bluestein_choose_pad;
    using admiral::detail::bluestein_inner_six_step_admits;
    using admiral::detail::bluestein_plan;
    REQUIRE(bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{524287})));
    REQUIRE(bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{401407})));
    REQUIRE(!bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{262143})));
    REQUIRE(!bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{393749})));

    for (std::size_t n : {std::size_t{524287}, std::size_t{401407}}) {
        INFO("n=" << n << " pad=" << bluestein_choose_pad(n));
        bluestein_plan<T> fwd(n, true), inv(n, false);
        const T one = T(1), invn = T(1) / T(n);

        {
            std::vector<std::complex<T>> x(n, {0, 0});
            x.at(0) = {1, 0};
            for (bool f : {true, false}) {
                std::vector<std::complex<T>> X(n);
                (f ? fwd : inv).execute(x.data(), X.data(), f ? one : invn);
                require_close(X, std::vector<std::complex<T>>(n, {f ? one : invn, 0}),
                              fft_tol<T>(64));
            }
        }
        for (bool f : {true, false}) {
            constexpr std::size_t k0 = 17;
            std::vector<std::complex<T>> x(n);
            for (std::size_t mm = 0; mm < n; ++mm) {
                x[mm] = unit_phasor<T>((f ? 1 : -1) * turn_fraction(k0, mm, n));
                if (f) x[mm] /= T(n);
            }
            std::vector<std::complex<T>> X(n), want(n, {0, 0});
            want.at(k0) = {1, 0};
            (f ? fwd : inv).execute(x.data(), X.data(), f ? one : invn);
            INFO("tone forward=" << f);
            require_close(X, want, fft_tol<T>(64));
        }
        const auto in = make_signal<T>(n);
        std::vector<std::complex<T>> spec(n), recov(n);
        fwd.execute(in.data(), spec.data(), one);
        inv.execute(spec.data(), recov.data(), invn);
        require_close(in, recov, fft_tol<T>(64));
        require_parseval<T>(in, spec, n, 64.0);
    }
}

TEST_CASE("Public large gate requires a cycle-free split", "[coverage][large]") {
    using admiral::detail::plan_impl;
    REQUIRE(std::string(plan_impl<double>(787500, true).route_name()) == "iterative_dif");
    REQUIRE(std::string(plan_impl<double>(802816, true).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<double>(787500, true, 16).route_name()) == "iterative_dif");
    REQUIRE(std::string(plan_impl<double>(810000, true, 16).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<float>(4194304, true).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<float>(1048576, true).route_name()) == "iterative_dif");

    for (std::size_t N : {std::size_t{787500}, std::size_t{802816}}) {
        INFO("N=" << N);
        const auto in = make_signal<double>(N);
        std::vector<std::complex<double>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<double>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<double>>(spec), admiral::span(recov));
        require_close(in, recov, fft_tol<double>(4));
        require_parseval<double>(in, spec, N);
    }
    {
        constexpr std::size_t N = 4194304;
        const auto in = make_signal<float>(N);
        std::vector<std::complex<float>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<float>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<float>>(spec), admiral::span(recov));
        require_close(in, recov, fft_tol<float>(4));
        require_parseval<float>(in, spec, N);
    }
}

TEST_CASE("f32 small-ido pass cover", "[coverage][small_ido]") {
    using T = float;
    for (std::size_t N : {std::size_t{13312}, std::size_t{33000}, std::size_t{33075}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<T>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<T>>(spec), admiral::span(recov));
        require_close(in, recov, fft_tol<T>(4));

        require_parseval<T>(in, spec, N);
    }
}

TEST_CASE("f32 batched four-step routed sizes", "[coverage][four_step]") {
    using T = float;
    for (std::size_t N : {std::size_t{128}, std::size_t{256}, std::size_t{384},
                          std::size_t{448}, std::size_t{512}, std::size_t{640},
                          std::size_t{768}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(admiral::span<const std::complex<T>>(in), admiral::span(spec));
        admiral::inverse(admiral::span<const std::complex<T>>(spec), admiral::span(recov));
        require_close(in, recov, fft_tol<T>(4));
        const auto ref = reference_dft<T>(in, true);
        require_close(spec, ref, fft_tol<T>());
    }
}

TEST_CASE("Rader prime with four-step inner convolution", "[coverage][rader]") {
    using namespace admiral::detail;
    const std::size_t p = 79;
    for (bool fwd : {true, false}) {
        INFO("forward=" << fwd);
        const auto in = make_signal<double>(p);
        rader_plan<double> plan(p, fwd);
        std::vector<std::complex<double>> out(p);
        plan.execute(in.data(), out.data());
        const auto ref = reference_dft<double>(in, fwd);
        require_close(out, ref, fft_tol<double>());
    }
}
