// Transform-correctness sweeps that reach every routing path. Ideas mirrored from
// the FFTW3, pocketfft and numpy.fft suites: exhaustive-N, prime-N/Bluestein,
// impulse/DC known-answers, Parseval.

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
#include <span>
#include <vector>

using namespace Catch::Matchers;

// Every route (codelet / iterative_dif / four_step / rader / bluestein /
// good_thomas) is hit somewhere in 2..512, the measured routing table's dense
// domain. Round-trip for all; naive-DFT cross-check on top, which masks nothing a
// round-trip would (sign, direction and twiddle-index bugs all surface).
TEMPLATE_TEST_CASE("exhaustive integer-N round-trip and forward vs naive DFT",
                   "[coverage][sweep]", float, double) {
    using T = TestType;
    for (std::size_t N = 2; N <= 512; ++N) {
        const auto in = make_signal<T>(N);

        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        INFO("round-trip N=" << N);
        require_close(in, recov, fft_tol<T>(2));

        const auto ref = reference_dft<T>(in, /*forward=*/true);
        INFO("vs-naive N=" << N);
        require_close(spec, ref, fft_tol<T>());
    }
}

// Impulse / DC known-answers + Parseval across routing boundaries. Catches
// off-by-one twiddle indexing that a round-trip hides. (FFTW3/numpy.)
TEMPLATE_TEST_CASE("known-answer impulse, DC, and Parseval", "[coverage][known]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 14> sizes = {1, 2, 3, 5, 7, 8, 12, 16, 60, 64, 67, 120, 128, 251};
    for (std::size_t N : sizes) {
        INFO("N=" << N);

        // Both known answers are exactly representable, so the reference carries no
        // error of its own and the budget is the library's alone: fft_tol, not a
        // hand-picked 1e-4 (which at f64 is 4.5e11 eps and cannot fail).

        // delta[0] -> flat spectrum (all bins == 1).
        {
            std::vector<std::complex<T>> x(N, std::complex<T>(0, 0));
            x.at(0) = std::complex<T>(1, 0);  // .at(): checked, so no -Wnull-dereference
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            require_close(X, std::vector<std::complex<T>>(N, std::complex<T>(1, 0)), fft_tol<T>());
        }

        // constant (DC) -> single non-zero bin at k=0 of value 2N.
        {
            std::vector<std::complex<T>> x(N, std::complex<T>(T(2), T(0)));
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            std::vector<std::complex<T>> want(N, std::complex<T>(0, 0));
            want.at(0) = std::complex<T>(T(2) * T(N), T(0));
            require_close(X, want, fft_tol<T>());
        }

        // Parseval: ||X||^2 == N * ||x||^2.
        {
            const auto x = make_signal<T>(N);
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            require_parseval<T>(x, X, N);
        }
    }
}

// Large N: terminal-base DIF path (bn 8/16/32/64) and deep Bluestein for a large
// prime, verified by round-trip + Parseval (naive DFT is infeasible).
TEST_CASE("large-N round-trip (terminal DIF and deep Bluestein)", "[coverage][large]") {
    using T = double;
    // Pow2 leaves every pass tail unreachable (ido % W == 0) and primes go to
    // Bluestein, so the mixed-radix sizes carry the tail coverage:
    //   16807 = 7^5      smallest N giving a non-in-place pass ido in [4, W)
    //                    (dif_pass_small_ido); below it the in-place prefix absorbs all.
    //   880 = 2^4*5*11   smallest N reaching dif_pass_body's scalar tail.
    // Both enumerated over N=2..60000 against the driver's own pass loop.
    for (std::size_t N : {std::size_t{8192}, std::size_t{16384}, std::size_t{32768},
                          std::size_t{65536}, std::size_t{8191}, std::size_t{65537},
                          std::size_t{16807}, std::size_t{880}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        require_close(in, recov, fft_tol<T>(4));

        require_parseval<T>(in, spec, N);
    }
}

// Bluestein at pads crossing the six-step delegate line (bluestein.hpp): the inner
// padded transform runs four_step_large there. 524287 pads to 2^20 (1024^2 fused);
// 401407 pads to 802816 = 896^2 (smooth, fused). Naive DFT is infeasible at these
// sizes, so the checks are known answers (impulse, tone: a sign flip moves the
// bin to N-k0), round-trip and Parseval. Tolerance x64: |DFT(chirp)| is a Gauss
// sum ~sqrt(2n), so intermediates carry sqrt(2n)x the input magnitude.
TEST_CASE("Bluestein pad on the six-step delegate", "[coverage][large][bluestein]") {
    using T = double;
    using admiral::detail::bluestein_choose_pad;
    using admiral::detail::bluestein_inner_six_step_admits;
    using admiral::detail::bluestein_plan;
    // Gate pinning: the two cells must cross and the two controls must not,
    // 262143 pads to 2^19 (below the byte line) and 393749 pads to 787500 = 875x900
    // (n2 % n1 != 0 -> the transpose-cycles path). Refusals are ISA-free; if the
    // gate moves, re-pick the cells.
    REQUIRE(bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{524287})));
    REQUIRE(bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{401407})));
    REQUIRE(!bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{262143})));
    REQUIRE(!bluestein_inner_six_step_admits<T>(bluestein_choose_pad(std::size_t{393749})));

    for (std::size_t n : {std::size_t{524287}, std::size_t{401407}}) {
        INFO("n=" << n << " pad=" << bluestein_choose_pad(n));
        bluestein_plan<T> fwd(n, true), inv(n, false);
        const T one = T(1), invn = T(1) / T(n);

        // Impulse -> flat (fwd: all ones; inv with 1/n: all 1/n).
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
        // Tone -> single bin at k0 (a sign/index flip moves it to N-k0). Forward
        // tone is pre-normalized (fct=1); inverse tone is raw with fct=1/n, so
        // both expect bin amplitude 1 (one normalization per case, not two).
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
        // Round-trip on the shared signal + Parseval on the forward shot.
        const auto in = make_signal<T>(n);
        std::vector<std::complex<T>> spec(n), recov(n);
        fwd.execute(in.data(), spec.data(), one);
        inv.execute(spec.data(), recov.data(), invn);
        require_close(in, recov, fft_tol<T>(64));
        require_parseval<T>(in, spec, n, 64.0);
    }
}

// Public large-route gate (plan.hpp large_route_admits): above the byte line,
// four_step_large additionally requires a cycle-free split at every nthreads
// (n2 % n1 != 0 sends both transposes to the serial element-cycle fallback), plus
// band-fused W alignment when serial. Pins are ISA-free: 787500 = 875x900 is
// cycle-shaped on any W; 802816 = 896^2 and 810000 = 900^2 divide evenly;
// 1048576 f32 sits below the f32 byte line, where the DIF chain still leads.
TEST_CASE("Public large gate requires a cycle-free split", "[coverage][large]") {
    using admiral::detail::plan_impl;
    REQUIRE(std::string(plan_impl<double>(787500, true).route_name()) == "iterative_dif");
    REQUIRE(std::string(plan_impl<double>(802816, true).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<double>(787500, true, 16).route_name()) == "iterative_dif");
    REQUIRE(std::string(plan_impl<double>(810000, true, 16).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<float>(4194304, true).route_name()) == "four_step_large");
    REQUIRE(std::string(plan_impl<float>(1048576, true).route_name()) == "iterative_dif");

    // Both directions on the flipped f64 cell and the kept fused control, plus
    // the newly admitted f32 cell end-to-end through the public API.
    for (std::size_t N : {std::size_t{787500}, std::size_t{802816}}) {
        INFO("N=" << N);
        const auto in = make_signal<double>(N);
        std::vector<std::complex<double>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<double>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<double>>(spec), std::span(recov));
        require_close(in, recov, fft_tol<double>(4));
        require_parseval<double>(in, spec, N);
    }
    {
        constexpr std::size_t N = 4194304;
        const auto in = make_signal<float>(N);
        std::vector<std::complex<float>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<float>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<float>>(spec), std::span(recov));
        require_close(in, recov, fft_tol<float>(4));
        require_parseval<float>(in, spec, N);
    }
}

// f32 small-ido cover: a non-in-place pass with 4 <= ido < W routes to
// dif_pass_small_ido's exact-width pieces. Reachable only above N=13312 for float;
// the sweep above stops at 512 where the in-place prefix absorbs these passes.
// ido is 13, 8 and 15 here: three different sized_cover decompositions.
TEST_CASE("f32 small-ido pass cover", "[coverage][small_ido]") {
    using T = float;
    for (std::size_t N : {std::size_t{13312}, std::size_t{33000}, std::size_t{33075}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        require_close(in, recov, fft_tol<T>(4));

        require_parseval<T>(in, spec, N);
    }
}

// f32 batched four-step: sizes 128..768 route to four_step_batched only at W=8;
// at W=16 they take iterative_dif. Round-trip + forward-vs-naive exercise both
// leaves of the split. (Covers four_step.hpp on the v3 build.)
TEST_CASE("f32 batched four-step routed sizes", "[coverage][four_step]") {
    using T = float;
    for (std::size_t N : {std::size_t{128}, std::size_t{256}, std::size_t{384},
                          std::size_t{448}, std::size_t{512}, std::size_t{640},
                          std::size_t{768}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        require_close(in, recov, fft_tol<T>(4));
        const auto ref = reference_dft<T>(in, /*forward=*/true);
        require_close(spec, ref, fft_tol<T>());
    }
}

// Rader with a four-step inner convolution: for p=79, L=p-1=78 is neither a power
// of two nor codelet-supported, so pick_inner routes the length-L convolution
// through four_step (the rader.hpp branch the routed sweep never hits). rader_plan
// is direction-fixed and unnormalized, matching reference_dft.
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
