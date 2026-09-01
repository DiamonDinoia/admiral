
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/four_step_large.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/rader.hpp>

#include "utils/reference.hpp"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static_assert(admiral::detail::choose_large_split(768).n1 == 24);
static_assert(admiral::detail::choose_large_split(768).n2 == 32);
static_assert(admiral::detail::choose_large_split(1 << 20).n1 == 1024);
static_assert(!admiral::detail::choose_large_split(17).valid());

TEMPLATE_TEST_CASE("forced route names itself and round-trips", "[coverage][route]",
                   float, double) {
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    const auto pick = [](R r) {
        std::size_t n = 0;
        for (std::size_t k = 2; k <= 768; ++k)
            if (P::route_available(r, k)) n = k;
        return n;
    };
    const std::pair<R, const char*> routes[] = {
        {R::codelet,           "codelet"},
        {R::iterative_dif,     "iterative_dif"},
        {R::four_step,         "four_step"},
        {R::four_step_batched, "four_step_batched"},
        {R::four_step_large,   "four_step_large"},
        {R::rader,             "rader"},
        {R::bluestein,         "bluestein"},
        {R::good_thomas,       "good_thomas"},
    };

    std::size_t exercised = 0;

    for (const auto& [route, name] : routes) {
        const std::size_t n = pick(route);
        CAPTURE(name, n);
        if (n == 0) {
            REQUIRE_THROWS_AS(P(60u, true, route), admiral::unsupported_error);
            continue;
        }
        ++exercised;
        const P fwd(n, true, route);
        const P inv(n, false, route);
        REQUIRE(std::string(fwd.route_name()) == name);
        REQUIRE(fwd.size() == n);
        REQUIRE(inv.size() == n);
        REQUIRE(fwd.is_forward());
        REQUIRE_FALSE(inv.is_forward());

        const auto split = fwd.four_step_split_used();
        REQUIRE(split.valid() == (route == R::four_step));
        if (split.valid()) REQUIRE(split.n1 * split.n2 == n);

        const auto x = make_signal<TestType>(n);
        auto got = x;
        fwd.execute(admiral::span(got));
        inv.execute(admiral::span(got));
        require_close(x, got, fft_tol<TestType>(2));
    }

    REQUIRE(exercised >= 3);

    REQUIRE_FALSE(P::route_available(R::good_thomas, 17u));
    REQUIRE_THROWS_AS(P(17u, true, R::good_thomas), admiral::unsupported_error);
    REQUIRE_THROWS_AS(P(0u, true, R::bluestein), admiral::size_error);
}

TEMPLATE_TEST_CASE("forced four_step_large round-trips at every admitted size",
                   "[coverage][route][four_step_large]", float, double) {
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    std::size_t checked = 0;
    for (std::size_t n = 2; n <= 1024; ++n) {
        if (!P::route_available(R::four_step_large, n)) continue;
        CAPTURE(n);
        const auto x = make_signal<TestType>(n);
        auto got = x;
        P(n, true, R::four_step_large).execute(admiral::span(got));
        P(n, false, R::four_step_large).execute(admiral::span(got));
        require_close(x, got, fft_tol<TestType>(2));
        ++checked;
    }
    REQUIRE(checked > 50);
}

TEMPLATE_TEST_CASE("threaded four_step_large round-trips forced and raced",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    std::size_t n = 0;
    for (std::size_t k = BASE_MODEL_NMAX + 1; k <= 4096 && n == 0; ++k) {
        if (!P::route_available(R::four_step_large, k)) continue;
        if (choose_large_split(k).n2 % choose_large_split(k).n1 == 0) n = k;
    }
    REQUIRE(n != 0);
    CAPTURE(n);

    constexpr std::size_t threads = 3;
    const auto x = make_signal<TestType>(n);

    auto got = x;
    P(n, true, R::four_step_large, threads).execute(admiral::span(got));
    P(n, false, R::four_step_large, threads).execute(admiral::span(got));
    require_close(x, got, fft_tol<TestType>(2));

    got = x;
    P(n, true, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    P(n, false, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    require_close(x, got, fft_tol<TestType>(2));
}

TEMPLATE_TEST_CASE("raced route round-trips where the gate admits four_step_large",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;

    constexpr std::size_t threads = 3;
    std::size_t n = 0;
    for (std::size_t k = 1; k <= (std::size_t{1} << 19) && n == 0; k <<= 1)
        if (k > BASE_MODEL_NMAX && P(k, true, threads, nullptr, admiral::effort::estimate)
                                           .route_name() == std::string("four_step_large"))
            n = k;
    REQUIRE(n != 0);
    CAPTURE(n);

    const auto x = make_signal<TestType>(n);
    auto got = x;
    P(n, true, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    P(n, false, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    require_close(x, got, fft_tol<TestType>(2));
}

TEMPLATE_TEST_CASE("threaded four_step_large admission scales with nthreads",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    constexpr std::size_t elem = sizeof(std::complex<TestType>);

    constexpr std::size_t kThreadCounts[] = {2, 4, 6, 16};
    for (const std::size_t threads : kThreadCounts) {
        const std::size_t line = large_route_threaded_bytes(threads);
        std::size_t below = 0, above = 0;
        for (std::size_t n = 1024; n <= (std::size_t{1} << 22); n *= 2) {
            if (!P::route_available(R::four_step_large, n)) continue;
            if (n * elem <= line) below = n;
            else if (above == 0) above = n;
        }
        CAPTURE(threads, line, below, above);
        REQUIRE(below != 0);
        REQUIRE(above != 0);

        constexpr admiral::effort est = admiral::effort::estimate;
        const std::string lo = P(below, true, threads, nullptr, est).route_name();
        const std::string hi = P(above, true, threads, nullptr, est).route_name();
        REQUIRE(lo != "four_step_large");
        REQUIRE(hi == "four_step_large");
    }
}

template<typename T>
static bool routes_large(std::size_t n, std::size_t threads) {
    using P = admiral::detail::plan_impl<T>;
    return P(n, true, threads, nullptr, admiral::effort::estimate).route_name()
           == "four_step_large";
}

TEST_CASE("threaded four_step_large admission matches the measured crossovers",
          "[coverage][route][four_step_large][threads]") {
    struct row {
        std::size_t threads, admit, reject;
    };
    for (const row r : {row{2, 65536, 32768}, row{4, 32768, 16384}, row{6, 32768, 16384}}) {
        CAPTURE(r.threads, r.admit, r.reject);
        REQUIRE(routes_large<double>(r.admit, r.threads));
        REQUIRE_FALSE(routes_large<double>(r.reject, r.threads));
    }
    REQUIRE(routes_large<float>(65536, 4));
    REQUIRE_FALSE(routes_large<float>(32768, 4));

    using admiral::detail::large_route_threaded_bytes;
    REQUIRE(large_route_threaded_bytes(4) < large_route_threaded_bytes(2));
    REQUIRE(large_route_threaded_bytes(16) == large_route_threaded_bytes(4));
}

TEST_CASE("serial four_step_large admission is bounded above for f32 only",
          "[coverage][route][four_step_large]") {
    REQUIRE(routes_large<float>(4194304, 1));
    REQUIRE_FALSE(routes_large<float>(8388608, 1));
    REQUIRE(routes_large<double>(8388608, 1));
    REQUIRE(routes_large<float>(8388608, 6));
}

TEMPLATE_TEST_CASE("plan rejects bad sizes and handles N==1", "[coverage][route]",
                   float, double) {
    using P = admiral::detail::plan_impl<TestType>;

    REQUIRE_THROWS_AS(P(0u, true), admiral::size_error);
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::estimate), admiral::size_error);
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::automatic), admiral::size_error);
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::measure), admiral::size_error);

    const admiral::detail::dif_factor_plan chain;
    REQUIRE_THROWS_AS(P(13u, true, 1u, &chain), admiral::unsupported_error);

    const P fwd(1u, true), inv(1u, false);
    std::vector<std::complex<TestType>> one{{TestType(3), TestType(-1)}};
    std::vector<std::complex<TestType>> two(2);
    REQUIRE_THROWS_AS(fwd.execute(admiral::span(two)), admiral::size_error);
    fwd.execute(admiral::span(one));
    REQUIRE(one[0] == std::complex<TestType>(TestType(3), TestType(-1)));
    inv.execute(admiral::span(one));
    REQUIRE(one[0] == std::complex<TestType>(TestType(3), TestType(-1)));

    constexpr std::size_t rows = 4, stride = 3;
    std::vector<std::complex<TestType>> run(rows * stride, {TestType(0), TestType(0)});
    for (std::size_t r = 0; r < rows; ++r) run[r * stride] = {TestType(r + 1), TestType(0)};
    fwd.execute_many(run.data(), rows, stride, {TestType(2)});
    for (std::size_t r = 0; r < rows; ++r) {
        REQUIRE(run[r * stride].real() == TestType(2 * (r + 1)));
        REQUIRE(run[r * stride + 1] == std::complex<TestType>(TestType(0), TestType(0)));
    }
}

TEST_CASE("estimated_plan_cost takes each modeled route", "[coverage][route]") {
    using namespace admiral::detail;

    REQUIRE(estimated_plan_cost(0) == 0.0);
    REQUIRE(estimated_plan_cost(1) == 0.0);

    std::size_t n_fs = 0;
    for (std::size_t N = 65; N < 4096 && !n_fs; ++N) {
        if (admiral::detail::has_single_bit(N) || is_codelet_supported(N)) continue;
        const four_step_split s = choose_four_step_split(N);
        if (s.valid() && gate_four_step_cost(s.n1, s.n2) < bluestein_model_cost(N)) n_fs = N;
    }
    REQUIRE(n_fs != 0);
    {
        const four_step_split s = choose_four_step_split(n_fs);
        REQUIRE(estimated_plan_cost(n_fs) == gate_four_step_cost(s.n1, s.n2));
    }

    std::size_t p_rader = 0;
    for (std::size_t p = 65; p < 4096 && !p_rader; ++p)
        if (rader_supported(p)) p_rader = p;
    REQUIRE(p_rader != 0);
    REQUIRE(estimated_plan_cost(p_rader)
            == 2.0 * estimated_plan_cost(p_rader - 1) + 17.0 * double(p_rader));

    std::size_t p_blue = 0;
    for (std::size_t p = 65; p < 100000 && !p_blue; ++p) {
        if (admiral::detail::has_single_bit(p) || is_codelet_supported(p)) continue;
        if (choose_four_step_split(p).valid()) continue;
        if (rader_supported(p)) continue;
        p_blue = p;
    }
    REQUIRE(p_blue != 0);
    REQUIRE(estimated_plan_cost(p_blue) == bluestein_model_cost(p_blue));
}

TEMPLATE_TEST_CASE("Rader with a codelet inner transform", "[coverage][route][catalog]",
                   float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    std::size_t p = 0;
    for (const std::size_t L : CODELET_CATALOG_SIZES)
        if (L > 64 && rader_supported(L + 1)) p = L + 1;
    if (p == 0)
        SKIP("no catalog size L > 64 with L+1 prime, so rader_inner_kind::codelet is "
             "unreachable here; set ADM_CODELET_EXTRA_SIZES to cover it");

    CAPTURE(p);
    REQUIRE(is_codelet_catalog(p - 1));

    const auto x = make_signal<TestType>(p);
    for (const bool forward : {true, false}) {
        const P pl(p, forward, R::rader);
        REQUIRE(std::string(pl.route_name()) == "rader");
        auto got = x;
        pl.execute(admiral::span(got));
        auto ref = reference_dft(x, forward);
        if (!forward)
            for (auto& v : ref) v /= TestType(p);
        require_close(got, ref, fft_tol<TestType>(4.0));
    }
}

TEMPLATE_TEST_CASE("forced Bluestein with a codelet-catalog pad", "[coverage][route]",
                   float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    constexpr std::size_t n = 17;
    if (!is_codelet_catalog(admiral::detail::bit_ceil(2 * n - 1)))
        SKIP("pad 64 is outside this build's codelet catalog (sanitizer cap)");
    REQUIRE(P::route_available(R::bluestein, n));
    for (const bool forward : {true, false}) {
        const P pl(n, forward, R::bluestein);
        REQUIRE(std::string(pl.route_name()) == "bluestein");
        auto x = make_signal<TestType>(n);
        auto got = x;
        pl.execute(admiral::span(got));
        auto ref = reference_dft(x, forward);
        if (!forward)
            for (auto& v : ref) v /= TestType(n);
        require_close(got, ref, fft_tol<TestType>());
    }
}
