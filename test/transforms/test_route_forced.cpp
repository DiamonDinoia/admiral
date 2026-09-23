
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/four_step_large.hpp>
#include <admiral/detail/nd_plan.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/rader.hpp>

#include "utils/reference.hpp"

#include <admiral/detail/granule_codelet.hpp>

#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(__has_include)
#if __has_include(<valgrind/valgrind.h>)
#include <valgrind/valgrind.h>
#define PROBE_SMOKE_HAVE_VGR 1  // the probe smoke case's wall bound widens under valgrind
#endif
#endif

static_assert(admiral::detail::choose_large_split(768).n1 == 24);
static_assert(admiral::detail::choose_large_split(768).n2 == 32);
static_assert(admiral::detail::choose_large_split(1 << 20).n1 == 1024);
static_assert(!admiral::detail::choose_large_split(17).valid());
static_assert(!admiral::detail::choose_large_split(3720).valid());

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
    // The router resolves the pool width the same way; injecting it here keeps the case a
    // pure consistency walk between the law and route election, host-independent.
    const std::size_t pool = resolve_nthreads(0);

    constexpr std::size_t kThreadCounts[] = {2, 4, 6, 16};
    for (const std::size_t threads : kThreadCounts) {
        const std::size_t line = large_route_threaded_bytes(elem, threads, pool);
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

TEMPLATE_TEST_CASE("route election survives an invalid large split under threads",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;

    // 3720's large split is invalid; measure/automatic at nthreads > 1 is the arm that once
    // took sp.n2 % sp.n1 on it.
    constexpr std::size_t n = 3720;
    static_assert(n > BASE_MODEL_NMAX);

    REQUIRE_NOTHROW(P(n, true, 2, nullptr, admiral::effort::measure));
    REQUIRE_NOTHROW(P(n, true, 2, nullptr, admiral::effort::automatic));
}

template<typename T>
static bool routes_large(std::size_t n, std::size_t threads) {
    using P = admiral::detail::plan_impl<T>;
    // std::string_view, because route_name() hands back a const char*: at -O0 the call lands in
    // the shared library and the two literals have different addresses, so == compares pointers.
    return std::string_view(P(n, true, threads, nullptr, admiral::effort::estimate).route_name())
           == "four_step_large";
}

TEST_CASE("threaded four_step_large admission matches the measured crossovers",
          "[coverage][route][four_step_large][threads]") {
    // The plateau is element-keyed, so the same element counts pin both precisions, at
    // every nt below the knees and at any pool width -- the plateau rungs are the only
    // route-level pins the law's pool anchoring allows a host-independent case to make.
    // The nt=2 f32 cell reads the cap outright (its bracket sits far above the plateau on
    // every class: [64K, 512K] elems, union geo-mid 1.41 MiB).
    for (const std::size_t nt : {std::size_t{2}, std::size_t{4}, std::size_t{8}}) {
        CAPTURE(nt);
        REQUIRE_FALSE(routes_large<double>(16384, nt));
        REQUIRE(routes_large<double>(32768, nt));
    }
    for (const std::size_t nt : {std::size_t{4}, std::size_t{8}}) {
        CAPTURE(nt);
        REQUIRE_FALSE(routes_large<float>(16384, nt));
        REQUIRE(routes_large<float>(32768, nt));
    }
    REQUIRE_FALSE(routes_large<float>(131072, 2));  // 1 MiB, below the nt=2 cap
    REQUIRE(routes_large<float>(262144, 2));        // 2 MiB, above it
}

TEST_CASE("the threaded large-route law's shape: floor, knees, rise to the shared cap",
          "[route][four_step_large][gate]") {
    using namespace admiral::detail;
    // Absolute values on purpose, same discipline as the serial pins: the law and the
    // measured grid it was read from (beat-standings/wi2c-large1d.md) are one artifact.
    REQUIRE(kLargeRouteThreadFloorF64Bytes == 370727);  // [256, 512] KiB geo-mid
    REQUIRE(kLargeRouteThreadFloorF32Bytes == 185363);  // [128, 256] KiB geo-mid
    REQUIRE(kLargeRouteThreadKneeF64Nt == 32);
    REQUIRE(kLargeRouteThreadKneeF32Nt == 8);
    REQUIRE(kLargeRouteThreadCapBytes == 1482910);      // [1, 2] MiB geo-mid

    // Flat at or below the knee at any pool width; nt=2 f32 is the cap outright.
    REQUIRE(large_route_threaded_bytes(16, 2, 128) == kLargeRouteThreadFloorF64Bytes);
    REQUIRE(large_route_threaded_bytes(16, 32, 64) == kLargeRouteThreadFloorF64Bytes);
    REQUIRE(large_route_threaded_bytes(8, 4, 96) == kLargeRouteThreadFloorF32Bytes);
    REQUIRE(large_route_threaded_bytes(8, 8, 16) == kLargeRouteThreadFloorF32Bytes);
    REQUIRE(large_route_threaded_bytes(8, 2, 128) == kLargeRouteThreadCapBytes);
    // Rise: linear in nt from the knee to the cap at nt = pool width, min-pinned; the
    // 2*knee span stand-in governs only when the pool is narrower than the knee.
    REQUIRE(large_route_threaded_bytes(16, 64, 128) == 741454);
    REQUIRE(large_route_threaded_bytes(16, 128, 128) == kLargeRouteThreadCapBytes);
    REQUIRE(large_route_threaded_bytes(8, 32, 64) == 741454);
    REQUIRE(large_route_threaded_bytes(8, 32, 96) == 539239);
    REQUIRE(large_route_threaded_bytes(8, 64, 64) == kLargeRouteThreadCapBytes);
    REQUIRE(large_route_threaded_bytes(16, 33, 16) == 405482);
    REQUIRE(large_route_threaded_bytes(16, 64, 16) == kLargeRouteThreadCapBytes);
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

TEST_CASE("serial four_step_large admission follows the injected line and the fallback",
          "[coverage][route][four_step_large]") {
    using namespace admiral::detail;
    constexpr std::size_t mib = std::size_t{1} << 20;
    // Absolute sizes on purpose. Deriving them from the constants would make the case pass
    // for any value of them, and the line and the A/B crossover it was read from are one
    // artifact: moving either without re-deriving the other is the defect this guards.
    {
        // Fallback pins (the shipped constants). The probe never runs while overridden,
        // so this case cannot see one host's measurement.
        const large_route_serial_override_scope pin(12 * mib, 16 * mib - 1);
        REQUIRE(large_route_serial_override(16) == 12 * mib);
        REQUIRE(large_route_serial_override(8) == 16 * mib - 1);
        REQUIRE_FALSE(routes_large<double>(262144, 1));   //  4 MiB, below the f64 fallback
        REQUIRE_FALSE(routes_large<double>(524288, 1));   //  8 MiB, likewise
        REQUIRE(routes_large<double>(2097152, 1));        // 32 MiB, above
        // f32's lower edge. The 32 MiB cap that used to close its window above is deleted
        // (the WI-2c sweep measured four_step winning at every rung past it on every
        // class), so there is no upper f32 line to pin against.
        REQUIRE_FALSE(routes_large<float>(1048576, 1));   //  8 MiB, below the f32 fallback
        REQUIRE(routes_large<float>(4194304, 1));         // 32 MiB, above the 16 MiB - 1
        REQUIRE(routes_large<float>(8388608, 1));         // 64 MiB, past the deleted cap
    }
    REQUIRE(large_route_serial_override(16) == 0);  // the scope cleared both
    REQUIRE(large_route_serial_override(8) == 0);
    {
        // The seam: injected lines move admission exactly, on both sides of a rung.
        // 8 MiB f64 is rejected under the fallback and admitted under 6 MiB -- the genoa
        // bracket [4, 8] MiB answered by injection, so the adaptivity is pinned without
        // keying any host.
        const large_route_serial_override_scope pin(6 * mib, 12 * mib);
        REQUIRE(routes_large<double>(524288, 1));
        REQUIRE(routes_large<float>(2097152, 1));  // 16 MiB under a 12 MiB injected line
    }
    {
        const large_route_serial_override_scope pin(20 * mib, 24 * mib);
        REQUIRE_FALSE(routes_large<double>(524288, 1));
        REQUIRE_FALSE(routes_large<float>(2097152, 1));  // and rejected again under 24 MiB
    }
    REQUIRE(large_route_serial_override(16) == 0);
    REQUIRE(large_route_serial_override(8) == 0);
}

// The memo's key must carry the serial-line override: without it a later scope at the same
// plan parameters replays the first scope's answer. Asserted on the memo's size, not a route
// name, because measure_route races and may legitimately disagree with estimate.
TEST_CASE("measured_route's memo keys on the override, not just the plan parameters",
          "[coverage][route][four_step_large][memo]") {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<double>;
    constexpr std::size_t mib = std::size_t{1} << 20;
    constexpr std::size_t n = 524288;  // 8 MiB f64, past BASE_MODEL_NMAX, valid large split.

    {
        const large_route_serial_override_scope pin(20 * mib, 24 * mib);
        const std::size_t m0 = P::measured_route_memo_size();
        const std::string first = P(n, true, 1, nullptr, admiral::effort::measure).route_name();
        REQUIRE(P::measured_route_memo_size() == m0 + 1);  // cold key: an election
        REQUIRE(P(n, true, 1, nullptr, admiral::effort::measure).route_name() == first);
        REQUIRE(P::measured_route_memo_size() == m0 + 1);  // same key: a hit
    }
    {
        const large_route_serial_override_scope pin(6 * mib, 12 * mib);
        const std::size_t m1 = P::measured_route_memo_size();
        [[maybe_unused]] const std::string second =
            P(n, true, 1, nullptr, admiral::effort::measure).route_name();
        REQUIRE(P::measured_route_memo_size() == m1 + 1);  // only the override moved
    }
}

TEST_CASE("large-route serial probe: side rule, geo-mid and ladder selection",
          "[route][four_step_large][gate]") {
    using namespace admiral::detail;
    constexpr std::size_t mib = std::size_t{1} << 20;
    // Absolute values on purpose, same discipline as the admission pins.
    REQUIRE(large_route_geo_mid(4 * mib, 8 * mib) == 5931641);    // bracket [4, 8] MiB
    REQUIRE(large_route_geo_mid(8 * mib, 16 * mib) == 11863283);  // [8, 16] MiB
    REQUIRE(large_route_geo_mid(16 * mib, 32 * mib) == 23726566); // [16, 32] MiB

    // The side rule: a tie (|ratio - 1| <= 5%) keeps the rung dif-side; only a decisive
    // four_step win moves it. icelake's 0.996/1.034 serial readings keep their rungs.
    REQUIRE(large_route_rung_dif_side(0.996));
    REQUIRE(large_route_rung_dif_side(1.034));
    REQUIRE(large_route_rung_dif_side(1.070));
    REQUIRE_FALSE(large_route_rung_dif_side(0.95));
    REQUIRE_FALSE(large_route_rung_dif_side(0.686));

    // The ladder constants are the priors the probe walks from; the envelope floors are
    // half the bottom rung (no probed line admits below it, so the router never probes).
    REQUIRE(kLargeRouteSerialF64Bytes == 12 * mib);
    REQUIRE(kLargeRouteSerialF32Bytes == 16 * mib - 1);
    REQUIRE(kLargeRouteProbeLadderF64[0] == 4 * mib);
    REQUIRE(kLargeRouteProbeLadderF64[kLargeRouteProbeLadderCount - 1] == 16 * mib);
    REQUIRE(kLargeRouteProbeLadderF32[0] == 8 * mib);
    REQUIRE(kLargeRouteProbeLadderF32[kLargeRouteProbeLadderCount - 1] == 32 * mib);
    REQUIRE(large_route_serial_min_bytes(16) == 2 * mib);
    REQUIRE(large_route_serial_min_bytes(8) == 4 * mib);
    REQUIRE(large_route_serial_fallback(16) == kLargeRouteSerialF64Bytes);
    REQUIRE(large_route_serial_fallback(8) == kLargeRouteSerialF32Bytes);

    // Selection over a walked segment: interior bracket, all-four_step bottom clamp,
    // all-dif 2x-top answer. These are the only three shapes the walk can produce.
    constexpr std::size_t lad[] = {4 * mib, 8 * mib, 16 * mib};
    const bool interior2[] = {true, false};   // dif keeps 4 MiB, four_step wins 8 MiB
    REQUIRE(large_route_serial_from_ladder(lad, interior2, 2) == 5931641);
    const bool interior3[] = {true, true, false};
    REQUIRE(large_route_serial_from_ladder(lad, interior3, 3) == 11863283);
    const bool all_fs[] = {false, false};     // four_step holds the walked segment
    REQUIRE(large_route_serial_from_ladder(lad, all_fs, 2) == 2 * mib);
    const bool all_dif[] = {true, true};      // dif holds the segment: 2x its top rung
    REQUIRE(large_route_serial_from_ladder(lad, all_dif, 2) == 16 * mib);
    // (a real walk's segment reaches a ladder edge, so 2x the segment top is 2x the
    // ladder top -- genoa f32's all-dif answer on the f32 ladder is 64 MiB)
    const bool all_dif_top[] = {true, true};
    REQUIRE(large_route_serial_from_ladder(lad + 1, all_dif_top, 2) == 32 * mib);
    // a one-rung segment (the walk's all-one-side stop) still answers
    const bool one_dif[] = {true};
    REQUIRE(large_route_serial_from_ladder(lad, one_dif, 1) == 8 * mib);
    const bool one_fs[] = {false};
    REQUIRE(large_route_serial_from_ladder(lad, one_fs, 1) == 2 * mib);
}

TEST_CASE("the serial large-route probe runs and answers inside its ladder envelope",
          "[route][four_step_large][probe]") {
    using namespace admiral::detail;
    using clock = std::chrono::steady_clock;
    // Smoke only: prove the machinery runs and answers with one of the ladder's reachable
    // values inside the design wall. The ladder's fitness against the banked sweep lives
    // in the receipt's simulation, not here.
    const auto a = clock::now();
    const std::size_t line64 = plan_impl<double>::probe_large_route_serial();
    const std::size_t line32 = plan_impl<float>::probe_large_route_serial();
    const double ms = std::chrono::duration<double, std::milli>(clock::now() - a).count();
    CAPTURE(line64, line32, ms);
    REQUIRE(line64 >= kLargeRouteProbeLadderF64[0] / 2);
    REQUIRE(line64 <= 2 * kLargeRouteProbeLadderF64[kLargeRouteProbeLadderCount - 1]);
    REQUIRE(line32 >= kLargeRouteProbeLadderF32[0] / 2);
    REQUIRE(line32 <= 2 * kLargeRouteProbeLadderF32[kLargeRouteProbeLadderCount - 1]);
    // The wall bound is evidence for the probe's cost class: 2 s covers the ~500 ms both
    // ladders pay on a loaded release host with 8x headroom; instrumented builds run the
    // same executes 10-20x slower (measured 3.8 s under asan+ubsan, 6.5 s under tsan) and
    // valgrind another 3x past that (8.6 s).
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define PROBE_SMOKE_SLOW 20000.0
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(undefined_behavior_sanitizer)
#define PROBE_SMOKE_SLOW 20000.0
#endif
#endif
#if defined(PROBE_SMOKE_HAVE_VGR) && !defined(PROBE_SMOKE_SLOW)
#define PROBE_SMOKE_SLOW (RUNNING_ON_VALGRIND ? 60000.0 : 2000.0)
#endif
#ifndef PROBE_SMOKE_SLOW
#define PROBE_SMOKE_SLOW 2000.0
#endif
    REQUIRE(ms < PROBE_SMOKE_SLOW);
#undef PROBE_SMOKE_SLOW
}

// ---------------------------------------------------------------------------
// Zero-timing gate pins for the N-D constants the cost model does not cover: the E2
// col-batch gate's L3 threshold (kE2Len64MinL3PerCoreBytes), the rank-2 fast path's
// admission terms, and the granule-reachability col-codelet predicate. Every cell
// feeds synthetic inputs, so no verdict moves with the probing host — this is the fix
// for the cohort-tuned-constant disease, where a pin read off host A's probe silently
// stops constraining on host B. Runtime REQUIREs, matching this TU's other constant
// pins: a compile-time assert would fail the build before naming the moved cell.

TEST_CASE("e2_len_cap_of answers from L3 per PHYSICAL core on synthetic topologies",
          "[route][nd][gate]") {
    using admiral::detail::cache_bytes;
    using admiral::detail::e2_len_cap_of;

    constexpr std::size_t mib = std::size_t{1} << 20;
    const auto cap = [](std::size_t l3, std::size_t phys) {
        return e2_len_cap_of(cache_bytes{/*l2=*/0, l3, /*l3_cores=*/0, phys, /*l1d=*/0});
    };

    // Absolute byte values on purpose, astride the fitted 2 MiB-per-physical-core line:
    // deriving the slices from kE2Len64MinL3PerCoreBytes would pass for any value of the
    // constant, and the threshold and the knob A/B it was read from are one artifact.
    struct row {
        std::size_t l3, phys, want;
    };
    for (const row r : {row{2 * mib - 1, 1, 32},   // closest slice below the line
                        row{2 * mib, 1, 64},       // on the line
                        row{2 * mib + 512, 1, 64}, // just above
                        row{4 * mib - 1, 2, 32},   // per-core 2 MiB - 1 over two cores
                        row{45 * mib, 16, 64},     // 2.8 MiB per physical core
                        row{45 * mib, 32, 32},     // same total, an SMT-flavored count
                        row{0, 8, 32},             // l3 unprobed
                        row{96 * mib, 0, 32}}) {   // divisor unprobed
        CAPTURE(r.l3, r.phys);
        REQUIRE(cap(r.l3, r.phys) == r.want);
    }
}

TEMPLATE_TEST_CASE("fast2d gate pins the shapes astride its admission terms",
                   "[route][nd][gate]", float, double) {
    using admiral::detail::is_codelet_catalog;
    using admiral::detail::nd_runtime_plan;

    // test_nd.cpp's bit-identity case already builds plans this cheap way in-test, and
    // uses_fast2d() IS the gate's evaluated terms, so the pin reads the flag rather
    // than duplicating the predicate.
    const auto fast2d = [](const auto& shape) {
        return nd_runtime_plan<TestType>(
                   admiral::span<const std::size_t>(shape.data(), shape.size()), true)
            .uses_fast2d();
    };
    using s2 = std::array<std::size_t, 2>;
    using s3 = std::array<std::size_t, 3>;

    // Fully admitted, every term true. Col extents <= 32 hold at both gate-cap values
    // (32 and 64), and 12/16 are catalogued in the fullest and in the sanitizer-trimmed
    // catalog alike, so these cells cannot move with host or build flavor.
    REQUIRE(fast2d(s2{16, 16}));
    REQUIRE(fast2d(s2{12, 16}));

    // One term flipped per shape:
    REQUIRE_FALSE(fast2d(s2{1, 16}));      // col extent 1
    REQUIRE_FALSE(fast2d(s2{16, 1}));      // row extent 1
    REQUIRE_FALSE(fast2d(s3{4, 16, 16}));  // rank 3
    // 96 is codelet-supported (11-smooth) but outside the catalog in every shipped
    // flavor: the row axis needs codelet_dispatch_many's catalog, not mere support.
    REQUIRE_FALSE(fast2d(s2{16, 96}));
    // The gate's serial term is unpinnable at plan level: every admitted shape is below
    // kThreadMinElems, so a thread request resolves back to pool == nullptr and the
    // gate sees a serial plan either way.
    if constexpr (is_codelet_catalog(32))  // trimmed catalog stops at 16
        REQUIRE(fast2d(s2{32, 32}));
    if constexpr (is_codelet_catalog(120))
        // Catalogued, 11-smooth, but past min(e2_len_cap(), kFourStepLeafMax) <= 64 at
        // any cap value, so the col axis can never take the arm the gate wants.
        REQUIRE_FALSE(fast2d(s2{120, 16}));
}

TEMPLATE_TEST_CASE("granule reachability pins the col-codelet predicate astride its terms",
                   "[route][nd][gate][granule]", float, double) {
    using admiral::detail::is_codelet_catalog;
    using admiral::detail::make_nd_axis_state;

    const auto col_codelet = [](std::size_t len, std::size_t inner) {
        return make_nd_axis_state<TestType>(len, inner, true, /*innermost=*/false)
            .col_codelet;
    };

    // The axis state a {12, 10, 9} plan builds for its outermost axis: len 12 is
    // catalogued, 11-smooth and under every e2 cap (12 <= 32), and ONLY
    // inner = 10*9 = 90 > 64 keeps the granule-reachable col-codelet arm off.
    REQUIRE_FALSE(col_codelet(12, 90));
    // The positive cell: {12, 12} puts the same length at inner 12 <= 64.
    REQUIRE(col_codelet(12, 12));
    // Astride the inner <= 64 term, one over the line in each direction.
    REQUIRE(col_codelet(12, 64));
    REQUIRE_FALSE(col_codelet(12, 65));
    if constexpr (is_codelet_catalog(120))
        // Catalogued and 11-smooth, but past min(e2_len_cap(), kFourStepLeafMax) <= 64
        // at any cap value: no host ever routes this length through the arm.
        REQUIRE_FALSE(col_codelet(120, 8));
}

TEMPLATE_TEST_CASE("fast3d gate pins the cube shapes astride its admission terms",
                   "[route][nd][gate][granule]", float, double) {
    using admiral::detail::is_codelet_catalog;
    using admiral::detail::kGranuleFmaddsub;
    using admiral::detail::nd_runtime_plan;

    // uses_fast3d() IS the gate's evaluated terms, so the pin reads the flag rather
    // than duplicating the predicate (the fast2d pin's own convention, above).
    const auto fast3d = [](const auto& shape) {
        return nd_runtime_plan<TestType>(
                   admiral::span<const std::size_t>(shape.data(), shape.size()), true)
            .uses_fast3d();
    };
    using s2 = std::array<std::size_t, 2>;
    using s3 = std::array<std::size_t, 3>;
    using s4 = std::array<std::size_t, 4>;

#ifndef ADM_GRANULE_ADMIT_OFF
    if constexpr (kGranuleFmaddsub && is_codelet_catalog(4) && is_codelet_catalog(8)) {
        // Fully admitted, every term true. 4 and 8 are in every catalog flavor (the
        // sanitizer trim stops at 16), so these cells cannot move with host or build.
        REQUIRE(fast3d(s3{4, 4, 4}));
        REQUIRE(fast3d(s3{8, 8, 8}));
    } else {
        REQUIRE_FALSE(fast3d(s3{4, 4, 4}));
        REQUIRE_FALSE(fast3d(s3{8, 8, 8}));
    }
#else
    // The ADM_GRANULE_ADMIT knob folds the cube predicate false with the rows/cols one.
    REQUIRE_FALSE(fast3d(s3{4, 4, 4}));
    REQUIRE_FALSE(fast3d(s3{8, 8, 8}));
#endif

    // One term flipped per shape:
    REQUIRE_FALSE(fast3d(s3{4, 4, 8}));      // extents unequal
    REQUIRE_FALSE(fast3d(s3{8, 8, 16}));
    REQUIRE_FALSE(fast3d(s2{8, 8}));         // rank 2
    REQUIRE_FALSE(fast3d(s4{4, 4, 4, 4}));   // rank 4
    REQUIRE_FALSE(fast3d(s3{12, 12, 12}));   // equal extents but outside the cube set
    // The gate's serial term is unpinnable at plan level, for fast2d's reason: the
    // admitted cubes are below kThreadMinElems, so a thread request resolves back to
    // pool == nullptr and the gate sees a serial plan either way.
}

TEMPLATE_TEST_CASE("fast3d cube execution matches the separable 3-D reference",
                   "[route][nd][granule]", float, double) {
    if constexpr (!admiral::detail::kGranuleFmaddsub) {
        SKIP("granule dialect compiled out at this ISA");
    }
#ifdef ADM_GRANULE_ADMIT_OFF
    SKIP("granule admission compiled out (ADM_GRANULE_ADMIT=OFF)");
#endif
    using T = TestType;
    unsigned seed = 61;
    for (const std::size_t n : {std::size_t{4}, std::size_t{8}}) {
        const std::size_t n3 = n * n * n;
        const std::array<std::size_t, 3> shape{n, n, n};
        for (const bool forward : {true, false}) {
            CAPTURE(n, forward);
            const auto in = make_input<T>(n3, seed++);
            admiral::detail::nd_runtime_plan<T> p(
                admiral::span<const std::size_t>(shape.data(), shape.size()), forward);
            REQUIRE(p.uses_fast3d());

            // The seat folds the whole factor into one pass; mirror by scaling the
            // unnormalized reference once. The default inverse is 1/N^3.
            const auto ref0 = reference_cube3d<T, long double>(in, n, forward);
            std::vector<std::complex<long double>> ref(n3);
            const long double def = forward ? 1.0L : 1.0L / static_cast<long double>(n3);
            for (std::size_t i = 0; i < n3; ++i) ref[i] = ref0[i] * def;

            auto a = in;
            p.execute(a.data());
            require_close_pointwise(a, ref);

            std::vector<std::complex<T>> b(n3);
            p.execute(in.data(), b.data());
            require_close_pointwise(b, ref);

            // A custom factor rides the same single fold on both plan kinds.
            admiral::detail::exec_options<T> opts{};
            opts.fct = T(2);
            std::vector<std::complex<T>> c(n3);
            p.execute(in.data(), c.data(), opts);
            std::vector<std::complex<long double>> refc(n3);
            for (std::size_t i = 0; i < n3; ++i) refc[i] = ref0[i] * 2.0L;
            require_close_pointwise(c, refc);
        }
    }
}
