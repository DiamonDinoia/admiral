// Forced routes and the selecting ctor's own guards.
//
// `plan_impl` carries a documented test-only force-route ctor. Without that ctor the
// `route_name` switch, the diagnostics accessors and the unavailable-route rejection
// stay unexercised: an enum added without a name mapping returns "?" unnoticed.
// Forcing also pins each route to a round-trip at a size the cost model may never
// send, which selector-driven tests cannot do by construction.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/four_step_large.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/rader.hpp>  // estimated_plan_cost and the route cost model

#include "utils/reference.hpp"

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// `choose_large_split` is `constexpr`, so its contract is a build-time assertion.
// Both leaves must be usable lengths and multiply back to N exactly.
static_assert(admiral::detail::choose_large_split(768).n1 == 24);
static_assert(admiral::detail::choose_large_split(768).n2 == 32);
static_assert(admiral::detail::choose_large_split(1 << 20).n1 == 1024);
static_assert(!admiral::detail::choose_large_split(17).valid());   // prime: no split

TEMPLATE_TEST_CASE("forced route names itself and round-trips", "[coverage][route]",
                   float, double) {
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    // No hardcoded sizes: availability is ISA- and precision-dependent, so any
    // literal is wrong on some build (`four_step_batched` is a W==8 band capped at
    // `FSB_MAX_N`=768). Ask for the largest size each route admits.
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

    // Route availability is NOT fixed across builds, so this asserts an invariant,
    // not a list. The `four_step_batched` split table is f32/W==8 only, and
    // `good_thomas` gates on the register-fit rule
    // 2*ceil(N/W) + 2*max(Ni) + 4 <= vector_register_count(): at SSE2, f64 W=2,
    // it admits nothing. Both routes are legitimately empty on some build.
    std::size_t exercised = 0;

    for (const auto& [route, name] : routes) {
        const std::size_t n = pick(route);
        CAPTURE(name, n);
        if (n == 0) {
            // If a route is unavailable across the range, `route_available` must
            // agree with the ctor. That agreement is the property that matters.
            REQUIRE_THROWS_AS(P(60u, true, route), admiral::unsupported_error);
            continue;
        }
        ++exercised;
        const P fwd(n, true, route);
        const P inv(n, false, route);
        REQUIRE(std::string(fwd.route_name()) == name);  // never "?"
        REQUIRE(fwd.size() == n);
        REQUIRE(inv.size() == n);
        REQUIRE(fwd.is_forward());
        REQUIRE_FALSE(inv.is_forward());

        // Only the plain four-step state reports a split, and a reported split must
        // factor n exactly.
        const auto split = fwd.four_step_split_used();
        REQUIRE(split.valid() == (route == R::four_step));
        if (split.valid()) REQUIRE(split.n1 * split.n2 == n);

        const auto x = make_signal<TestType>(n);
        auto got = x;
        fwd.execute(admiral::span(got));   // default fct: forward 1, inverse 1/n
        inv.execute(admiral::span(got));
        require_close(x, got, fft_tol<TestType>(2));
    }

    // `codelet`, `iterative_dif` and `bluestein` gate on size alone. A build where
    // fewer than three routes round-trip has lost a route, not an ISA.
    REQUIRE(exercised >= 3);

    // Documented contract: the ctor rejects a route that cannot serve (T, n); the
    // ctor never swaps in a route that works silently. 17 is prime at every ISA,
    // so 17 has no coprime Good-Thomas factorization.
    REQUIRE_FALSE(P::route_available(R::good_thomas, 17u));
    REQUIRE_THROWS_AS(P(17u, true, R::good_thomas), admiral::unsupported_error);
    REQUIRE_THROWS_AS(P(0u, true, R::bluestein), admiral::size_error);
}

// Every size a forced `four_step_large` admits must round-trip. A literal sweep also
// covers sizes `pick()` never selects. The route gate is a byte threshold the force
// path ignores, and the first column pass changes shape on a single-radix leaf chain.
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
    REQUIRE(checked > 50);   // the split table admits hundreds of sizes below 1024
}

// The forced ctor takes `nthreads` because the `four_step_large` executor is the only
// reader of `m.pool`. A trial built without a pool times the SERIAL route under the
// threaded route's name.
TEMPLATE_TEST_CASE("threaded four_step_large round-trips forced and raced",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    // Searched, never hardcoded: above `BASE_MODEL_NMAX`, so the race reaches the
    // offer, and pool-safe (n2 % n1 == 0, the clause `large_route_admits` applies to
    // threaded splits).
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

    // Whichever route the race elects has to be correct, pool or no pool.
    got = x;
    P(n, true, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    P(n, false, threads, nullptr, admiral::effort::automatic).execute(admiral::span(got));
    require_close(x, got, fft_tol<TestType>(2));
}

// The reverse direction of the offer. Where the gate ADMITS `four_step_large`, the race
// prices `iterative_dif` against it. The race then builds a DIF plan that carries a pool
// at a large threaded size. The search above stops at the smallest pool-safe size, which
// the gate rejects.
TEMPLATE_TEST_CASE("raced route round-trips where the gate admits four_step_large",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;

    constexpr std::size_t threads = 3;
    // Searched, not hardcoded: the line moves with `nthreads` and with `sizeof(T)`.
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

// The threaded admission line falls with the thread count down to its floor, so the
// crossover moves. The pin at `estimate` stops a future edit from reintroducing a fixed
// line, which would fit one machine and strand the others.
TEMPLATE_TEST_CASE("threaded four_step_large admission scales with nthreads",
                   "[coverage][route][four_step_large][threads]", float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    constexpr std::size_t elem = sizeof(std::complex<TestType>);

    // Straddle the line at each measured thread count instead of hardcoding a size.
    // The pair is the largest power of two at or below the threshold and the next
    // one up.
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

        // `estimate` has no race, so the byte line alone decides, and the route name
        // is the gate's own output.
        constexpr admiral::effort est = admiral::effort::estimate;
        const std::string lo = P(below, true, threads, nullptr, est).route_name();
        const std::string hi = P(above, true, threads, nullptr, est).route_name();
        REQUIRE(lo != "four_step_large");
        REQUIRE(hi == "four_step_large");
    }
}

// Measured crossovers as literals. The scaling test above derives its straddle pair
// from `large_route_threaded_bytes`, so that test pins the gate against the helper,
// not against the machine. These literals fail when the law reverts to a fixed
// budget/nthreads.
template<typename T>
static bool routes_large(std::size_t n, std::size_t threads) {
    using P = admiral::detail::plan_impl<T>;
    return P(n, true, threads, nullptr, admiral::effort::estimate).route_name()
           == "four_step_large";
}

TEST_CASE("threaded four_step_large admission matches the measured crossovers",
          "[coverage][route][four_step_large][threads]") {
    // {threads, admitted size, rejected size}: the measured crossover sits between them.
    // Only cells the fitted law reproduces appear here.
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

    // The line falls with the thread count until the floor binds. A constant helper
    // satisfies the straddle test above but not this one.
    using admiral::detail::large_route_threaded_bytes;
    REQUIRE(large_route_threaded_bytes(4) < large_route_threaded_bytes(2));
    REQUIRE(large_route_threaded_bytes(16) == large_route_threaded_bytes(4));
}

// Serial f32 admission is a window: the DIF chain wins again above 32 MiB, while
// serial f64 keeps winning. Threading removes the upper crossover, so only the serial
// arm bounds the window.
TEST_CASE("serial four_step_large admission is bounded above for f32 only",
          "[coverage][route][four_step_large]") {
    REQUIRE(routes_large<float>(4194304, 1));         // 32 MiB, at the line
    REQUIRE_FALSE(routes_large<float>(8388608, 1));   // 64 MiB, past it
    REQUIRE(routes_large<double>(8388608, 1));        // 128 MiB f64, no upper bound
    REQUIRE(routes_large<float>(8388608, 6));         // threaded f32 keeps winning
}

// The selecting ctor's own guards and the N==1 degenerate transform. Distinct
// from the force-route ctor above: that one is a separate overload, so its
// size==0 rejection does not exercise this one's.
TEMPLATE_TEST_CASE("plan rejects bad sizes and handles N==1", "[coverage][route]",
                   float, double) {
    using P = admiral::detail::plan_impl<TestType>;

    REQUIRE_THROWS_AS(P(0u, true), admiral::size_error);
    // Every effort, the default included. The delegating ctor resolves the route in its
    // argument before the guard below runs, so each effort reaches the chain DP on its
    // own path. Size 0 must throw, not hang: `is_pentanomial(0)` halves 0 forever.
    // A throws-check catches a hang through the test's timeout.
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::estimate), admiral::size_error);
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::automatic), admiral::size_error);
    REQUIRE_THROWS_AS(P(0u, true, 1u, nullptr, admiral::effort::measure), admiral::size_error);

    // A `dif_factor_plan` override can represent an 11-smooth N only: every radix in
    // `dif_radix_set` is 11-smooth. Forcing 13 must throw; a silent reroute would be
    // timed as a valid A/B baseline.
    const admiral::detail::dif_factor_plan chain;
    REQUIRE_THROWS_AS(P(13u, true, 1u, &chain), admiral::unsupported_error);

    // N==1: the DFT is the identity times `fct`, and both `execute` overloads
    // shortcut before any kernel. `execute_many` also applies `fct` down a
    // strided run.
    const P fwd(1u, true), inv(1u, false);
    std::vector<std::complex<TestType>> one{{TestType(3), TestType(-1)}};
    std::vector<std::complex<TestType>> two(2);
    REQUIRE_THROWS_AS(fwd.execute(admiral::span(two)), admiral::size_error);  // size mismatch
    fwd.execute(admiral::span(one));
    REQUIRE(one[0] == std::complex<TestType>(TestType(3), TestType(-1)));
    inv.execute(admiral::span(one));  // inverse fct = 1/N = 1
    REQUIRE(one[0] == std::complex<TestType>(TestType(3), TestType(-1)));

    constexpr std::size_t rows = 4, stride = 3;
    std::vector<std::complex<TestType>> run(rows * stride, {TestType(0), TestType(0)});
    for (std::size_t r = 0; r < rows; ++r) run[r * stride] = {TestType(r + 1), TestType(0)};
    fwd.execute_many(run.data(), rows, stride, {TestType(2)});
    for (std::size_t r = 0; r < rows; ++r) {
        REQUIRE(run[r * stride].real() == TestType(2 * (r + 1)));
        REQUIRE(run[r * stride + 1] == std::complex<TestType>(TestType(0), TestType(0)));  // untouched
    }
}

// `estimated_plan_cost` tail coverage: the four-step, Rader and Bluestein returns.
//
// The Rader-vs-Bluestein gate (`rader.hpp`) is the only caller of the Rader tail,
// always with N = p-1 for a prime p. The early returns absorb every such N whose
// p-1 is a power of two or codelet-supported. The cost model is pure, so the test
// drives it directly.
//
// Each branch's witness is SEARCHED with the model's own public predicates, never
// hardcoded. A magic size silently stops exercising its branch when the catalog or
// the split table moves; a failed search fails the test.
TEST_CASE("estimated_plan_cost takes each modeled route", "[coverage][route]") {
    using namespace admiral::detail;

    REQUIRE(estimated_plan_cost(0) == 0.0);
    REQUIRE(estimated_plan_cost(1) == 0.0);

    // (a) four-step return: composite, not pow2, not codelet-supported, with a valid
    // split the model prefers to Bluestein.
    std::size_t n_fs = 0;
    for (std::size_t N = 65; N < 4096 && !n_fs; ++N) {
        if (admiral::detail::has_single_bit(N) || is_codelet_supported(N)) continue;
        const four_step_split s = choose_four_step_split(N);
        if (s.valid() && gate_four_step_cost(s.n1, s.n2) < bluestein_model_cost(N)) n_fs = N;
    }
    REQUIRE(n_fs != 0);  // without a witness, the four-step return is unreachable
    {
        const four_step_split s = choose_four_step_split(n_fs);
        REQUIRE(estimated_plan_cost(n_fs) == gate_four_step_cost(s.n1, s.n2));
    }

    // (b) Rader return, including the recursion into the inner transform. A prime
    // has no valid split, so it always falls past the four-step branch.
    std::size_t p_rader = 0;
    for (std::size_t p = 65; p < 4096 && !p_rader; ++p)
        if (rader_supported(p)) p_rader = p;
    REQUIRE(p_rader != 0);
    REQUIRE(estimated_plan_cost(p_rader)
            == 2.0 * estimated_plan_cost(p_rader - 1) + 17.0 * double(p_rader));

    // (c) Bluestein fallback: a size no earlier branch claims, so not pow2, not
    // codelet-supported, no valid split, not Rader-eligible. The first such size is
    // composite (134 = 2*67), not prime: Rader declines 134 for failing primality,
    // and the split table has no entry for 134.
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

// Rader's codelet inner kind is gated on the CATALOG, not on the size:
// `rader_supported` rejects exactly the primes the catalog covers, and `pick_inner`
// picks `codelet` only when p-1 is itself a catalog member. A witness needs a
// catalog size L with L+1 prime beyond the contiguous range. The default catalog's
// only extra is 120 (121 = 11^2), so no witness exists by default. Add one with
// -DADM_CODELET_EXTRA_SIZES="120;66" (66 -> p=67), as the `catalog` arm of
// `validate.sh` does.
//
// The witness is SEARCHED, never hardcoded, and on an empty search the case SKIPs.
// A quiet success on an empty search would leave the configuration uncovered.
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
    REQUIRE(is_codelet_catalog(p - 1));  // exactly the condition `pick_inner` tests

    // Both directions: the inner codelet call is shared, but only the inverse also
    // takes the `scale_inplace` line beside the call.
    const auto x = make_signal<TestType>(p);
    for (const bool forward : {true, false}) {
        const P pl(p, forward, R::rader);
        REQUIRE(std::string(pl.route_name()) == "rader");
        auto got = x;
        pl.execute(admiral::span(got));
        auto ref = reference_dft(x, forward);
        if (!forward)
            for (auto& v : ref) v /= TestType(p);   // `execute`'s default inverse `fct` is 1/n
        // The suite's flat 32-eps discipline, scaled x4 (128 eps) for the p-1 inner
        // chain. An N-scaled bound would allow ~2144 eps here.
        require_close(got, ref, fft_tol<TestType>(4.0));
    }
}

TEMPLATE_TEST_CASE("forced Bluestein with a codelet-catalog pad", "[coverage][route]",
                   float, double) {
    using namespace admiral::detail;
    using P = admiral::detail::plan_impl<TestType>;
    using R = typename P::route_kind;

    // If `is_codelet_catalog(bit_ceil(2N-1))` holds, the padded inner transform runs
    // a catalog codelet. N=17 gives pad 64: compiled in a default build, but outside
    // the sanitizer cap [2..16]. Skip with the reason.
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
