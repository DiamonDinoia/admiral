// The routing cost model's structural helpers, against independent oracles.
//
// base_cost_model.hpp decides every route for 2 <= N <= 512, and it scores a plan with
// lpf_nfac, balanced_split and chain_work from math.hpp — the same definitions
// tools/fit_cost_model.cpp fits the coefficients with. A single shared definition can
// still be uniformly wrong, which sharing cannot catch.
//
// So the oracles here are deliberately the OTHER traversal: chain_work ships as a DP
// over the divisors of n; the oracle is the plain recursion. Agreement between a DP
// and a recursion is a real check; a second copy of the DP is not.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/base_cost_model.hpp>
#include <admiral/detail/math.hpp>

#include <cstddef>
#include <map>
#include <utility>

namespace {

// Every value here is an integer held in a double: the lane count (m + w - 1) / w, and
// chain_work sums those, all far below 2^53. So the comparisons below are exact equality
// on purpose -- a tolerance would hide exactly the drift this file exists for.
double chain_work_recursive(std::size_t n, std::size_t w, std::size_t regs,
                            std::map<std::size_t, double>& memo) {
    if (n <= 1) return 0.0;
    if (const auto it = memo.find(n); it != memo.end()) return it->second;
    const std::size_t n_radices = regs >= 32 ? 11u : 7u;
    double best = -1.0;
    for (std::size_t k = 0; k < n_radices; ++k) {
        const std::size_t r = admiral::detail::kChainRadices[k];
        if (n % r) continue;
        const double sub = chain_work_recursive(n / r, w, regs, memo);
        if (sub < 0.0) continue;  // -1 propagates: a partial chain is not a chain
        const double c = double((n / r + w - 1) / w) + sub;
        if (best < 0.0 || c < best) best = c;
    }
    memo[n] = best;
    return best;
}

// (W, regs) pairs spanning every target in the receipt set the coefficients are pooled
// over, so the DP is checked where the fitter actually evaluates it and not only at this
// build's width: f64/f32 at W=2..16, narrow and wide radix sets both.
constexpr std::pair<std::size_t, std::size_t> kTargets[] = {
    {2, 16}, {4, 16}, {8, 16}, {8, 32}, {16, 32}};

}  // namespace

// chain_work is called past BASE_MODEL_NMAX: the bluestein feature passes the convolution
// pad, which reaches ~1024 for n=512, so the range follows the caller rather than the
// model's own domain.
TEST_CASE("cost model: chain_work DP equals the recursion it replaced") {
    for (const auto& [w, regs] : kTargets) {
        std::map<std::size_t, double> memo;
        for (std::size_t n = 1; n <= 1200; ++n) {
            const double oracle = chain_work_recursive(n, w, regs, memo);
            // The oracle carries the -1 sentinel; the shipped one clamps at its return.
            REQUIRE(admiral::detail::chain_work(n, w, regs) == (oracle < 0.0 ? 0.0 : oracle));
        }
    }
}

// The generated header's only local piece is the <T> binding of W and the register count,
// which the fitter cannot do because it pools targets in one process.
TEST_CASE("cost model: the header's <T> wrapper binds this build's width") {
    for (std::size_t n = 1; n <= 600; ++n) {
        REQUIRE(admiral::detail::base_model::chain_work<float>(n) ==
                admiral::detail::chain_work(n, admiral::detail::build_width<float>,
                                           admiral::detail::build_vector_regs));
        REQUIRE(admiral::detail::base_model::chain_work<double>(n) ==
                admiral::detail::chain_work(n, admiral::detail::build_width<double>,
                                           admiral::detail::build_vector_regs));
    }
}

TEST_CASE("cost model: balanced_split is the largest divisor <= sqrt(n)") {
    for (std::size_t n = 2; n <= 1200; ++n) {
        const auto [n1, n2] = admiral::detail::balanced_split(n);
        REQUIRE(n1 * n2 == n);
        REQUIRE(n1 <= n2);
        // The shipped one scans divisors upward and keeps the last; scan downward from n
        // and take the first at or below sqrt(n), which is a different traversal.
        std::size_t want = 1;
        for (std::size_t a = n; a >= 1; --a) {
            if (a * a <= n && n % a == 0) { want = a; break; }
        }
        REQUIRE(n1 == want);
    }
    // A prime has no split, and the model still scores four_step at every n, so this is the
    // degenerate case the feature expressions have to stay in range on.
    REQUIRE(admiral::detail::balanced_split(509)[0] == 1);
    REQUIRE(admiral::detail::balanced_split(509)[1] == 509);
}

TEST_CASE("cost model: lpf_nfac agrees with trial division") {
    for (std::size_t n = 2; n <= 1200; ++n) {
        const auto [lpf, nfac] = admiral::detail::lpf_nfac(n);
        std::size_t m = n, want_lpf = 1, want_nfac = 0;
        for (std::size_t d = 2; d <= m; ++d) {
            while (m % d == 0) { m /= d; ++want_nfac; want_lpf = d; }
        }
        REQUIRE(lpf == want_lpf);
        REQUIRE(nfac == want_nfac);
    }
    REQUIRE(admiral::detail::lpf_nfac(289)[0] == 17);  // the cofactor need not be prime
    REQUIRE(admiral::detail::lpf_nfac(1)[0] == 1);     // the rader feature reads L==1 at n==2
    REQUIRE(admiral::detail::lpf_nfac(1)[1] == 0);
}

// Two invariants the router reads directly: it takes the first buildable entry, so the
// ranking must be ascending in cost, and it falls through to the gate ladder on count == 0,
// which is the only thing keeping an out-of-domain size off an unfitted score.
TEMPLATE_TEST_CASE("cost model: the ranking is sorted, and empty exactly outside the domain",
                   "[route][model]", float, double) {
    using admiral::detail::base_route_ranking;
    for (std::size_t n = admiral::detail::BASE_MODEL_NMIN; n <= admiral::detail::BASE_MODEL_NMAX;
         ++n) {
        const auto& r = base_route_ranking<TestType>(n);
        REQUIRE(r.count == admiral::detail::base_model::NFORM);
        REQUIRE(r.best_cyc > 0.f);
        for (std::size_t k = 1; k < r.count; ++k) REQUIRE(r.log_cyc[k - 1] <= r.log_cyc[k]);
    }
    REQUIRE(base_route_ranking<TestType>(admiral::detail::BASE_MODEL_NMAX + 1).count == 0);
    REQUIRE(base_route_ranking<TestType>(admiral::detail::BASE_MODEL_NMIN - 1).count == 0);
    REQUIRE(base_route_ranking<TestType>(0).count == 0);
}

// The bluestein form carries no no_chain indicator (rader does) because its pad is
// {2,3,5,7}-smooth or a power of two, and 2, 3, 5, 7 are all admissible radices. The
// fitter asserts this at fit time to justify deleting that feature; assert it here too,
// against the narrow radix set the fitter checks, because the header that SHIPS is the one
// whose feature vector depends on it.
TEST_CASE("cost model: no bluestein pad falls off the radix chain") {
    for (std::size_t n = admiral::detail::BASE_MODEL_NMIN; n <= admiral::detail::BASE_MODEL_NMAX;
         ++n) {
        const std::size_t pad = admiral::detail::bluestein_choose_pad(n);
        REQUIRE(admiral::detail::chain_work(pad, 8, 16) > 0.0);
    }
}
