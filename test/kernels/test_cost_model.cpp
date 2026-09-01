#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/base_cost_model.hpp>
#include <admiral/detail/math.hpp>

#include <cstddef>
#include <map>
#include <utility>

namespace {

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
        if (sub < 0.0) continue;
        const double c = double((n / r + w - 1) / w) + sub;
        if (best < 0.0 || c < best) best = c;
    }
    memo[n] = best;
    return best;
}

constexpr std::pair<std::size_t, std::size_t> kTargets[] = {
    {2, 16}, {4, 16}, {8, 16}, {8, 32}, {16, 32}};

}

TEST_CASE("cost model: chain_work DP equals the recursion it replaced") {
    for (const auto& [w, regs] : kTargets) {
        std::map<std::size_t, double> memo;
        for (std::size_t n = 1; n <= 1200; ++n) {
            const double oracle = chain_work_recursive(n, w, regs, memo);
            REQUIRE(admiral::detail::chain_work(n, w, regs) == (oracle < 0.0 ? 0.0 : oracle));
        }
    }
}

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
        std::size_t want = 1;
        for (std::size_t a = n; a >= 1; --a) {
            if (a * a <= n && n % a == 0) { want = a; break; }
        }
        REQUIRE(n1 == want);
    }
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
    REQUIRE(admiral::detail::lpf_nfac(289)[0] == 17);
    REQUIRE(admiral::detail::lpf_nfac(1)[0] == 1);
    REQUIRE(admiral::detail::lpf_nfac(1)[1] == 0);
}

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

TEST_CASE("cost model: no bluestein pad falls off the radix chain") {
    for (std::size_t n = admiral::detail::BASE_MODEL_NMIN; n <= admiral::detail::BASE_MODEL_NMAX;
         ++n) {
        const std::size_t pad = admiral::detail::bluestein_choose_pad(n);
        REQUIRE(admiral::detail::chain_work(pad, 8, 16) > 0.0);
    }
}
