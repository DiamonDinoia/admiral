#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <admiral/detail/four_step.hpp>
#include <admiral/detail/twiddles.hpp>

#include <map>

namespace {

template<typename T>
double plan_cost(std::size_t N, const admiral::detail::dif_factor_plan& plan) {
    double cost = 0.0;
    std::size_t n = N;
    for (std::size_t i = 0; i < plan.count; ++i) {
        const std::size_t radix = plan[i];
        cost += admiral::detail::dif_stage_cost<T>(N, n, radix);
        n /= radix;
    }
    return cost;
}

template<typename T>
void enumerate_best(std::size_t N, std::size_t n, double partial, double& best) {
    if (n == 1) {
        best = std::min(best, partial);
        return;
    }
    for (const std::size_t radix : admiral::detail::dif_candidate_radices) {
        if (n % radix != 0 || !admiral::detail::dif_radix_admissible(N, radix)) continue;
        enumerate_best<T>(N, n / radix,
                          partial + admiral::detail::dif_stage_cost<T>(N, n, radix),
                          best);
    }
}

template<typename T>
void check_optimal(std::size_t N) {
    const auto plan = admiral::detail::build_dif_factor_plan<T>(N);
    REQUIRE(plan.count > 0);

    std::size_t product = 1;
    for (std::size_t i = 0; i < plan.count; ++i) {
        const std::size_t radix = plan[i];
        REQUIRE(std::find(admiral::detail::dif_candidate_radices.begin(),
                          admiral::detail::dif_candidate_radices.end(),
                          radix) != admiral::detail::dif_candidate_radices.end());
        product *= radix;
    }
    REQUIRE(product == N);
    if ((N & (N - 1u)) == 0u && N >= admiral::detail::kDifFuseMinN<T>) return;

    double best = std::numeric_limits<double>::max();
    enumerate_best<T>(N, N, 0.0, best);
    REQUIRE(best < std::numeric_limits<double>::max());
    constexpr double kReassocEps = 1e-12;
    REQUIRE(plan_cost<T>(N, plan) <= best * (1.0 + kReassocEps));
}

}

template<typename T>
std::pair<bool, std::size_t> min_valley_passes(std::size_t N, std::size_t n,
                                               std::map<std::size_t, std::pair<bool, std::size_t>>& memo) {
    if (n == 1) return {true, 0};
    if (const auto it = memo.find(n); it != memo.end()) return it->second;
    constexpr std::size_t W = xsimd::batch<T>::size;
    constexpr std::size_t kInf = std::numeric_limits<std::size_t>::max() / 4;
    std::size_t best = kInf;
    auto relax = [&](std::size_t radix) {
        const std::size_t ido = n / radix;
        const auto [ok, sub] = min_valley_passes<T>(N, n / radix, memo);
        if (ok) best = std::min(best, sub + ((ido > 1 && ido < W) ? 1u : 0u));
    };
    for (const std::size_t radix : admiral::detail::dif_candidate_radices)
        if (n % radix == 0 && admiral::detail::dif_radix_admissible(N, radix)) relax(radix);
    if (n != N)
        for (const std::size_t g : admiral::detail::dif_generic_radices)
            if (n % g == 0 && n / g > 1) relax(g);
    return memo[n] = {best != kInf, best == kInf ? 0 : best};
}

TEMPLATE_TEST_CASE("DIF factor planner never elects an avoidable valley pass", "[factor_plan]",
                   float, double) {
    using namespace admiral::detail;
    constexpr std::size_t W = xsimd::batch<TestType>::size;
    for (std::size_t N = 2; N <= 4096; ++N) {
        const auto plan = build_dif_factor_plan<TestType>(N);
        if (plan.count == 0) continue;
        if ((N & (N - 1u)) == 0u && N >= kDifFuseMinN<TestType>) continue;
        std::size_t n = N, valley = 0;
        for (std::size_t i = 0; i < plan.count; ++i) {
            const std::size_t ido = n / plan[i];
            if (ido > 1 && ido < W) ++valley;
            n = ido;
        }
        std::map<std::size_t, std::pair<bool, std::size_t>> memo;
        const auto [any, min_valley] = min_valley_passes<TestType>(N, N, memo);
        if (!any) continue;
        CAPTURE(N);
        REQUIRE(valley == min_valley);
    }
}

TEMPLATE_TEST_CASE("DIF factor planner is optimal for modeled cost", "[factor_plan]",
                   float, double) {
    using namespace admiral::detail;
    const auto additively_priced = [](const dif_factor_plan& p) {
        for (std::size_t i = 0; i < p.count; ++i)
            if (std::find(dif_candidate_radices.begin(), dif_candidate_radices.end(), p[i])
                == dif_candidate_radices.end())
                return false;
        return true;
    };
    for (std::size_t N = 2; N <= 512; ++N) {
        const dif_factor_plan plan = build_dif_factor_plan<TestType>(N);
        if (plan.count == 0 || !additively_priced(plan)) continue;
        CAPTURE(N);
        check_optimal<TestType>(N);
    }
    for (const std::size_t N : {720u, 1000u, 1024u, 2048u, 2520u, 4096u}) {
        CAPTURE(N);
        check_optimal<TestType>(N);
    }
}

TEMPLATE_TEST_CASE("dif candidate list: legal, distinct, and the election lands on it",
                   "[factor_plan]", float, double) {
    using namespace admiral::detail;
    const auto census = [](std::size_t N) {
        const dif_chain_list c = dif_chain_candidates<TestType>(N);
        CAPTURE(N, c.count);
        REQUIRE(c.count >= 1);
        REQUIRE(c.count <= kDifCandidates);
        for (std::size_t k = 0; k < c.count; ++k) {
            CAPTURE(k);
            std::size_t prod = 1;
            for (std::size_t i = 0; i < c[k].count; ++i) prod *= c[k][i];
            REQUIRE(prod == N);
            if (dif_chain_shape_ok<TestType>(N, c[k]))
                for (std::size_t i = 0; i < c[k].count; ++i) {
                    const std::size_t r = c[k][i];
                    CAPTURE(i, r);
                    if (dif_is_generic_radix(r)) {
                        REQUIRE(i > 0);
                        REQUIRE(i + 1 < c[k].count);
                    } else {
                        REQUIRE(in_seq(dif_radix_set{}, r));
                    }
                }
            for (std::size_t j = 0; j < k; ++j)
                REQUIRE_FALSE((c[j].count == c[k].count &&
                               std::equal(c[j].radices.begin(),
                                          c[j].radices.begin()
                                              + static_cast<std::ptrdiff_t>(c[j].count),
                                          c[k].radices.begin())));
        }
        const auto dp = build_dif_factor_plan<TestType>(N);
        REQUIRE(dp.count == c[0].count);
        REQUIRE(std::equal(dp.radices.begin(),
                       dp.radices.begin() + static_cast<std::ptrdiff_t>(dp.count),
                       c[0].radices.begin()));
        const auto el = dif_elected_chain<TestType>(N);
        bool member = false;
        for (std::size_t k = 0; k < c.count; ++k)
            member = member || (el.count == c[k].count &&
                                std::equal(el.radices.begin(),
                                           el.radices.begin()
                                               + static_cast<std::ptrdiff_t>(el.count),
                                           c[k].radices.begin()));
        REQUIRE(member);
        bool any_ok = false;
        for (std::size_t k = 0; k < c.count; ++k)
            any_ok = any_ok || dif_chain_shape_ok<TestType>(N, c[k]);
        REQUIRE(any_ok == dif_chain_shape_ok<TestType>(N, el));
    };
    for (std::size_t N = 2; N <= 20000; ++N) census(N);
    for (const std::size_t N : {std::size_t{25575}, std::size_t{30000}, std::size_t{58960},
                                std::size_t{90475}, std::size_t{100000}, std::size_t{113876},
                                std::size_t{122220}, std::size_t{160688}, std::size_t{194000},
                                std::size_t{206360}, std::size_t{218889}, std::size_t{245630},
                                std::size_t{262144}, std::size_t{390625}, std::size_t{1048575}})
        census(N);
}

TEMPLATE_TEST_CASE("starved generic tail bounds chain availability from both sides",
                   "[factor_plan]", float, double) {
    using namespace admiral::detail;
    constexpr std::size_t W = xsimd::batch<TestType>::size;
    if constexpr (W > 2) {
        CHECK_FALSE(dif_chain_shape_ok<TestType>(188, dif_factor_plan{2, 47, 2}));
        CHECK(dif_chain_shape_ok<TestType>(124, dif_factor_plan{2, 31, 2}));
    }
    static_assert(W <= 16);
    CHECK(dif_chain_shape_ok<TestType>(1504, dif_factor_plan{2, 47, 4, 4}));
}

TEMPLATE_TEST_CASE("a wide radix in the valley outprices a narrow one", "[factor_plan]",
                   float, double) {
    using namespace admiral::detail;
    constexpr std::size_t W = xsimd::batch<TestType>::size;
    if constexpr (W > 2) {
        CHECK(dif_valley_penalty<TestType>(2, 5) > 0.0);
        CHECK(dif_valley_penalty<TestType>(2, 15) > dif_valley_penalty<TestType>(2, 5));
    }
    CHECK(dif_valley_penalty<TestType>(W, 15) == 0.0);
    const auto chain = dif_elected_chain<TestType>(120);
    REQUIRE(chain.count >= 2);
    if constexpr (W > 8)
        CHECK(chain[0] != 15u);
}

TEMPLATE_TEST_CASE("four_step split: argmin, admissible, and order-blind",
                   "[factor_plan]", float, double) {
    using namespace admiral::detail;

    constexpr double lam = sizeof(TestType) == 4
                               ? kStridePenaltyF32
                               : (xsimd::batch<TestType>::size >= 4 ? kStridePenaltyF64 : 0.0);
    constexpr std::size_t Lsat = sizeof(TestType) == 4 ? kStrideSatF32 : kStrideSatF64;
    const auto exec_cost = [&](std::size_t n1, std::size_t n2) {
        const double stride = double(std::min(n1, Lsat) + std::min(n2, Lsat));
        return double(n1) * gate_leaf_cyc(n2) + double(n2) * gate_leaf_cyc(n1)
               + lam * double(n1 * n2) * stride;
    };

    const auto admissible = [](std::size_t n1, std::size_t n2) {
        return n1 <= kFourStepLeafMax && n2 <= kFourStepLeafMax && is_codelet_catalog(n1)
               && is_codelet_catalog(n2);
    };

    for (std::size_t N = 2; N <= 6000; ++N) {
        const four_step_split sym = choose_four_step_split(N);
        const four_step_split ex = choose_four_step_split_exec<TestType>(N);

        bool any = false;
        for (std::size_t n1 = 2; n1 * n1 <= N; ++n1)
            if (N % n1 == 0) any = any || admissible(n1, N / n1);
        REQUIRE(sym.valid() == any);
        REQUIRE(ex.valid() == any);
        REQUIRE(four_step_supported(N) == (N > kFourStepLeafMax && any));
        if (!any) continue;

        for (const four_step_split s : {sym, ex}) {
            REQUIRE(s.n1 * s.n2 == N);
            REQUIRE(admissible(s.n1, s.n2));
        }
        const double fwd = exec_cost(ex.n1, ex.n2), rev = exec_cost(ex.n2, ex.n1);
        REQUIRE(std::abs(fwd - rev) <= 1e-12 * std::abs(fwd));

        double floor_cost = std::numeric_limits<double>::max();
        for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
            if (N % n1 != 0 || !admissible(n1, N / n1)) continue;
            floor_cost = std::min(floor_cost, gate_four_step_cost(n1, N / n1));
        }
        REQUIRE(gate_four_step_cost(sym.n1, sym.n2) == floor_cost);
    }
}
