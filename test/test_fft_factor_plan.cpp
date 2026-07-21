#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <admiral/detail/twiddles.hpp>

namespace {

template<typename T>
double plan_cost(std::size_t N, const admiral::detail::dif_factor_plan& plan) {
    double cost = 0.0;
    std::size_t n = N;
    for (std::size_t i = 0; i < plan.count; ++i) {
        const unsigned radix = plan[i];
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
    for (unsigned radix : admiral::detail::dif_candidate_radices) {
        if (n % radix != 0) continue;
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
        const unsigned radix = plan[i];
        REQUIRE(std::find(admiral::detail::dif_candidate_radices.begin(),
                          admiral::detail::dif_candidate_radices.end(),
                          radix) != admiral::detail::dif_candidate_radices.end());
        product *= radix;
    }
    REQUIRE(product == N);

    // The additive-cost optimality invariant below only holds where the additive
    // DP is authoritative. For pow2 N in the fusion band the planner instead
    // minimizes a richer fusion-discounted cost (enumerate_pow2_dif_plan, with
    // fuse discounts + terminal codelets), so a naive additive brute-force is
    // not its objective and would flag a legitimately-better plan as suboptimal.
    // That path is validated by the FFT correctness tests + role-swapped A/B
    // receipts, not here. (kDifFuseMinN is ISA/precision-dependent: v4=8192, so
    // no test size reaches it there — this only fired on v1/v2/v3.)
    if ((N & (N - 1u)) == 0u && N >= admiral::detail::kDifFuseMinN<T>) return;

    // max() as the min-search sentinel: infinity is UB under -ffast-math.
    double best = std::numeric_limits<double>::max();
    enumerate_best<T>(N, N, 0.0, best);
    REQUIRE(best < std::numeric_limits<double>::max());  // a factorization was costed
    // -ffast-math reassociates independently at each inline site of
    // dif_stage_cost, so identical chains can differ by a few ulps.
    REQUIRE(plan_cost<T>(N, plan) <= best * (1.0 + 1e-12));
}

} // namespace

TEMPLATE_TEST_CASE("DIF factor planner is optimal for modeled cost", "[factor_plan]",
                   float, double) {
    for (std::size_t N : {60u, 90u, 120u, 360u, 720u, 1000u, 1024u,
                          2048u, 2520u, 4096u}) {
        CAPTURE(N);
        check_optimal<TestType>(N);
    }
}
