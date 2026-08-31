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
    // Apply the same admissibility filter `min_valley_passes` applies. Without the filter
    // the oracle can reach a chain the DP is forbidden to elect (r25 off a
    // non-pentanomial `N`). The oracle then reports the planner suboptimal against a plan
    // that does not exist.
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
    // The additive-cost optimality invariant below holds only where the additive DP is
    // authoritative. For pow2 `N` in the fusion band the planner instead minimizes a
    // richer fusion-discounted cost (`enumerate_pow2_dif_plan`, fuse discounts + terminal
    // codelets). An additive brute-force is not the objective there and would flag a
    // legitimately-better plan as suboptimal. The FFT correctness tests validate the
    // fusion-band path, not this test. `kDifFuseMinN` is ISA/precision-dependent: 8192 at
    // v4, which no test size here reaches.
    if ((N & (N - 1u)) == 0u && N >= admiral::detail::kDifFuseMinN<T>) return;

    // `max()` as the min-search sentinel: infinity is UB under `-ffast-math`.
    double best = std::numeric_limits<double>::max();
    enumerate_best<T>(N, N, 0.0, best);
    REQUIRE(best < std::numeric_limits<double>::max());  // a factorization was costed
    // `-ffast-math` reassociates independently at each inline site of
    // `dif_stage_cost`, so identical chains can differ by a few ulps.
    constexpr double kReassocEps = 1e-12;
    REQUIRE(plan_cost<T>(N, plan) <= best * (1.0 + kReassocEps));
}

} // namespace

// The DP prices a 1 < `ido` < `W` pass with a tier far above any real chain cost, so an
// elected chain may carry a valley pass ONLY when some valley pass is unavoidable. The
// oracle mirrors the DP's candidate set (static radices + generic middle passes only,
// same admissibility) and memoizes over divisors.
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
        // `enumerate_pow2_dif_plan` scores the fused-pow2 band on the same terms, but the
        // chain set is restricted on purpose, so skip the fused-pow2 band here.
        if ((N & (N - 1u)) == 0u && N >= kDifFuseMinN<TestType>) continue;
        std::size_t n = N, valley = 0;
        for (std::size_t i = 0; i < plan.count; ++i) {
            const std::size_t ido = n / plan[i];
            if (ido > 1 && ido < W) ++valley;
            n = ido;
        }
        std::map<std::size_t, std::pair<bool, std::size_t>> memo;
        const auto [any, min_valley] = min_valley_passes<TestType>(N, N, memo);
        if (!any) continue;  // no DP-reachable factorization at all (odd prime mix)
        CAPTURE(N);
        REQUIRE(valley == min_valley);
    }
}

TEMPLATE_TEST_CASE("DIF factor planner is optimal for modeled cost", "[factor_plan]",
                   float, double) {
    using namespace admiral::detail;
    // A sweep, not a hand-picked list. Brute-force enumeration is a valid oracle at every
    // `N` the additive DP prices: every `N` whose elected chain is all candidate radices.
    // A generic prime in the chain means `dif_generic_stage_cost` priced the chain
    // instead, a different objective, like the pow2 fusion band `check_optimal` skips.
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

// The DP hands the plan-time race its K cheapest chains, and ANY of the K can be
// elected. Every invariant is a property of the whole list, not of the argmin alone.
// Structural legality holds per candidate, and the list itself must be well-formed:
// distinct entries, argmin first, and an election that lands on a member.
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
            REQUIRE(prod == N);  // a candidate that does not factor `N` would transform garbage
            // The assertions cover the chains that can be ELECTED. The list carries the
            // DP's no-chain fallback (bare 13 -> [13]) at sizes with no chain at all. The
            // fallback is a feature-test channel, not a runnable plan.
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
            for (std::size_t j = 0; j < k; ++j)  // a duplicate wastes a race slot
                REQUIRE_FALSE((c[j].count == c[k].count &&
                               std::equal(c[j].radices.begin(), c[j].radices.begin() + c[j].count,
                                          c[k].radices.begin())));
        }
        // `build_dif_factor_plan` IS candidate 0, and the election is one of the candidates.
        const auto dp = build_dif_factor_plan<TestType>(N);
        REQUIRE(dp.count == c[0].count);
        REQUIRE(std::equal(dp.radices.begin(), dp.radices.begin() + dp.count, c[0].radices.begin()));
        const auto el = dif_elected_chain<TestType>(N);
        bool member = false;
        for (std::size_t k = 0; k < c.count; ++k)
            member = member || (el.count == c[k].count &&
                                std::equal(el.radices.begin(), el.radices.begin() + el.count,
                                           c[k].radices.begin()));
        REQUIRE(member);
        // Whenever any candidate can run, the elected one runs.
        bool any_ok = false;
        for (std::size_t k = 0; k < c.count; ++k)
            any_ok = any_ok || dif_chain_shape_ok<TestType>(N, c[k]);
        REQUIRE(any_ok == dif_chain_shape_ok<TestType>(N, el));
    };
    for (std::size_t N = 2; N <= 20000; ++N) census(N);
    // Hand-picked awkward factorisations, plus the large generic-prime placements.
    for (const std::size_t N : {std::size_t{25575}, std::size_t{30000}, std::size_t{58960},
                                std::size_t{90475}, std::size_t{100000}, std::size_t{113876},
                                std::size_t{122220}, std::size_t{160688}, std::size_t{194000},
                                std::size_t{206360}, std::size_t{218889}, std::size_t{245630},
                                std::size_t{262144}, std::size_t{390625}, std::size_t{1048575}})
        census(N);
}

// `kGenericStarvedTailMinRadix` is a measured bound, so this case pins the bound from
// BOTH sides. A generic radix above the bound at a part-width `ido` must lose
// availability, and one below the bound must keep availability. Moving the bound either
// way loses cells.
TEMPLATE_TEST_CASE("starved generic tail bounds chain availability from both sides",
                   "[factor_plan]", float, double) {
    using namespace admiral::detail;
    constexpr std::size_t W = xsimd::batch<TestType>::size;
    // 188 = 2*47*2 and 124 = 2*31*2 park their generic prime at `ido` == 2. The two share
    // a shape but get opposite verdicts, because 47 is at the bound and 31 is below the
    // bound. Assert the band through `dif_chain_shape_ok`, never by restating the
    // constant. A numeric assertion on `kGenericStarvedTailMinRadix` short-circuits the
    // case and leaves the guarded behaviour untested.
    if constexpr (W > 2) {
        CHECK_FALSE(dif_chain_shape_ok<TestType>(188, dif_factor_plan{2, 47, 2}));
        CHECK(dif_chain_shape_ok<TestType>(124, dif_factor_plan{2, 31, 2}));
    }
    // Above the valley the large prime is admissible again. 1504 = 2*47*4*4 sits the 47
    // at `ido` == 16, a full tile at every width the library builds for.
    static_assert(W <= 16);
    CHECK(dif_chain_shape_ok<TestType>(1504, dif_factor_plan{2, 47, 4, 4}));
}

// One flat valley tier cancels between two chains that each hold exactly one valley pass,
// so the residual ranks the two. One flat tier elected an r15 valley over the valley's
// own 3*5 split. The wide-radix tier separates the two, so this case asserts the ordering
// directly.
TEMPLATE_TEST_CASE("a wide radix in the valley outprices a narrow one", "[factor_plan]",
                   float, double) {
    using namespace admiral::detail;
    constexpr std::size_t W = xsimd::batch<TestType>::size;
    if constexpr (W > 2) {
        // `ido` == 2 is a valley at every width the library builds for.
        CHECK(dif_valley_penalty<TestType>(2, 5) > 0.0);
        CHECK(dif_valley_penalty<TestType>(2, 15) > dif_valley_penalty<TestType>(2, 5));
    }
    // A full tile is never a valley, however wide the radix.
    CHECK(dif_valley_penalty<TestType>(W, 15) == 0.0);
    // 120 = 15*8 parks r15 at `ido` == 8, a valley only once `W` exceeds 8. If `ido` == 8
    // is a valley the election must split the 15; if not, r15 stays admissible there.
    const auto chain = dif_elected_chain<TestType>(120);
    REQUIRE(chain.count >= 2);
    if constexpr (W > 8)
        CHECK(chain[0] != 15u);
}

// The four-step split chooser is the route decision for every `N` above the codelet
// ceiling that is not prime, and nothing re-derives the chooser's argmin. Both choosers
// are pure functions of `N`, so the test brute-forces the same search and compares.
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

        // Existence: the symmetric chooser finds a split iff one exists. The exec chooser
        // searches the same admissible set, so the two agree on validity even though the
        // two minimise different objectives.
        bool any = false;
        for (std::size_t n1 = 2; n1 * n1 <= N; ++n1)
            if (N % n1 == 0) any = any || admissible(n1, N / n1);
        REQUIRE(sym.valid() == any);
        REQUIRE(ex.valid() == any);
        REQUIRE(four_step_supported(N) == (N > kFourStepLeafMax && any));
        if (!any) continue;

        // Both results are admissible factorisations of `N`,
        for (const four_step_split s : {sym, ex}) {
            REQUIRE(s.n1 * s.n2 == N);
            REQUIRE(admissible(s.n1, s.n2));
        }
        // The exec chooser's objective is symmetric in (`n1`, `n2`): both the leaf term
        // and min(n1,L)+min(n2,L) are. So the objective CANNOT rank the two memory orders,
        // even though execution can tell the two apart. Only FMA contraction separates the
        // two, at a few ulps. The contraction is what makes the `n1 <= N / 2` bound elect
        // the larger factor first at a handful of sizes. Assert the symmetry with the same
        // reassociation epsilon the rest of the file uses. A real asymmetric penalty must
        // surface here as a test change, never as rounding.
        const double fwd = exec_cost(ex.n1, ex.n2), rev = exec_cost(ex.n2, ex.n1);
        REQUIRE(std::abs(fwd - rev) <= 1e-12 * std::abs(fwd));

        // and the symmetric one is the argmin of the cost the symmetric chooser minimises.
        double floor_cost = std::numeric_limits<double>::max();
        for (std::size_t n1 = 2; n1 * n1 <= N; ++n1) {
            if (N % n1 != 0 || !admissible(n1, N / n1)) continue;
            floor_cost = std::min(floor_cost, gate_four_step_cost(n1, N / n1));
        }
        REQUIRE(gate_four_step_cost(sym.n1, sym.n2) == floor_cost);
    }
}
