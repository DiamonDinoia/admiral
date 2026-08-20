// The runtime-prime DIF middle pass (dif_pass_prime_chip, primes 13..97 mid-chain;
// batched-Rader leaf per a-tile). What pins what:
// * Forced chains vs the naive reference at small N (both directions, in place and
//   out of place), covering a generic masked tail (465's ido==5 < W8) and a ragged
//   one (25575's ido==341 % 8 = 5).
// * Elected-route cells: route name + impulse flatness at 25575 (DP elects
//   5-15-31-11) and at 262143 / 1048575, where the O(N^2) reference is infeasible
//   and X[k] == 1 for impulse input is exact at any N.
// * Boundary trap: bare or bookended generic fallback chains (13, 26, 65, 806) must
//   stay out of iterative_dif: electing them throws at dispatch (poet no_match).

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/plan.hpp>

#include "utils/reference.hpp"

#include <complex>
#include <string>
#include <type_traits>
#include <vector>

using admiral::detail::dif_factor_plan;
using admiral::detail::exec_options;
using admiral::detail::plan_impl;

// API pin: threading is fixed at plan creation; execute-time options carry
// no pool member, so no call site can ever hold one (compile-fail-by-construction).
template<typename O, typename = void>
inline constexpr bool has_pool_member = false;
template<typename O>
inline constexpr bool has_pool_member<O, std::void_t<decltype(std::declval<O&>().pool)>> = true;
static_assert(!has_pool_member<exec_options<float>>);
static_assert(!has_pool_member<exec_options<double>>);

namespace {

template<typename T>
void check_forced(std::size_t N, const dif_factor_plan& ov, bool forward, bool in_place) {
    auto in = make_signal<T>(N);
    std::vector<std::complex<T>> out(in);  // in-place shape starts from in either way
    plan_impl<T> pl(N, forward, 1, &ov);
    const exec_options<T> one{.fct = T(1)};
    if (in_place) pl.execute(out.data(), out.data(), one);
    else          pl.execute(in.data(), out.data(), one);
    require_close(out, reference_dft<T>(in, forward), fft_tol<T>(8));
}

// Driver-level force: the same kernels and the same tape, without plan_impl's
// availability veto (which deliberately keeps a chiplet prime at a SHORT ido out
// of production routes; see dif_chain_supported). The chiplet cells below sit
// on the vetoed side by construction (97/83 at ido 8..44).
template<typename T>
void check_forced_raw(std::size_t N, const dif_factor_plan& ov, bool forward, bool in_place) {
    auto in = make_signal<T>(N);
    std::vector<std::complex<T>> out(in);
    const auto tw = admiral::detail::build_dif_twiddle_set<T>(N, &ov);
    if (in_place)
        admiral::detail::dif_execute_in_place<T>(forward, out.data(), out.data(), N, tw, T(1));
    else
        admiral::detail::dif_execute_in_place<T>(forward, in.data(), out.data(), N, tw, T(1));
    require_close(out, reference_dft<T>(in, forward), fft_tol<T>(8));
}

template<typename T>
void impulse_flat(std::size_t N, double tol_scale) {
    std::vector<std::complex<T>> in(N, {T(0), T(0)}), got(N);
    in[0] = {T(1), T(0)};
    plan_impl<T> fwd(N, true, 1, nullptr);
    const exec_options<T> one{.fct = T(1)};
    fwd.execute(in.data(), got.data(), one);
    T num = 0;
    for (const auto& v : got) num += std::norm(v - std::complex<T>{T(1), T(0)});
    const double rel = std::sqrt(double(num) / double(N));
    INFO("impulse flatness N=" << N << " rel=" << rel << " route=" << fwd.route_name());
    REQUIRE(rel <= fft_tol<T>(tol_scale));
}

}  // namespace

TEMPLATE_TEST_CASE("generic prime pass: forced chains vs naive DFT", "[accuracy][generic]",
                   float, double) {
    using T = TestType;
    REQUIRE(admiral::detail::dif_is_generic_radix(31u));
    // Generic strictly mid-chain; every pass radix valid on ALL ISAs (the
    // register-parametric radix set drops 9/15/16/25/32 when regs <= 16, and forcing
    // one of them throws by design). Masked tails: 465's ido=5 (full-width at
    // no ISA... it masks at W=2/4), 25575's generic pass sees ido=11.
    for (bool fwd : {false, true}) {
        check_forced<T>(465, dif_factor_plan{3, 31, 5}, fwd, false);      // ido=5 masked tile
        check_forced<T>(465, dif_factor_plan{3, 31, 5}, fwd, true);
        check_forced<T>(3135, dif_factor_plan{5, 19, 11, 3}, fwd, false);
        check_forced<T>(3135, dif_factor_plan{5, 19, 11, 3}, fwd, true);
        // Generic at a W-multiple ido: the es2 W-blocked SoA generation can bracket
        // the pass (audit hole id). 832 = 8*13*8, ido at the 13 is 8.
        check_forced<T>(832, dif_factor_plan{8, 13, 8}, fwd, false);
        check_forced<T>(832, dif_factor_plan{8, 13, 8}, fwd, true);
    }
    // 25575 round-trips through a chain one slot deeper in the generic region,
    // both directions, in place and out, with ISA-portable pass radices.
    for (bool ip : {false, true}) {
        const auto ov = dif_factor_plan{5, 3, 31, 11, 5};
        const std::vector<std::complex<T>> in = make_signal<T>(25575);
        std::vector<std::complex<T>> buf(in);
        plan_impl<T> fwd(25575, true, 1, &ov), inv(25575, false, 1, &ov);
        const exec_options<T> one{.fct = T(1)};
        const exec_options<T> unscale{.fct = T(1) / T(25575)};  // inverse normalization
        if (ip) {
            fwd.execute(buf.data(), buf.data(), one);
            inv.execute(buf.data(), buf.data(), unscale);
        } else {
            std::vector<std::complex<T>> spec(in.size());
            fwd.execute(in.data(), spec.data(), one);
            inv.execute(spec.data(), buf.data(), unscale);
        }
        require_close(buf, in, fft_tol<T>(2));
    }
}

TEMPLATE_TEST_CASE("prime chiplet census: every pool prime vs naive DFT", "[accuracy][generic]",
                   float, double) {
    using T = TestType;
    // Every pool prime's chiplet table, forced. Bookends are ISA-portable; ido at
    // the generic pass: 4 (masked at W>=8-f64), 8 (full tiles at W<=8), 7 (ragged).
    for (const std::size_t g : admiral::detail::dif_generic_radices) {
        CAPTURE(g);
        for (bool fwd : {false, true}) {
            check_forced_raw<T>(4 * g * 4, dif_factor_plan{4, g, 4}, fwd, false);
            check_forced_raw<T>(4 * g * 4, dif_factor_plan{4, g, 4}, fwd, true);
            check_forced_raw<T>(4 * g * 8, dif_factor_plan{4, g, 8}, fwd, false);
            check_forced_raw<T>(3 * g * 7, dif_factor_plan{3, g, 7}, fwd, false);
        }
    }
}

TEMPLATE_TEST_CASE("prime chiplet pass: forced chains vs naive DFT", "[accuracy][generic]",
                   float, double) {
    using T = TestType;
    REQUIRE(admiral::detail::dif_is_generic_radix(67u));
    REQUIRE(admiral::detail::dif_is_generic_radix(97u));
    // The prime chiplet (batched Rader per a-tile) is the leaf for every radix in
    // dif_generic_radices; all bookend radices below are valid on all ISAs.
    // Masked tails: 1005's ido==5 at the 67, 4268's ido==11 (masks at
    // every W), 25564's ido==44 at the 83 (ragged at every W).
    // Every prime below exceeds kGenericStarvedTailMinRadix at a part-width ido, so all
    // of them take the raw driver path. The plan path's generic coverage, which keeps
    // plan_impl's availability veto, is 465 / 3135 / 832 / 25575.
    for (bool fwd : {false, true}) {
        check_forced_raw<T>(1005, dif_factor_plan{3, 67, 5}, fwd, false);
        check_forced_raw<T>(1005, dif_factor_plan{3, 67, 5}, fwd, true);
        check_forced_raw<T>(4268, dif_factor_plan{4, 97, 11}, fwd, false);
        check_forced_raw<T>(4268, dif_factor_plan{4, 97, 11}, fwd, true);
    }
    // Round-trip force (O(N^2) references are suite-hostile past ~5k on the
    // baseline ISAs): a dropped sign in the chiplet's consteval tables is still
    // pinned by the elected-route impulse flats below and by the 1005/4268
    // reference checks above. 6208 = 8.97.8 sits the 97 at ido 8 (W-multiple on
    // f64 -> full tiles through the es2 W-blocked bracket); 25564 = 7.83.11.4
    // puts the 83 mid-chain at ido 44 (ragged tail at every W).
    for (bool ip : {false, true}) {
        for (const auto& [N, ov] : {std::pair{std::size_t{6208}, dif_factor_plan{8, 97, 8}},
                                    {std::size_t{25564}, dif_factor_plan{7, 83, 11, 4}}}) {
            const std::vector<std::complex<T>> in = make_signal<T>(N);
            std::vector<std::complex<T>> buf(in);
            const auto tw = admiral::detail::build_dif_twiddle_set<T>(N, &ov);
            if (ip) {
                admiral::detail::dif_execute_in_place<T>(true, buf.data(), buf.data(), N, tw, T(1));
                admiral::detail::dif_execute_in_place<T>(false, buf.data(), buf.data(), N, tw,
                                                         T(1) / T(N));
            } else {
                std::vector<std::complex<T>> spec(in.size());
                admiral::detail::dif_execute_in_place<T>(true, in.data(), spec.data(), N, tw, T(1));
                admiral::detail::dif_execute_in_place<T>(false, spec.data(), buf.data(), N, tw,
                                                         T(1) / T(N));
            }
            require_close(buf, in, fft_tol<T>(2));
        }
    }
}

TEMPLATE_TEST_CASE("elected generic-pass sizes: route + impulse flatness", "[accuracy][generic]",
                   float, double) {
    using T = TestType;
    {
        plan_impl<T> fwd(25575, true, 1, nullptr);
        // f64 elects iterative_dif here; the chain that wins it (and whether dif
        // beats bluestein at all) is ISA-dependent, so pin accuracy not routes.
        if constexpr (sizeof(T) == 8)
            if (poet::vector_register_count() >= 32)
                CHECK(std::string(fwd.route_name()) == "iterative_dif");
    }
    impulse_flat<T>(25575, 2.0);
    impulse_flat<T>(262143, 4.0);
    impulse_flat<T>(1048575, 6.0);
    // Chiplet cells (f64 W=8: the candidate walk elects a 97-carrying chain here;
    // other ISAs route elsewhere, and impulse flatness pins accuracy on any).
    impulse_flat<T>(194000, 4.0);
    impulse_flat<T>(122220, 4.0);
    // A broken election silently reverts these to bluestein: wherever the candidate walk
    // finds a runnable chain for this (T, ISA), the route must stay iterative_dif.
    for (const std::size_t N : {std::size_t{83424}, std::size_t{113876}, std::size_t{122220},
                                std::size_t{160688}, std::size_t{194000}, std::size_t{262143},
                                std::size_t{1048575}}) {
        CAPTURE(N);
        if (admiral::detail::dif_chain_shape_ok<T>(N, admiral::detail::dif_elected_chain<T>(N))) {
            plan_impl<T> fwd(N, true, 1, nullptr);
            CHECK(std::string(fwd.route_name()) == "iterative_dif");
        }
    }
}

TEMPLATE_TEST_CASE("boundary-generic chains stay out of iterative_dif", "[coverage][generic]",
                   float, double) {
    using T = TestType;
    using P = plan_impl<T>;
    using R = typename P::route_kind;
    for (const std::size_t N :
         {std::size_t{13}, std::size_t{26}, std::size_t{65}, std::size_t{806}}) {
        REQUIRE_FALSE(P::route_available(R::iterative_dif, N));
        // ... and the size still transforms correctly wherever it routes.
        std::vector<std::complex<T>> in(N), out(N);
        in = make_signal<T>(N);
        const exec_options<T> one{.fct = T(1)};
        plan_impl<T> fwd(N, true, 1, nullptr);
        fwd.execute(in.data(), out.data(), one);
        require_close(out, reference_dft<T>(in, true), fft_tol<T>(4));
    }
    // A forced chain may not close with a generic pass either: [3,31] has no
    // dif_pass_last<31>, so executing it must throw (poet no-match) rather than
    // leave a silently untransformed buffer.
    {
        const auto bad = dif_factor_plan{3, 31};
        std::vector<std::complex<T>> in(93), out(93);
        const exec_options<T> one{.fct = T(1)};
        REQUIRE_THROWS(([&] {
            plan_impl<T> pl(93, true, 1, &bad);
            pl.execute(in.data(), out.data(), one);
        }()));
    }
    // Veto shape: a big prime at ido >= W is admitted (the O(p^2) loss the ido<64
    // veto guarded is gone with the chiplet); ido < W stays out, half-idle tiles
    // still lose to bluestein. The shape is also priced out of the DP (2x the valley
    // tier), so the DP itself only elects supported placements: 9603's 97 lands at
    // ido=33 at W=16, while 776's only placements {2,4} stay out for W>4.
    {
        constexpr std::size_t W = xsimd::batch<T>::size;
        REQUIRE(P::route_available(R::iterative_dif, 9603u));
        REQUIRE(P::route_available(R::iterative_dif, 776u) == (W <= 4));
        impulse_flat<T>(9603, 2.0);
    }
}
