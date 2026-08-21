#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>                 // forward / inverse / plan
#include <admiral/detail/nd_plan.hpp>          // nd_runtime_plan, tested directly
#include <admiral/detail/simd_swizzle.hpp>    // real_run_copy, driven here directly

#include <array>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <vector>

using namespace Catch::Matchers;

// Half-spectrum complex element count for a real shape. Derived here rather than read
// back from plan_r2c::cplx_size(), so the expected layout stays independent of the code
// under test.
template<std::size_t Dim>
std::size_t cplx_count(std::array<std::size_t, Dim> shape) {
    shape[Dim - 1] = shape[Dim - 1] / 2 + 1;
    return shape_product(shape);
}

// ----------------------------------------------------------------------------
// Forward analytical / round-trip / Parseval drivers.
// ----------------------------------------------------------------------------

// Checks the N-D forward transform against an exact analytical reference.
//
// Input: separable single-tone complex exponential
//   x[n_0,...,n_{D-1}] = exp(+2*pi*i * sum_d (K_d * n_d / N_d))
// with K_d chosen deterministically from `seed`.
//
// Analytical result: zero everywhere except a spike of magnitude Ntot at the
// row-major flat index of the bin tuple (K_0,...,K_{D-1}).
template<typename T, std::size_t Dim>
void check_forward_analytical(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);

    // Per-axis bin: 0 when N_d==1, else (seed+1+d) % N_d.
    std::array<std::size_t, Dim> K;
    for (std::size_t d = 0; d < Dim; ++d)
        K[d] = (shape[d] > 1) ? ((seed + 1u + static_cast<unsigned>(d)) % shape[d]) : 0u;

    // Build the single-tone exponential.
    std::vector<std::complex<T>> in(n);
    for (std::size_t flat = 0; flat < n; ++flat) {
        std::size_t tmp = flat;
        long double turns = 0.0L;
        for (std::size_t d = Dim; d-- > 0; ) {
            const std::size_t nd = tmp % shape[d];
            tmp /= shape[d];
            turns += turn_fraction(K[d], nd, shape[d]);
        }
        in[flat] = unit_phasor<T>(turns);
    }

    auto out = in;
    admiral::forward(out.data(), shape);

    // Row-major flat index of K.
    std::size_t K_flat = 0;
    for (std::size_t d = 0; d < Dim; ++d)
        K_flat = K_flat * shape[d] + K[d];

    // One tone: every bin zero but K_flat, which holds n. A per-element absolute
    // check would need a tol*n fudge for that bin; relative L2 self-normalizes.
    std::vector<std::complex<T>> expected(n, std::complex<T>(T(0), T(0)));
    expected[K_flat] = std::complex<T>(static_cast<T>(n), T(0));
    INFO("forward analytical Dim=" << Dim << " Ntot=" << n);
    require_close(out, expected, fft_tol<T>());
}

template<typename T, std::size_t Dim>
void check_roundtrip(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_input<T>(n, seed);

    auto out = in;
    admiral::forward(out.data(), shape);
    admiral::inverse(out.data(), shape);

    const double tol = fft_tol<T>();
    INFO("round-trip Dim=" << Dim << " Ntot=" << n);
    require_close(out, in, tol);
}

template<typename T, std::size_t Dim>
void check_parseval(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_input<T>(n, seed);

    auto out = in;
    admiral::forward(out.data(), shape);

    INFO("Dim=" << Dim);
    require_parseval<T>(in, out, n);
}

// ----------------------------------------------------------------------------
// r2c / c2r drivers: analytical checks and r2c -> c2r round-trip identity.
// ----------------------------------------------------------------------------

// Constant input 1 -> half-spectrum holds Ntot at DC (flat index 0), zero elsewhere.
// No third-party oracle needed.
template<typename T, std::size_t Dim>
void check_r2c_analytical(std::array<std::size_t, Dim> shape) {
    const std::size_t n = shape_product(shape);
    std::vector<T> in(n, T(1));
    std::vector<std::complex<T>> out(cplx_count(shape));
    admiral::forward(in.data(), out.data(), shape);

    std::vector<std::complex<T>> expected(out.size(), std::complex<T>(T(0), T(0)));
    expected[0] = std::complex<T>(static_cast<T>(n), T(0));
    INFO("r2c DC check Dim=" << Dim << " Ntot=" << n);
    require_close(out, expected, fft_tol<T>());
}

template<typename T, std::size_t Dim>
void check_r2c_roundtrip(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_real_input<T>(n, seed);

    std::vector<std::complex<T>> spec(cplx_count(shape));
    admiral::forward(in.data(), spec.data(), shape);
    std::vector<T> back(n);
    admiral::inverse(spec.data(), back.data(), shape);

    INFO("r2c round-trip Dim=" << Dim << " Ntot=" << n);
    require_close(back, in, fft_tol<T>());
}

// ----------------------------------------------------------------------------
// 2D correctness vs analytical.
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("2D FFT vs analytical (forward)", "[nd][2d]", float, double) {
    using T = TestType;
    // pow2 x pow2
    check_forward_analytical<T, 2>({8, 8}, 1);
    check_forward_analytical<T, 2>({16, 32}, 2);
    check_forward_analytical<T, 2>({64, 64}, 3);
    // 7-smooth x 7-smooth (exercises radix 3/5/7 column passes)
    check_forward_analytical<T, 2>({6, 9}, 4);
    check_forward_analytical<T, 2>({12, 15}, 5);
    check_forward_analytical<T, 2>({20, 35}, 6);
    // square
    check_forward_analytical<T, 2>({7, 7}, 7);
    check_forward_analytical<T, 2>({32, 32}, 8);
    // 11-smooth
    check_forward_analytical<T, 2>({11, 22}, 9);
}

TEMPLATE_TEST_CASE("2D FFT mixed/fallback column routes vs analytical", "[nd][2d][fallback]",
                   float, double) {
    using T = TestType;
    // Catalog-prime column (17): non-7-smooth -> scalar fallback (codelet route).
    check_forward_analytical<T, 2>({17, 8}, 11);
    check_forward_analytical<T, 2>({8, 17}, 12);
    // Small non-smooth (19): scalar fallback (direct DFT route).
    check_forward_analytical<T, 2>({19, 8}, 13);
    check_forward_analytical<T, 2>({8, 19}, 14);
    // 31: scalar fallback (Bluestein route) as outer and as inner axis.
    check_forward_analytical<T, 2>({31, 16}, 15);
    check_forward_analytical<T, 2>({16, 31}, 16);
}

TEMPLATE_TEST_CASE("2D FFT round-trip identity", "[nd][2d][roundtrip]", float, double) {
    using T = TestType;
    check_roundtrip<T, 2>({8, 8}, 21);
    check_roundtrip<T, 2>({16, 32}, 22);
    check_roundtrip<T, 2>({12, 15}, 23);
    check_roundtrip<T, 2>({17, 19}, 24);
    check_roundtrip<T, 2>({31, 9}, 25);
    check_roundtrip<T, 2>({64, 64}, 26);
}

TEMPLATE_TEST_CASE("2D FFT Parseval", "[nd][2d][parseval]", float, double) {
    using T = TestType;
    // Parseval only sees energy, so it cannot catch a permuted, conjugated or
    // sign-flipped spectrum. Every shape checked here also gets the analytical
    // known-answer, so none of them rests on the energy identity alone.
    for (const std::array<std::size_t, 2> shape : {std::array<std::size_t, 2>{16, 16},
                                                   {12, 20},
                                                   {32, 8}}) {
        const unsigned seed = 31 + unsigned(shape[1]);
        check_parseval<T, 2>(shape, seed);
        check_forward_analytical<T, 2>(shape, seed);
    }
}

TEMPLATE_TEST_CASE("2D FFT degenerate extents", "[nd][2d][edge]", float, double) {
    using T = TestType;
    // A length-1 axis is the identity along that axis.
    check_forward_analytical<T, 2>({1, 16}, 41);
    check_forward_analytical<T, 2>({16, 1}, 42);
    check_roundtrip<T, 2>({1, 13}, 43);
    check_roundtrip<T, 2>({13, 1}, 44);
}

TEMPLATE_TEST_CASE("rank-0 plan is the identity", "[nd][edge]", float, double) {
    using T = TestType;
    using C = std::complex<T>;
    // An empty shape is one element: the identity, with fct applied if given.
    const admiral::plan<T> p(std::array<std::size_t, 0>{});
    REQUIRE(p.size() == 1);

    C a{T(2), T(-3)};
    p.forward(&a);
    REQUIRE(a == C{T(2), T(-3)});

    C dst{T(99), T(99)};
    p.forward(&a, &dst);  // out-of-place must write dst, not leave it stale
    REQUIRE(dst == C{T(2), T(-3)});

    p.inverse(&a, &dst, T(0.25));
    REQUIRE(dst == C{T(0.5), T(-0.75)});
    p.forward(&a, T(0.25));
    REQUIRE(a == C{T(0.5), T(-0.75)});
}

// ----------------------------------------------------------------------------
// Large-outer / small-inner: the outer axis takes the gather -> row-FFT ->
// scatter (small_inner) path when the contiguous inner batch is too narrow or
// non-W-aligned. Must match analytical and round-trip for both precisions.
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("2D large-outer/small-inner vs analytical", "[nd][2d][small-inner]",
                   float, double) {
    using T = TestType;
    check_forward_analytical<T, 2>({256, 16}, 301);   // narrow aligned inner
    check_forward_analytical<T, 2>({1024, 64}, 302);  // few column blocks
    check_forward_analytical<T, 2>({60, 60}, 303);    // non-W-multiple inner (60%8=4)
    check_forward_analytical<T, 2>({128, 20}, 304);   // non-W-multiple inner (20%8=4)
    check_roundtrip<T, 2>({256, 16}, 311);
    check_roundtrip<T, 2>({60, 60}, 312);
    check_roundtrip<T, 2>({1024, 64}, 313);
}

// Live cover for the elected 16384 chain on the COLUMN driver: a
// strided non-innermost axis executes 8-16-16-8 there, which the row-form harness
// cannot see.
//
// Double only: both entries are keyed on batch<double>::size==8.
TEST_CASE("2D 16384-long strided f64 axis vs analytical", "[nd][2d]") {
    check_forward_analytical<double, 2>({16384, 4}, 321);
    check_roundtrip<double, 2>({16384, 4}, 322);
}

// ----------------------------------------------------------------------------
// Dim=1: nd_plan must reproduce the legacy 1D path.
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("1D via nd_plan matches legacy 1D FFT", "[nd][1d]", float, double) {
    using T = TestType;
    for (std::size_t N : {std::size_t{8}, std::size_t{16}, std::size_t{7},
                          std::size_t{13}, std::size_t{31}, std::size_t{60}}) {
        const auto in = make_input<T>(N, static_cast<unsigned>(N + 100));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(std::span<const std::complex<T>>(in), std::span(legacy));

        auto nd = in;
        admiral::forward(nd.data(), {N});

        const double tol = fft_tol<T>();
        INFO("Dim=1 N=" << N);
        require_close(nd, legacy, tol);
    }
}

// ----------------------------------------------------------------------------
// Dim=3 smoke (vs analytical + round-trip).
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("3D FFT smoke vs analytical", "[nd][3d]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 3>({4, 4, 4}, 51);
    check_forward_analytical<T, 3>({2, 3, 5}, 52);
    check_forward_analytical<T, 3>({8, 4, 2}, 53);
    check_roundtrip<T, 3>({4, 5, 6}, 54);
    check_roundtrip<T, 3>({8, 8, 8}, 55);
}

// 3D hardening: Parseval, fallback-route outer/middle/inner axes, degenerate.
TEMPLATE_TEST_CASE("3D FFT Parseval", "[nd][3d][parseval]", float, double) {
    using T = TestType;
    for (const std::array<std::size_t, 3> shape : {std::array<std::size_t, 3>{8, 8, 8},
                                                   {6, 10, 4},
                                                   {5, 5, 5}}) {
        const unsigned seed = 101 + unsigned(shape[2]);
        check_parseval<T, 3>(shape, seed);
        check_forward_analytical<T, 3>(shape, seed);
    }
}

TEMPLATE_TEST_CASE("3D FFT fallback-route axes vs analytical", "[nd][3d][fallback]", float, double) {
    using T = TestType;
    // Non-7-smooth axis in each position -> scalar fallback column/row pass.
    check_forward_analytical<T, 3>({17, 8, 8}, 111);  // outer catalog-prime (codelet)
    check_forward_analytical<T, 3>({8, 31, 8}, 112);  // middle Bluestein axis
    check_forward_analytical<T, 3>({8, 8, 19}, 113);  // inner direct-DFT axis
    check_roundtrip<T, 3>({31, 8, 9}, 114);           // mixed fallback round-trip
}

TEMPLATE_TEST_CASE("3D FFT degenerate extents", "[nd][3d][edge]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 3>({1, 8, 8}, 121);
    check_forward_analytical<T, 3>({8, 1, 8}, 122);
    check_forward_analytical<T, 3>({8, 8, 1}, 123);
    check_roundtrip<T, 3>({1, 13, 1}, 124);
}

// ----------------------------------------------------------------------------
// Dim=4 smoke (vs analytical + round-trip).
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("4D FFT smoke vs analytical", "[nd][4d]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 4>({4, 4, 4, 4}, 131);
    check_roundtrip<T, 4>({2, 3, 4, 5}, 132);
}

// ----------------------------------------------------------------------------
// r2c / c2r: analytical forward check and round-trip identity, 1D/2D/3D, both
// precisions, even and odd innermost N (odd exercises the full-c2c fallback).
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("r2c forward vs analytical", "[nd][r2c]", float, double) {
    using T = TestType;
    // 1D
    check_r2c_analytical<T, 1>({64});
    check_r2c_analytical<T, 1>({30});
    check_r2c_analytical<T, 1>({15});   // odd -> fallback
    // 2D
    check_r2c_analytical<T, 2>({8, 16});
    check_r2c_analytical<T, 2>({12, 20});
    check_r2c_analytical<T, 2>({8, 15});   // odd inner -> fallback
    // WS-B batched even path: rows not a multiple of the SIMD width W exercise the
    // tile loop + the <W scalar tail (rows=6: f64 W=4 tail=2, f32 rows<8 all-tail;
    // rows=14: f64 tail=2, f32 W=8 tail=6), with a mixed-radix inner M.
    check_r2c_analytical<T, 2>({6, 16});
    check_r2c_analytical<T, 2>({14, 24});  // M=12 = 4*3
    // 3D
    check_r2c_analytical<T, 3>({4, 6, 8});
    check_r2c_analytical<T, 3>({5, 4, 6});
    check_r2c_analytical<T, 3>({4, 4, 9});  // odd inner -> fallback
}

TEMPLATE_TEST_CASE("r2c -> c2r round-trip identity", "[nd][r2c][roundtrip]", float, double) {
    using T = TestType;
    check_r2c_roundtrip<T, 1>({64}, 211);
    check_r2c_roundtrip<T, 1>({30}, 219);
    check_r2c_roundtrip<T, 1>({15}, 212);      // odd
    check_r2c_roundtrip<T, 2>({16, 16}, 213);
    check_r2c_roundtrip<T, 2>({8, 16}, 224);
    check_r2c_roundtrip<T, 2>({12, 20}, 214);
    check_r2c_roundtrip<T, 2>({9, 8}, 215);    // odd outer (even inner)
    check_r2c_roundtrip<T, 2>({8, 9}, 216);    // odd inner -> fallback
    check_r2c_roundtrip<T, 2>({8, 15}, 225);   // odd inner -> fallback
    check_r2c_roundtrip<T, 2>({6, 16}, 222);   // WS-B W-tail (see r2c-vs-analytical above)
    check_r2c_roundtrip<T, 2>({14, 24}, 223);
    check_r2c_roundtrip<T, 3>({4, 6, 8}, 217);
    check_r2c_roundtrip<T, 3>({5, 4, 6}, 226);
    check_r2c_roundtrip<T, 3>({4, 4, 7}, 218); // odd inner -> fallback
    check_r2c_roundtrip<T, 3>({4, 4, 9}, 227); // odd inner -> fallback
}

// ----------------------------------------------------------------------------
// Runtime-rank dispatch (nd_runtime_plan) agrees with the compile-time API.
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Runtime nd_execute matches compile-time nd_plan", "[nd][dispatch]",
                   float, double) {
    using T = TestType;
    auto run = [](std::array<std::size_t, 2> shape, unsigned seed) {
        const std::size_t n = shape[0] * shape[1];
        const auto in = make_input<T>(n, seed);

        auto a = in;
        admiral::forward(a.data(), shape);

        auto b = in;
        const std::size_t dims[2] = {shape[0], shape[1]};
        admiral::detail::nd_runtime_plan<T>(std::span<const std::size_t>(dims, 2), /*forward=*/true).execute(b.data());

        const double tol = fft_tol<T>();
        require_close(a, b, tol);
    };
    run({16, 16}, 61);
    run({12, 20}, 62);
    run({31, 8}, 63);
}

// ----------------------------------------------------------------------------
// Unified admiral::plan<T> handles every rank from a runtime shape (no Dim param).
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("Unified plan<T> handles N-D via runtime shape", "[nd][plan]",
                   float, double) {
    using T = TestType;

    auto check_2d = [](std::array<std::size_t, 2> shape, unsigned seed) {
        const std::size_t n = shape[0] * shape[1];
        const auto in = make_input<T>(n, seed);

        admiral::plan<T> p(shape);  // reusable, bidirectional, built from a runtime shape

        auto a = in;
        p.forward(a.data());

        // Reference: compile-time forward (verified analytically by check_forward_analytical).
        auto ref = in;
        admiral::forward(ref.data(), shape);
        require_close(a, ref, fft_tol<T>());

        p.inverse(a.data());  // round-trip via the same plan
        require_close(a, in, fft_tol<T>());
    };
    check_2d({16, 16}, 71);
    check_2d({12, 20}, 72);
    check_2d({8, 8}, 73);

    // 1D through the same unified type must reproduce the legacy 1D path.
    {
        const std::size_t N = 64;
        const auto in = make_input<T>(N, 74);

        admiral::plan<T> p(N);
        auto a = in;
        p.forward(std::span<std::complex<T>>(a.data(), N));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(std::span<const std::complex<T>>(in), std::span(legacy));
        require_close(a, legacy, fft_tol<T>());
    }
}

// The narrow-run criterion of choose_line_route is a pure function of the shape, but no
// tensor a portable test can afford elects it. It needs a slab past L2 behind a run of
// one or two elements. This case drives it directly, with the budget read from the same
// helper the criterion uses, so the sizes hold on any cache geometry.
TEMPLATE_TEST_CASE("choose_line_route narrow-run criterion", "[nd][route]", float, double) {
    using T = TestType;
    using admiral::detail::choose_line_route;
    using admiral::detail::line_route;

    admiral::detail::nd_axis_state<T> st;
    st.dif = true;  // a column chain exists, so both routes are available

    const std::size_t budget = admiral::detail::col_cache_budget(1);
    const std::size_t len = 64;
    const std::size_t wide = budget / (len * sizeof(std::complex<T>)) + 1;
    REQUIRE(len * wide * sizeof(std::complex<T>) > budget);

    // 2*run_len <= batch size for every ISA: batch<T>::size is at least 2.
    constexpr std::size_t run_len = 1;
    CHECK(choose_line_route<T>(st, len, wide, run_len, 1) == line_route::transposed);
    // Same run, a slab that fits: the criterion must be able to answer the other way,
    // or the check above would pass on a function that always returns transposed.
    CHECK(choose_line_route<T>(st, len, 1, run_len, 1) == line_route::col_dif);
    // A run wide enough to fill the register keeps the column route even uncached.
    const std::size_t full = xsimd::batch<T>::size;
    CHECK(choose_line_route<T>(st, len, wide, full, 1) == line_route::col_dif);
    // No column chain: transposed is the only thing left, whatever the shape says.
    st.dif = false;
    CHECK(choose_line_route<T>(st, len, 1, full, 1) == line_route::transposed);
}

// real_run_copy is nd_plan's band-pack helper. Its `full` branch (n >= W, one
// unmasked batch ahead of the masked remainder) runs only for a band at least a
// register wide; the only call site passes 2*band_width. This case sweeps every
// length the type admits rather than spot-checking, because which branch a given n
// takes follows W and therefore the build.
TEMPLATE_TEST_CASE("real_run_copy copies exactly its run", "[nd][swizzle]", float, double) {
    using T = TestType;
    constexpr std::size_t W = xsimd::batch<T>::size;

    // 3*W of source so the masked load past a leading full batch always has lanes to
    // read, even though it only touches the masked ones.
    std::vector<T> src(3 * W);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = T(i + 1);

    bool saw_full = false, saw_masked = false;
    for (std::size_t n = 0; n <= 2 * W; ++n) {
        CAPTURE(n, W);
        const auto cp = admiral::detail::real_run_copy<T>::make(n);
        saw_full |= cp.full;
        saw_masked |= cp.masked;
        REQUIRE(cp.full == (n >= W));

        std::vector<T> dst(3 * W, T(-1));
        cp(src.data(), dst.data());
        for (std::size_t i = 0; i < n; ++i) REQUIRE(dst[i] == src[i]);
        // Nothing past the run may move: a mask that is too wide would show up here and
        // in the real caller it would corrupt the neighbouring band.
        for (std::size_t i = n; i < dst.size(); ++i) REQUIRE(dst[i] == T(-1));
    }
    REQUIRE(saw_full);     // the sweep must reach both branches, or it proves nothing
    REQUIRE(saw_masked);
}

// The measuring efforts race candidate routes at plan time. test_plan.cpp covers that
// for 1-D; N-D reaches it once per axis through a different planner (nd_runtime_plan),
// which is where a per-axis race would go wrong without showing up in 1-D.
TEMPLATE_TEST_CASE("N-D plans honour the measuring efforts", "[nd][effort]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{40, 24};
    const std::size_t Ntot = shape[0] * shape[1];
    const auto in = make_input<T>(Ntot, 0xE770u);

    for (const admiral::effort eff : {admiral::effort::automatic, admiral::effort::measure}) {
        CAPTURE(int(eff));
        const admiral::plan<T> p(shape, {.eff = eff});
        std::vector<std::complex<T>> v = in;
        p.forward(v.data(), v.data());
        p.inverse(v.data(), v.data());
        require_close(v, in, fft_tol<T>());
    }
}
