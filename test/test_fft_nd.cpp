#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/admiral.hpp>                 // forward / inverse / nd_plan

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

// ----------------------------------------------------------------------------
// Test helpers (kept self-contained: each test TU is its own executable).
// ----------------------------------------------------------------------------

// Relative error budget for an Ntot-point transform; same shape as the 1D
// suite (eps * sqrt(Ntot) * (log2(Ntot)+1) * 64 * scale), Ntot = product of
// extents for the separable N-D transform.
template<typename T>
T fft_tol(std::size_t Ntot, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(static_cast<T>(Ntot)) + T(1);
    return eps * std::sqrt(static_cast<T>(Ntot)) * log2N * T(64) * scale;
}

template<typename T>
T max_magnitude(const std::vector<std::complex<T>>& v) {
    T m = T(1);
    for (const auto& x : v) m = std::max(m, std::abs(x));
    return m;
}

template<typename T>
bool vectors_approx_equal(const std::vector<std::complex<T>>& a,
                          const std::vector<std::complex<T>>& b, T tolerance) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) return false;
    }
    return true;
}

template<typename T>
std::vector<std::complex<T>> make_input(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<T>> v(n);
    for (auto& z : v) z = std::complex<T>(static_cast<T>(dist(gen)), static_cast<T>(dist(gen)));
    return v;
}

template<std::size_t Dim>
std::size_t shape_product(const std::array<std::size_t, Dim>& s) {
    std::size_t n = 1;
    for (auto e : s) n *= e;
    return n;
}

template<typename T>
std::vector<T> make_real_input(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T> v(n);
    for (auto& x : v) x = static_cast<T>(dist(gen));
    return v;
}

template<typename T>
T max_magnitude_real(const std::vector<T>& v) {
    T m = T(1);
    for (auto x : v) m = std::max(m, std::abs(x));
    return m;
}

// Half-spectrum complex element count for a real shape.
template<std::size_t Dim>
std::size_t cplx_count(std::array<std::size_t, Dim> shape) {
    shape[Dim - 1] = shape[Dim - 1] / 2 + 1;
    std::size_t n = 1; for (auto e : shape) n *= e;
    return n;
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
        T phase = T(0);
        for (std::size_t d = Dim; d-- > 0; ) {
            const std::size_t nd = tmp % shape[d];
            tmp /= shape[d];
            phase += static_cast<T>(K[d]) * static_cast<T>(nd) / static_cast<T>(shape[d]);
        }
        const T angle = T(2) * std::numbers::pi_v<T> * phase;
        in[flat] = std::complex<T>(std::cos(angle), std::sin(angle));
    }

    auto out = in;
    admiral::forward<T, Dim>(out.data(), shape);

    // Row-major flat index of K.
    std::size_t K_flat = 0;
    for (std::size_t d = 0; d < Dim; ++d)
        K_flat = K_flat * shape[d] + K[d];

    const T Ntot = static_cast<T>(n);
    const T tol  = fft_tol<T>(n, Ntot);
    INFO("forward analytical Dim=" << Dim << " Ntot=" << n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::complex<T> expected =
            (i == K_flat) ? std::complex<T>(Ntot, T(0)) : std::complex<T>(T(0), T(0));
        REQUIRE(std::abs(out[i] - expected) <= tol);
    }
}

template<typename T, std::size_t Dim>
void check_roundtrip(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_input<T>(n, seed);

    auto out = in;
    admiral::forward<T, Dim>(out.data(), shape);
    admiral::inverse<T, Dim>(out.data(), shape);

    const T tol = fft_tol<T>(n, max_magnitude(in));
    INFO("round-trip Dim=" << Dim << " Ntot=" << n);
    REQUIRE(vectors_approx_equal(out, in, tol));
}

template<typename T, std::size_t Dim>
void check_parseval(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_input<T>(n, seed);

    auto out = in;
    admiral::forward<T, Dim>(out.data(), shape);

    double energy_in = 0.0, energy_out = 0.0;
    for (const auto& z : in) energy_in += std::norm(static_cast<std::complex<double>>(z));
    for (const auto& z : out) energy_out += std::norm(static_cast<std::complex<double>>(z));

    // Unnormalized forward: sum|X|^2 = Ntot * sum|x|^2.
    const double rel = (sizeof(T) == 4) ? 1e-3 : 1e-9;
    INFO("Parseval Dim=" << Dim << " Ntot=" << n);
    REQUIRE_THAT(energy_out, WithinRel(static_cast<double>(n) * energy_in, rel));
}

// ----------------------------------------------------------------------------
// r2c / c2r drivers: analytical checks and r2c -> c2r round-trip identity.
// ----------------------------------------------------------------------------

// Validates r2c analytically (no third-party oracle):
//   (a) Constant input 1 -> half-spectrum has value Ntot at DC (flat index 0),
//       ~0 at every other bin.
//   (b) r2c -> c2r round-trip identity on a random real input.
template<typename T, std::size_t Dim>
void check_r2c_analytical(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n   = shape_product(shape);
    const T           Ntot = static_cast<T>(n);

    // (a) DC check.
    {
        std::vector<T> dc_in(n, T(1));
        std::vector<std::complex<T>> dc_out(cplx_count(shape));
        admiral::forward<T, Dim>(dc_in.data(), dc_out.data(), shape);

        const T tol = fft_tol<T>(n, Ntot);
        INFO("r2c DC check Dim=" << Dim << " Ntot=" << n);
        REQUIRE(std::abs(dc_out[0] - std::complex<T>(Ntot, T(0))) <= tol);
        for (std::size_t i = 1; i < dc_out.size(); ++i)
            REQUIRE(std::abs(dc_out[i]) <= tol);
    }

    // (b) Round-trip identity.
    {
        const auto in = make_real_input<T>(n, seed);
        std::vector<std::complex<T>> spec(cplx_count(shape));
        admiral::forward<T, Dim>(in.data(), spec.data(), shape);
        std::vector<T> back(n);
        admiral::inverse<T, Dim>(spec.data(), back.data(), shape);

        const T tol = fft_tol<T>(n, max_magnitude_real(in));
        INFO("r2c round-trip Dim=" << Dim << " Ntot=" << n);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(std::abs(back[i] - in[i]) <= tol);
    }
}

template<typename T, std::size_t Dim>
void check_r2c_roundtrip(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);
    const auto in = make_real_input<T>(n, seed);

    std::vector<std::complex<T>> spec(cplx_count(shape));
    admiral::forward<T, Dim>(in.data(), spec.data(), shape);
    std::vector<T> back(n);
    admiral::inverse<T, Dim>(spec.data(), back.data(), shape);

    const T tol = fft_tol<T>(n, max_magnitude_real(in));
    INFO("r2c round-trip Dim=" << Dim << " Ntot=" << n);
    for (std::size_t i = 0; i < n; ++i) REQUIRE(std::abs(back[i] - in[i]) <= tol);
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
    check_parseval<T, 2>({16, 16}, 31);
    check_parseval<T, 2>({12, 20}, 32);
    check_parseval<T, 2>({32, 8}, 33);
}

TEMPLATE_TEST_CASE("2D FFT degenerate extents", "[nd][2d][edge]", float, double) {
    using T = TestType;
    // A length-1 axis is the identity along that axis.
    check_forward_analytical<T, 2>({1, 16}, 41);
    check_forward_analytical<T, 2>({16, 1}, 42);
    check_roundtrip<T, 2>({1, 13}, 43);
    check_roundtrip<T, 2>({13, 1}, 44);
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

// ----------------------------------------------------------------------------
// Dim=1: the new layer must reproduce the legacy 1D path.
// ----------------------------------------------------------------------------

TEMPLATE_TEST_CASE("1D via nd_plan matches legacy 1D FFT", "[nd][1d]", float, double) {
    using T = TestType;
    for (std::size_t N : {std::size_t{8}, std::size_t{16}, std::size_t{7},
                          std::size_t{13}, std::size_t{31}, std::size_t{60}}) {
        const auto in = make_input<T>(N, static_cast<unsigned>(N + 100));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(std::span<const std::complex<T>>(in), std::span(legacy));

        auto nd = in;
        admiral::forward<T, 1>(nd.data(), {N});

        const T tol = fft_tol<T>(N, max_magnitude(legacy));
        INFO("Dim=1 N=" << N);
        REQUIRE(vectors_approx_equal(nd, legacy, tol));
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
    check_parseval<T, 3>({8, 8, 8}, 101);
    check_parseval<T, 3>({6, 10, 4}, 102);
    check_parseval<T, 3>({5, 5, 5}, 103);
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
    check_r2c_analytical<T, 1>({64}, 201);
    check_r2c_analytical<T, 1>({30}, 202);
    check_r2c_analytical<T, 1>({15}, 203);   // odd -> fallback
    // 2D
    check_r2c_analytical<T, 2>({8, 16}, 204);
    check_r2c_analytical<T, 2>({12, 20}, 205);
    check_r2c_analytical<T, 2>({8, 15}, 206);   // odd inner -> fallback
    // WS-B batched even path: rows not a multiple of the SIMD width W exercise the
    // tile loop + the <W scalar tail (rows=6: f64 W=4 tail=2, f32 rows<8 all-tail;
    // rows=14: f64 tail=2, f32 W=8 tail=6), with a mixed-radix inner M.
    check_r2c_analytical<T, 2>({6, 16}, 220);
    check_r2c_analytical<T, 2>({14, 24}, 221);  // M=12 = 4*3
    // 3D
    check_r2c_analytical<T, 3>({4, 6, 8}, 207);
    check_r2c_analytical<T, 3>({5, 4, 6}, 208);
    check_r2c_analytical<T, 3>({4, 4, 9}, 209);  // odd inner -> fallback
}

TEMPLATE_TEST_CASE("r2c -> c2r round-trip identity", "[nd][r2c][roundtrip]", float, double) {
    using T = TestType;
    check_r2c_roundtrip<T, 1>({64}, 211);
    check_r2c_roundtrip<T, 1>({15}, 212);      // odd
    check_r2c_roundtrip<T, 2>({16, 16}, 213);
    check_r2c_roundtrip<T, 2>({12, 20}, 214);
    check_r2c_roundtrip<T, 2>({9, 8}, 215);    // odd outer (even inner)
    check_r2c_roundtrip<T, 2>({8, 9}, 216);    // odd inner -> fallback
    check_r2c_roundtrip<T, 2>({6, 16}, 222);   // WS-B W-tail (see r2c-vs-analytical above)
    check_r2c_roundtrip<T, 2>({14, 24}, 223);
    check_r2c_roundtrip<T, 3>({4, 6, 8}, 217);
    check_r2c_roundtrip<T, 3>({4, 4, 7}, 218); // odd inner -> fallback
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
        admiral::forward<T, 2>(a.data(), shape);

        auto b = in;
        const std::size_t dims[2] = {shape[0], shape[1]};
        admiral::detail::nd_runtime_plan<T>(std::span<const std::size_t>(dims, 2), /*forward=*/true).execute(b.data());

        const T tol = fft_tol<T>(n, max_magnitude(a));
        REQUIRE(vectors_approx_equal(a, b, tol));
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
        admiral::forward<T, 2>(ref.data(), shape);
        REQUIRE(vectors_approx_equal(a, ref, fft_tol<T>(n, max_magnitude(ref))));

        p.inverse(a.data());  // round-trip via the same plan
        REQUIRE(vectors_approx_equal(a, in, fft_tol<T>(n, max_magnitude(in))));
    };
    check_2d({16, 16}, 71);
    check_2d({12, 20}, 72);
    check_2d({8, 8}, 73);

    // 1D through the same unified type still reproduces the legacy 1D path.
    {
        const std::size_t N = 64;
        const auto in = make_input<T>(N, 74);

        admiral::plan<T> p(N);
        auto a = in;
        p.forward(std::span<std::complex<T>>(a.data(), N));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(std::span<const std::complex<T>>(in), std::span(legacy));
        REQUIRE(vectors_approx_equal(a, legacy, fft_tol<T>(N, max_magnitude(legacy))));
    }
}
