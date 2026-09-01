#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/nd_plan.hpp>
#include <admiral/detail/simd_swizzle.hpp>

#include <array>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using namespace Catch::Matchers;

template<std::size_t Dim>
std::size_t cplx_count(std::array<std::size_t, Dim> shape) {
    shape[Dim - 1] = shape[Dim - 1] / 2 + 1;
    return shape_product(shape);
}

template<typename T, std::size_t Dim>
void check_forward_analytical(std::array<std::size_t, Dim> shape, unsigned seed) {
    const std::size_t n = shape_product(shape);

    std::array<std::size_t, Dim> K;
    for (std::size_t d = 0; d < Dim; ++d)
        K[d] = (shape[d] > 1) ? ((seed + 1u + static_cast<unsigned>(d)) % shape[d]) : 0u;

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

    std::size_t K_flat = 0;
    for (std::size_t d = 0; d < Dim; ++d)
        K_flat = K_flat * shape[d] + K[d];

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

TEMPLATE_TEST_CASE("2D FFT vs analytical (forward)", "[nd][2d]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 2>({8, 8}, 1);
    check_forward_analytical<T, 2>({16, 32}, 2);
    check_forward_analytical<T, 2>({64, 64}, 3);
    check_forward_analytical<T, 2>({6, 9}, 4);
    check_forward_analytical<T, 2>({12, 15}, 5);
    check_forward_analytical<T, 2>({20, 35}, 6);
    check_forward_analytical<T, 2>({7, 7}, 7);
    check_forward_analytical<T, 2>({32, 32}, 8);
    check_forward_analytical<T, 2>({11, 22}, 9);
}

TEMPLATE_TEST_CASE("2D FFT mixed/fallback column routes vs analytical", "[nd][2d][fallback]",
                   float, double) {
    using T = TestType;
    check_forward_analytical<T, 2>({17, 8}, 11);
    check_forward_analytical<T, 2>({8, 17}, 12);
    check_forward_analytical<T, 2>({19, 8}, 13);
    check_forward_analytical<T, 2>({8, 19}, 14);
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
    check_forward_analytical<T, 2>({1, 16}, 41);
    check_forward_analytical<T, 2>({16, 1}, 42);
    check_roundtrip<T, 2>({1, 13}, 43);
    check_roundtrip<T, 2>({13, 1}, 44);
}

TEMPLATE_TEST_CASE("rank-0 plan is the identity", "[nd][edge]", float, double) {
    using T = TestType;
    using C = std::complex<T>;
    admiral::plan<T> p(std::array<std::size_t, 0>{});
    REQUIRE(p.size() == 1);

    C a{T(2), T(-3)};
    p.forward(&a);
    REQUIRE(a == C{T(2), T(-3)});

    C dst{T(99), T(99)};
    p.forward(&a, &dst);
    REQUIRE(dst == C{T(2), T(-3)});

    p.inverse(&a, &dst, T(0.25));
    REQUIRE(dst == C{T(0.5), T(-0.75)});
    p.forward(&a, T(0.25));
    REQUIRE(a == C{T(0.5), T(-0.75)});
}

TEMPLATE_TEST_CASE("2D large-outer/small-inner vs analytical", "[nd][2d][small-inner]",
                   float, double) {
    using T = TestType;
    check_forward_analytical<T, 2>({256, 16}, 301);
    check_forward_analytical<T, 2>({1024, 64}, 302);
    check_forward_analytical<T, 2>({60, 60}, 303);
    check_forward_analytical<T, 2>({128, 20}, 304);
    check_roundtrip<T, 2>({256, 16}, 311);
    check_roundtrip<T, 2>({60, 60}, 312);
    check_roundtrip<T, 2>({1024, 64}, 313);
}

TEST_CASE("2D 16384-long strided f64 axis vs analytical", "[nd][2d]") {
    check_forward_analytical<double, 2>({16384, 4}, 321);
    check_roundtrip<double, 2>({16384, 4}, 322);
}

TEMPLATE_TEST_CASE("1D via nd_plan matches legacy 1D FFT", "[nd][1d]", float, double) {
    using T = TestType;
    for (std::size_t N : {std::size_t{8}, std::size_t{16}, std::size_t{7},
                          std::size_t{13}, std::size_t{31}, std::size_t{60}}) {
        const auto in = make_input<T>(N, static_cast<unsigned>(N + 100));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(admiral::span<const std::complex<T>>(in), admiral::span(legacy));

        auto nd = in;
        admiral::forward(nd.data(), {N});

        const double tol = fft_tol<T>();
        INFO("Dim=1 N=" << N);
        require_close(nd, legacy, tol);
    }
}

TEMPLATE_TEST_CASE("3D FFT smoke vs analytical", "[nd][3d]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 3>({4, 4, 4}, 51);
    check_forward_analytical<T, 3>({2, 3, 5}, 52);
    check_forward_analytical<T, 3>({8, 4, 2}, 53);
    check_roundtrip<T, 3>({4, 5, 6}, 54);
    check_roundtrip<T, 3>({8, 8, 8}, 55);
}

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
    check_forward_analytical<T, 3>({17, 8, 8}, 111);
    check_forward_analytical<T, 3>({8, 31, 8}, 112);
    check_forward_analytical<T, 3>({8, 8, 19}, 113);
    check_roundtrip<T, 3>({31, 8, 9}, 114);
}

TEMPLATE_TEST_CASE("3D FFT degenerate extents", "[nd][3d][edge]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 3>({1, 8, 8}, 121);
    check_forward_analytical<T, 3>({8, 1, 8}, 122);
    check_forward_analytical<T, 3>({8, 8, 1}, 123);
    check_roundtrip<T, 3>({1, 13, 1}, 124);
}

TEMPLATE_TEST_CASE("4D FFT smoke vs analytical", "[nd][4d]", float, double) {
    using T = TestType;
    check_forward_analytical<T, 4>({4, 4, 4, 4}, 131);
    check_roundtrip<T, 4>({2, 3, 4, 5}, 132);
}

TEMPLATE_TEST_CASE("r2c forward vs analytical", "[nd][r2c]", float, double) {
    using T = TestType;
    check_r2c_analytical<T, 1>({64});
    check_r2c_analytical<T, 1>({30});
    check_r2c_analytical<T, 1>({15});
    check_r2c_analytical<T, 2>({8, 16});
    check_r2c_analytical<T, 2>({12, 20});
    check_r2c_analytical<T, 2>({8, 15});
    check_r2c_analytical<T, 2>({6, 16});
    check_r2c_analytical<T, 2>({14, 24});
    check_r2c_analytical<T, 3>({4, 6, 8});
    check_r2c_analytical<T, 3>({5, 4, 6});
    check_r2c_analytical<T, 3>({4, 4, 9});
}

TEMPLATE_TEST_CASE("r2c -> c2r round-trip identity", "[nd][r2c][roundtrip]", float, double) {
    using T = TestType;
    check_r2c_roundtrip<T, 1>({64}, 211);
    check_r2c_roundtrip<T, 1>({30}, 219);
    check_r2c_roundtrip<T, 1>({15}, 212);
    check_r2c_roundtrip<T, 2>({16, 16}, 213);
    check_r2c_roundtrip<T, 2>({8, 16}, 224);
    check_r2c_roundtrip<T, 2>({12, 20}, 214);
    check_r2c_roundtrip<T, 2>({9, 8}, 215);
    check_r2c_roundtrip<T, 2>({8, 9}, 216);
    check_r2c_roundtrip<T, 2>({8, 15}, 225);
    check_r2c_roundtrip<T, 2>({6, 16}, 222);
    check_r2c_roundtrip<T, 2>({14, 24}, 223);
    check_r2c_roundtrip<T, 3>({4, 6, 8}, 217);
    check_r2c_roundtrip<T, 3>({5, 4, 6}, 226);
    check_r2c_roundtrip<T, 3>({4, 4, 7}, 218);
    check_r2c_roundtrip<T, 3>({4, 4, 9}, 227);
}

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
        admiral::detail::nd_runtime_plan<T>(admiral::span<const std::size_t>(dims, 2), true).execute(b.data());

        const double tol = fft_tol<T>();
        require_close(a, b, tol);
    };
    run({16, 16}, 61);
    run({12, 20}, 62);
    run({31, 8}, 63);
}

TEMPLATE_TEST_CASE("Unified plan<T> handles N-D via runtime shape", "[nd][plan]",
                   float, double) {
    using T = TestType;

    auto check_2d = [](std::array<std::size_t, 2> shape, unsigned seed) {
        const std::size_t n = shape[0] * shape[1];
        const auto in = make_input<T>(n, seed);

        admiral::plan<T> p(shape);

        auto a = in;
        p.forward(a.data());

        auto ref = in;
        admiral::forward(ref.data(), shape);
        require_close(a, ref, fft_tol<T>());

        p.inverse(a.data());
        require_close(a, in, fft_tol<T>());
    };
    check_2d({16, 16}, 71);
    check_2d({12, 20}, 72);
    check_2d({8, 8}, 73);

    {
        const std::size_t N = 64;
        const auto in = make_input<T>(N, 74);

        admiral::plan<T> p(N);
        auto a = in;
        p.forward(admiral::span<std::complex<T>>(a.data(), N));

        std::vector<std::complex<T>> legacy(N);
        admiral::forward<T>(admiral::span<const std::complex<T>>(in), admiral::span(legacy));
        require_close(a, legacy, fft_tol<T>());
    }
}

TEMPLATE_TEST_CASE("choose_line_route narrow-run criterion", "[nd][route]", float, double) {
    using T = TestType;
    using admiral::detail::choose_line_route;
    using admiral::detail::line_route;

    admiral::detail::nd_axis_state<T> st{};
    st.dif = true;

    const std::size_t budget = admiral::detail::col_cache_budget(1);
    const std::size_t len = 64;
    const std::size_t wide = budget / (len * sizeof(std::complex<T>)) + 1;
    REQUIRE(len * wide * sizeof(std::complex<T>) > budget);

    constexpr std::size_t run_len = 1;
    CHECK(choose_line_route<T>(st, len, wide, run_len, 1) == line_route::transposed);
    CHECK(choose_line_route<T>(st, len, 1, run_len, 1) == line_route::col_dif);
    const std::size_t full = xsimd::batch<T>::size;
    CHECK(choose_line_route<T>(st, len, wide, full, 1) == line_route::col_dif);
    st.dif = false;
    CHECK(choose_line_route<T>(st, len, 1, full, 1) == line_route::transposed);
    st.dif = true;

    constexpr std::size_t W = xsimd::batch<T>::size;
    using admiral::detail::col_budget_block;
    std::size_t len_coll = 2;
    while (col_budget_block<T>(len_coll, 1) >= 2 * W) len_coll *= 2;
    CAPTURE(len_coll);
    CHECK(choose_line_route<T>(st, len_coll, 1, 2 * W, 1) == line_route::transposed);
    if (len_coll > 2)
        CHECK(choose_line_route<T>(st, len_coll / 2, 1, 2 * W, 1) == line_route::col_dif);
    CHECK(choose_line_route<T>(st, len_coll, 1, 2 * W, 2) == line_route::col_dif);
}

TEMPLATE_TEST_CASE("real_run_copy copies exactly its run", "[nd][swizzle]", float, double) {
    using T = TestType;
    constexpr std::size_t W = xsimd::batch<T>::size;

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
        for (std::size_t i = n; i < dst.size(); ++i) REQUIRE(dst[i] == T(-1));
    }
    REQUIRE(saw_full);
    REQUIRE(saw_masked);
}

TEMPLATE_TEST_CASE("N-D plans honour the measuring efforts", "[nd][effort]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{40, 24};
    const std::size_t Ntot = shape[0] * shape[1];
    const auto in = make_input<T>(Ntot, 0xE770u);

    for (const admiral::effort eff : {admiral::effort::automatic, admiral::effort::measure}) {
        CAPTURE(int(eff));
        admiral::plan<T> p(shape, {0, eff});
        std::vector<std::complex<T>> v = in;
        p.forward(v.data(), v.data());
        p.inverse(v.data(), v.data());
        require_close(v, in, fft_tol<T>());
    }
}

namespace {

template<typename T>
std::vector<std::complex<T>> reference_nd(const std::vector<std::complex<T>>& x,
                                          const std::vector<std::size_t>& shape,
                                          bool forward) {
    std::vector<std::complex<T>> cur = x, next(x.size());
    std::size_t inner = 1;
    for (std::size_t d = shape.size(); d-- > 0;) {
        const std::size_t len = shape[d], outer = x.size() / (len * inner);
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t g = 0; g < inner; ++g) {
                std::vector<std::complex<T>> line(len);
                for (std::size_t p = 0; p < len; ++p) line[p] = cur[o * len * inner + p * inner + g];
                const auto out = reference_dft<T>(line, forward);
                for (std::size_t p = 0; p < len; ++p) next[o * len * inner + p * inner + g] = out[p];
            }
        cur = next;
        inner *= len;
    }
    return cur;
}

}

TEMPLATE_TEST_CASE("N-D out-of-place catalog rows match the reference DFT", "[nd][oop]",
                   float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes{
        {3, 2}, {5, 4}, {3, 8}, {5, 16}, {9, 32}, {2, 64}, {64, 64}, {3, 120},
        {2, 360}, {4, 66}, {5, 96}, {4, 8, 16}, {3, 4, 4}};
    for (const std::vector<std::size_t>& shape : shapes) {
        const std::size_t n = shape_product(shape);
        const auto in = make_input<T>(n, 4000u + unsigned(n));
        admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()));

        INFO("shape rank=" << shape.size() << " n=" << n);
        std::vector<std::complex<T>> out(n);
        p.forward(in.data(), out.data());
        require_close(out, reference_nd(in, shape, true), fft_tol<T>());

        auto ref = reference_nd(out, shape, false);
        for (auto& v : ref) v /= static_cast<T>(n);
        std::vector<std::complex<T>> back(n);
        p.inverse(out.data(), back.data());
        require_close(back, ref, fft_tol<T>());

        auto fwd2 = reference_nd(in, shape, true);
        for (auto& v : fwd2) v *= T(2);
        std::vector<std::complex<T>> out2(n);
        p.forward(in.data(), out2.data(), T(2));
        require_close(out2, fwd2, fft_tol<T>());
    }
}
