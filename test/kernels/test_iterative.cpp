
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/detail/dif_col_driver.hpp>
#include <admiral/detail/dif_driver.hpp>
#include <admiral/detail/dif_passes.hpp>
#include <admiral/detail/twiddles.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace admiral::detail;

namespace {

template<typename T, bool Forward>
void check_iterative_vs_reference(std::size_t N, const dif_factor_plan* forced = nullptr) {
    const auto x = make_input<T>(N);
    const auto ref = reference_dft<T, double>(x, Forward);

    std::vector<std::complex<T>> got(x.begin(), x.end());
    const auto dtw = build_dif_twiddle_set<T>(N, forced);
    std::vector<T> cc0re(N), cc0im(N), cc1re(N), cc1im(N);
    dif_dispatch<T>(Forward, got.data(), got.data(), N, cc0re.data(), cc0im.data(),
                    cc1re.data(), cc1im.data(), dtw);

    require_close(std::vector<std::complex<double>>(got.begin(), got.end()), ref, fft_tol<T>());
}

void check_sizes(std::initializer_list<std::size_t> sizes) {
    for (const std::size_t N : sizes) {
        check_iterative_vs_reference<double, true>(N);
        check_iterative_vs_reference<double, false>(N);
        check_iterative_vs_reference<float, true>(N);
        check_iterative_vs_reference<float, false>(N);
    }
}

}

TEST_CASE("iterative DIF matches the reference DFT: 7-smooth composites", "[iterative_dif]") {
    check_sizes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 24, 25, 27, 28, 30, 32, 35,
                 36, 48, 49, 56, 60, 63, 64, 105, 120, 125, 128, 175, 189, 210, 240, 256,
                 315, 360, 420, 512});
}

TEST_CASE("iterative DIF matches the reference DFT: 11-smooth sizes", "[iterative_dif]") {
    check_sizes({11,
                 22,
                 33,
                 55,
                 77,
                 99,
                 110,
                 121,
                 154,
                 242});
}

TEST_CASE("iterative DIF matches the reference DFT: pow2 sizes", "[iterative_dif]") {
    check_sizes({2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
}

TEST_CASE("a forced radix outside dif_radix_set is rejected", "[iterative_dif]") {
    dif_factor_plan plan;
    plan.push(6);
    plan.push(4);
    REQUIRE_THROWS_AS(build_dif_twiddle_set<double>(24, &plan), admiral::unsupported_error);
}

namespace {

template<typename T, std::size_t IP>
double first_pass_factored_dev(std::size_t ido) {
    constexpr std::size_t W = xsimd::batch<T>::size;
    const std::size_t N = IP * ido;
    const auto in = make_input<T>(N);

    std::vector<T> fre((IP - 1) * ido), fim(fre.size());
    for (std::size_t k = 1; k < IP; ++k)
        geom_twiddle_row<T, true>(k, N, ido, fre.data() + (k - 1) * ido,
                                  fim.data() + (k - 1) * ido);

    std::size_t blk = W;
    while (blk * blk < ido) blk *= 2;
    const std::size_t nb = (ido + blk - 1) / blk;
    std::vector<T> sre((IP - 1) * (blk + nb)), sim(sre.size());
    for (std::size_t k = 1; k < IP; ++k) {
        geom_twiddle_row<T, true>(k, N, blk, sre.data() + (k - 1) * blk,
                                  sim.data() + (k - 1) * blk);
        geom_twiddle_row<T, true>(k * blk, N, nb, sre.data() + (IP - 1) * blk + (k - 1) * nb,
                                  sim.data() + (IP - 1) * blk + (k - 1) * nb);
    }

    std::vector<T> ar(N), ai(N), br(N), bi(N);
    dif_pass_first_impl<T, true, IP, false>(in.data(), ar.data(), ai.data(), 1, ido, fre.data(),
                                            fim.data(), 1, 0);
    dif_pass_first_impl<T, true, IP, true>(in.data(), br.data(), bi.data(), 1, ido, sre.data(),
                                           sim.data(), 1, blk);

    double worst = 0, scale = 0;
    for (std::size_t i = 0; i < N; ++i) {
        worst = std::max(worst, std::hypot(double(ar[i]) - double(br[i]),
                                           double(ai[i]) - double(bi[i])));
        scale = std::max(scale, std::hypot(double(ar[i]), double(ai[i])));
    }
    return worst / scale;
}

template<typename T>
constexpr double factored_tol = 1e3 * double(std::numeric_limits<T>::epsilon());

}

TEST_CASE("the factored pass-0 row matches the flat one at every column",
          "[iterative_dif][twiddles]") {
    CHECK(first_pass_factored_dev<double, 16>(4096) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 5>(20000) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 3>(10925) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 2>(16397) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 4>(8196) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 2>(16389) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 3>(11013) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 7>(4741) < factored_tol<double>);
    CHECK(first_pass_factored_dev<float, 16>(4096) < factored_tol<float>);
    CHECK(first_pass_factored_dev<float, 7>(7817) < factored_tol<float>);
    CHECK(first_pass_factored_dev<float, 5>(11017) < factored_tol<float>);
}

TEST_CASE("iterative DIF matches the reference DFT: forced staged/first/last radix-32 chains",
          "[iterative_dif]") {
    // Radix 16 and 32 only join dif_radix_set under dif_wide_radices (>= 32 vector
    // registers), so every chain below is rejected by build_dif_twiddle_set on a
    // narrower build; skip rather than force an ISA this test cannot run under.
    if (poet::vector_register_count() < 32) {
        SKIP("radix 16/32 need dif_wide_radices (>= 32 vector registers)");
    }
    // Each chain forces a specific pass factoring so the staged first-pass split
    // (dif_staged_radix), the L1==1 first-pass fast path and the last pass's
    // lane-stage/row-split arm all get walked by a real forward+inverse-matching check.
    const std::pair<std::size_t, dif_factor_plan> chains[] = {
        {256, {8, 32}},   {256, {16, 16}},
        {1024, {8, 8, 16}}, {1024, {16, 8, 8}}, {1024, {32, 32}},
        {4096, {8, 8, 8, 8}}, {4096, {32, 16, 8}},
    };
    for (const auto& [N, plan] : chains) {
        CAPTURE(N);
        check_iterative_vs_reference<double, true>(N, &plan);
        check_iterative_vs_reference<double, false>(N, &plan);
        check_iterative_vs_reference<float, true>(N, &plan);
        check_iterative_vs_reference<float, false>(N, &plan);
    }
    // The staged first-pass footprint gate (T2) picks the split table only once the plain
    // table would blow the L1D admission, so it must fire at N=1024 (ido0=128) and stay off
    // at N=256 (ido0=32) for the same ip0=8 first radix.
    const dif_factor_plan plan_1024{8, 8, 16};
    const dif_factor_plan plan_256{8, 32};
    CHECK(build_dif_twiddle_set<double>(1024, &plan_1024).p0_block != 0);
    CHECK(build_dif_twiddle_set<double>(256, &plan_256).p0_block == 0);
}

TEST_CASE("forced radix-32 chain check fires on a perturbed result (positive control)",
          "[iterative_dif]") {
    if (poet::vector_register_count() < 32) {
        SKIP("radix 16/32 need dif_wide_radices (>= 32 vector registers)");
    }
    constexpr std::size_t N = 1024;
    const dif_factor_plan plan{16, 8, 8};
    const auto x = make_input<double>(N);
    const auto ref = reference_dft<double, double>(x, true);

    std::vector<std::complex<double>> got(x.begin(), x.end());
    const auto dtw = build_dif_twiddle_set<double>(N, &plan);
    std::vector<double> cc0re(N), cc0im(N), cc1re(N), cc1im(N);
    dif_dispatch<double>(true, got.data(), got.data(), N, cc0re.data(), cc0im.data(),
                        cc1re.data(), cc1im.data(), dtw);

    const double tol = fft_tol<double>();
    const double err_ok = relerrtwonorm(ref, got);
    INFO("unperturbed relative L2 error " << err_ok << " (tol " << tol << ")");
    CHECK(err_ok <= tol);

    got[0] += std::complex<double>(1.0, 0.0);
    const double err_bad = relerrtwonorm(ref, got);
    INFO("perturbed relative L2 error " << err_bad << " (tol " << tol << ")");
    CHECK(err_bad > tol);
}

TEST_CASE("col dif: first_src copy-in requires its own stride", "[iterative][col]") {
    const std::size_t N = 16;
    const auto dtw = build_dif_twiddle_set<double>(N, nullptr, false);
    std::vector<std::complex<double>> data(N * 4), src(N * 4);
    std::vector<double> cc0(N * 4), cc1(N * 4), cc2(N * 4), cc3(N * 4);
    REQUIRE_THROWS_AS(
        (col_dif_execute_ws<double, true>(data.data(), N, 4, 4, cc0.data(), cc1.data(),
                                          cc2.data(), cc3.data(), dtw, 1.0,
                                          src.data(), 0)),
        admiral::internal_error);
    REQUIRE_NOTHROW(
        (col_dif_execute_ws<double, true>(data.data(), N, 4, 4, cc0.data(), cc1.data(),
                                          cc2.data(), cc3.data(), dtw, 1.0, src.data(),
                                          4)));
}
