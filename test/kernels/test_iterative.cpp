// Iterative DIF driver validation against an independent O(N^2) oracle, NOT
// admiral's kernel<N>: the oracle must share no code with the thing under
// test. kernel<N> takes the same oracle at catalog sizes in test_codelet.cpp.

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
    // Wider than T: a f64 oracle narrowed to double carries ~1 eps of summation
    // error against the flat 32-eps budget, which the check tolerates.
    const auto ref = reference_dft<T, double>(x, Forward);

    std::vector<std::complex<T>> got(x.begin(), x.end());
    const auto dtw = build_dif_twiddle_set<T>(N, forced);
    std::vector<T> cc0re(N), cc0im(N), cc1re(N), cc1im(N);
    // Through the trampoline, like every production caller: naming a leaf directly would
    // make this TU instantiate the whole engine tree instead of referencing inst_dif_*.
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

}  // namespace

// 7-smooth composite sizes
TEST_CASE("iterative DIF matches the reference DFT: 7-smooth composites", "[iterative_dif]") {
    check_sizes({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 24, 25, 27, 28, 30, 32, 35,
                 36, 48, 49, 56, 60, 63, 64, 105, 120, 125, 128, 175, 189, 210, 240, 256,
                 315, 360, 420, 512});
}

// 11-smooth sizes: radix-11 routing. Covers radix-11 as a single fused pass (11),
// and in first / intermediate / last pass positions combined with radices 2/3/4/5/7.
// 121 = 11^2 is the worst case, being nothing but radix 11.
TEST_CASE("iterative DIF matches the reference DFT: 11-smooth sizes", "[iterative_dif]") {
    check_sizes({11,    // [11]
                 22,    // [2, 11]
                 33,    // [3, 11]
                 55,    // [5, 11]
                 77,    // [7, 11]
                 99,    // [3, 3, 11]
                 110,   // [2, 5, 11]
                 121,   // [11, 11]
                 154,   // [2, 7, 11]
                 242}); // [2, 11, 11]
}

// Power-of-2 sizes up to 1024
TEST_CASE("iterative DIF matches the reference DFT: pow2 sizes", "[iterative_dif]") {
    check_sizes({2, 4, 8, 16, 32, 64, 128, 256, 512, 1024});
}

// Every pass dispatches over dif_radix_set, so a forced radix outside it skipped the pass.
TEST_CASE("a forced radix outside dif_radix_set is rejected", "[iterative_dif]") {
    dif_factor_plan plan;
    plan.push(6);  // excluded as never emitted, on every ISA
    plan.push(4);
    REQUIRE_THROWS_AS(build_dif_twiddle_set<double>(24, &plan), std::invalid_argument);
}

// The factored pass-0 row (dif_twiddle_set::p0_block) is read W lanes at a time from a
// blk-wide block, so a batch is only valid while a0+W stays inside one block. The overlap
// tail starts at ido-W, not W-aligned when ido%W != 0; the tail then takes the flat row.
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

}  // namespace

TEST_CASE("the factored pass-0 row matches the flat one at every column",
          "[iterative_dif][twiddles]") {
    // ido % W == 0: no overlap tail at all, the shape the goal cells run.
    CHECK(first_pass_factored_dev<double, 16>(4096) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 5>(20000) < factored_tol<double>);
    // Overlap tail fires, batch stays inside its block.
    CHECK(first_pass_factored_dev<double, 3>(10925) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 2>(16397) < factored_tol<double>);
    // Overlap tail fires and would straddle a block: the regression.
    CHECK(first_pass_factored_dev<double, 4>(8196) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 2>(16389) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 3>(11013) < factored_tol<double>);
    CHECK(first_pass_factored_dev<double, 7>(4741) < factored_tol<double>);
    // f32 has a wider W, so it straddles at different ido.
    CHECK(first_pass_factored_dev<float, 16>(4096) < factored_tol<float>);
    CHECK(first_pass_factored_dev<float, 7>(7817) < factored_tol<float>);
    CHECK(first_pass_factored_dev<float, 5>(11017) < factored_tol<float>);
}

TEST_CASE("col dif: first_src copy-in requires its own stride", "[iterative][col]") {
    // The defensive throw at col_dif_execute_ws: no public caller reaches it
    // (four_step_large always passes n1), so it is checked here directly.
    const std::size_t N = 16;
    const auto dtw = build_dif_twiddle_set<double>(N, nullptr, /*fuse_packed=*/false);
    std::vector<std::complex<double>> data(N * 4), src(N * 4);
    std::vector<double> cc0(N * 4), cc1(N * 4), cc2(N * 4), cc3(N * 4);
    REQUIRE_THROWS_AS(
        (col_dif_execute_ws<double, true>(data.data(), N, 4, 4, cc0.data(), cc1.data(),
                                          cc2.data(), cc3.data(), dtw, 1.0,
                                          src.data(), 0)),
        std::invalid_argument);
    REQUIRE_NOTHROW(
        (col_dif_execute_ws<double, true>(data.data(), N, 4, 4, cc0.data(), cc1.data(),
                                          cc2.data(), cc3.data(), dtw, 1.0, src.data(),
                                          4)));
}
