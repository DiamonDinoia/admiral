// Bit-exact correctness tests for the iterative DIF (Gentleman-Sande) pass-chain
// driver against the compile-time recursive kernel<N> oracle.
//
// For every codelet-supported size (7-smooth composites and pow2 up to 1024),
// both forward and inverse transforms are checked for float and double.
// Tolerance follows the existing pattern: eps * sqrt(N) * (log2(N) + 1) * 64.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/detail/codelet.hpp>
#include <admiral/detail/kernels.hpp>
#include <admiral/detail/twiddles.hpp>

#include <cmath>
#include <complex>
#include <vector>

using namespace admiral::detail;

namespace {

// Deterministic but non-trivial input (same generator as test_fft_codelet.cpp).
template<typename T>
std::vector<std::complex<T>> make_input(std::size_t N) {
    std::vector<std::complex<T>> x(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                               std::cos(T(1.1) * T(n) - T(0.2)));
    }
    return x;
}

// Run the kernel<N> oracle and the iterative DIF driver on the same input,
// compare element-wise within tolerance.
template<unsigned N, bool Forward, typename T>
void check_iterative_vs_oracle() {
    const auto x = make_input<T>(N);

    // --- Oracle: compile-time recursive kernel<N> ---
    std::vector<T> xre_o(N), xim_o(N), yre_o(N), yim_o(N);
    for (std::size_t i = 0; i < N; ++i) {
        xre_o[i] = x[i].real();
        xim_o[i] = x[i].imag();
    }
    kernel<N, T, Forward>::apply(xre_o.data(), xim_o.data(), 1,
                                  yre_o.data(), yim_o.data());

    // --- Iterative DIF driver ---
    std::vector<std::complex<T>> data_it(x.begin(), x.end());
    {
        const auto dtw = Forward ? build_dif_twiddle_set<T, true>(N)
                                 : build_dif_twiddle_set<T, false>(N);
        std::vector<T> cc0re(N), cc0im(N), cc1re(N), cc1im(N);
        if (Forward) {
            iterative_dif_execute_ws<T, true>(data_it.data(), data_it.data(), N,
                cc0re.data(), cc0im.data(), cc1re.data(), cc1im.data(), dtw);
        } else {
            iterative_dif_execute_ws<T, false>(data_it.data(), data_it.data(), N,
                cc0re.data(), cc0im.data(), cc1re.data(), cc1im.data(), dtw);
        }
    }

    // Tolerance: eps * sqrt(N) * (log2(N) + 1) * 64
    const T eps = std::numeric_limits<T>::epsilon();
    const T tol = eps * std::sqrt(T(N)) * (std::log2(T(N)) + T(1)) * T(64);

    for (std::size_t k = 0; k < N; ++k) {
        const T ref_r = yre_o[k];
        const T ref_i = yim_o[k];
        REQUIRE_THAT(data_it[k].real(),
                     Catch::Matchers::WithinAbs(ref_r, tol * (T(1) + std::abs(ref_r))));
        REQUIRE_THAT(data_it[k].imag(),
                     Catch::Matchers::WithinAbs(ref_i, tol * (T(1) + std::abs(ref_i))));
    }
}

template<unsigned N>
void check_size() {
    check_iterative_vs_oracle<N, true,  double>();
    check_iterative_vs_oracle<N, false, double>();
    check_iterative_vs_oracle<N, true,  float>();
    check_iterative_vs_oracle<N, false, float>();
}

} // namespace

// 7-smooth composite sizes (same set as test_fft_codelet.cpp)
TEST_CASE("iterative DIF matches kernel<N> oracle — 7-smooth composites", "[iterative_dif]") {
    check_size<1>();
    check_size<2>();
    check_size<3>();
    check_size<4>();
    check_size<5>();
    check_size<6>();
    check_size<7>();
    check_size<8>();
    check_size<9>();
    check_size<10>();
    check_size<12>();
    check_size<15>();
    check_size<16>();
    check_size<24>();
    check_size<25>();
    check_size<27>();
    check_size<28>();
    check_size<30>();
    check_size<32>();
    check_size<35>();
    check_size<36>();
    check_size<48>();
    check_size<49>();
    check_size<56>();
    check_size<60>();
    check_size<63>();
    check_size<64>();
    check_size<105>();
    check_size<120>();
    check_size<125>();
    check_size<128>();
    check_size<175>();
    check_size<189>();
    check_size<210>();
    check_size<240>();
    check_size<256>();
    check_size<315>();
    check_size<360>();
    check_size<420>();
    check_size<512>();
}

// 11-smooth sizes: radix-11 routing (previously forced to Bluestein). Covers
// radix-11 as a single fused pass (11), and in first / intermediate / last pass
// positions combined with radices 2/3/4/5/7. 121 = 11^2 is the prior worst loser.
TEST_CASE("iterative DIF matches kernel<N> oracle — 11-smooth sizes", "[iterative_dif]") {
    check_size<11>();
    check_size<22>();   // [2, 11]
    check_size<33>();   // [3, 11]
    check_size<55>();   // [5, 11]
    check_size<77>();   // [7, 11]
    check_size<99>();   // [3, 3, 11]
    check_size<110>();  // [2, 5, 11]
    check_size<121>();  // [11, 11]
    check_size<154>();  // [2, 7, 11]
    check_size<242>();  // [2, 11, 11]
}

// Power-of-2 sizes up to 1024
TEST_CASE("iterative DIF matches kernel<N> oracle — pow2 sizes", "[iterative_dif]") {
    check_size<2>();
    check_size<4>();
    check_size<8>();
    check_size<16>();
    check_size<32>();
    check_size<64>();
    check_size<128>();
    check_size<256>();
    check_size<512>();
    check_size<1024>();
}
