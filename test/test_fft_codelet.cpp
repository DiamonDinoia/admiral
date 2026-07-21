// Correctness tests for the generic metaprogrammed codelet kernel<N, T, Forward>.
// Each instantiation is checked against a direct O(N^2) reference DFT, which is
// the ground truth the generic Cooley-Tukey recursion must reproduce.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/detail/codelet.hpp>
#include <admiral/detail/four_step.hpp>

#include <cmath>
#include <complex>
#include <vector>

using namespace admiral::detail;

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;

// Direct DFT reference (unnormalized). Forward uses exp(-2*pi*i*kn/N).
template<typename T>
std::vector<std::complex<T>> reference_dft(const std::vector<std::complex<T>>& x,
                                           bool forward) {
    const std::size_t N = x.size();
    const T sign = forward ? T(-1) : T(1);
    std::vector<std::complex<T>> out(N);
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<T> acc(0, 0);
        for (std::size_t n = 0; n < N; ++n) {
            const T ang = sign * T(2) * T(pi) * T(k) * T(n) / T(N);
            acc += x[n] * std::complex<T>(std::cos(ang), std::sin(ang));
        }
        out[k] = acc;
    }
    return out;
}

// Deterministic but non-trivial input.
template<typename T>
std::vector<std::complex<T>> make_input(std::size_t N) {
    std::vector<std::complex<T>> x(N);
    for (std::size_t n = 0; n < N; ++n) {
        x[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                               std::cos(T(1.1) * T(n) - T(0.2)));
    }
    return x;
}

template<unsigned N, bool Forward, typename T>
void check_one() {
    const auto x = make_input<T>(N);
    std::vector<T> xre(N), xim(N), yre(N), yim(N);
    for (std::size_t i = 0; i < N; ++i) {
        xre[i] = x[i].real();
        xim[i] = x[i].imag();
    }

    kernel<N, T, Forward>::apply(xre.data(), xim.data(), 1, yre.data(), yim.data());

    const auto ref = reference_dft<T>(x, Forward);

    // Error grows ~ sqrt(N)*log2(N); use a generous machine-epsilon multiple.
    const T eps = std::numeric_limits<T>::epsilon();
    const T tol = eps * std::sqrt(T(N)) * (std::log2(T(N)) + T(1)) * T(64);

    for (std::size_t k = 0; k < N; ++k) {
        REQUIRE_THAT(yre[k], Catch::Matchers::WithinAbs(ref[k].real(), tol * (T(1) + std::abs(ref[k].real()))));
        REQUIRE_THAT(yim[k], Catch::Matchers::WithinAbs(ref[k].imag(), tol * (T(1) + std::abs(ref[k].imag()))));
    }
}

template<unsigned N>
void check_size() {
    check_one<N, true, double>();
    check_one<N, false, double>();
    check_one<N, true, float>();
    check_one<N, false, float>();
}

} // namespace

TEST_CASE("codelet kernel<N> matches reference DFT", "[codelet]") {
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
    check_size<30>();
    check_size<32>();
    check_size<36>();
    check_size<48>();
    check_size<52>();  // cofactor-SIMD batched path (4 x Rader-13) for double
    check_size<60>();
    check_size<64>();
    check_size<120>();
    check_size<128>();
}

TEST_CASE("codelet kernel<N> small primes (direct DFT path)", "[codelet]") {
    check_size<11>();
    check_size<13>();
}

TEST_CASE("codelet kernel<N> prime composites (cofactor-SIMD batched path)", "[codelet]") {
    check_size<26>();  // 2*13
    check_size<34>();  // 2*17
    check_size<38>();  // 2*19
    check_size<39>();  // 3*13
    check_size<46>();  // 2*23
    check_size<51>();  // 3*17
    check_size<57>();  // 3*19
    check_size<58>();  // 2*29
    check_size<62>();  // 2*31
}

// Batched four-step (N = N1*N2, both <= 64 catalog, both multiples of W): the
// composition-of-kernels path for N > 64. Validate the index/twiddle math against
// the O(N^2) reference DFT on planar buffers.
template<unsigned N1, unsigned N2, typename T>
void check_four_step() {
    constexpr unsigned N = N1 * N2;
    const auto x = make_input<T>(N);
    std::vector<T> in_re(N), in_im(N), out_re(N), out_im(N), Gre(N), Gim(N);
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = x[i].real(); in_im[i] = x[i].imag(); }

    auto run = [&](bool fwd) {
        constexpr unsigned W = static_cast<unsigned>(xsimd::batch<T>::size);
        if (fwd) {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, true>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, true>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        } else {
            const auto tw = admiral::detail::build_four_step_twiddles_v<T, false>(N1, N2, W);
            std::vector<T> twre(N), twim(N);
            for (std::size_t i = 0; i < N; ++i) { twre[i] = tw[i].real(); twim[i] = tw[i].imag(); }
            admiral::detail::four_step_batched_ct<N1, N2, T, false>(
                in_re.data(), in_im.data(), out_re.data(), out_im.data(),
                twre.data(), twim.data(), Gre.data(), Gim.data());
        }
    };

    for (bool fwd : {true, false}) {
        run(fwd);
        const auto ref = reference_dft<T>(x, fwd);
        const T eps = std::numeric_limits<T>::epsilon();
        // The O(N^2) reference_dft computes cos(2*pi*k*n/N) with k*n up to ~N^2:
        // the large-angle argument loses ~(angle*eps) per term, so the REFERENCE
        // accumulates ~sqrt(N)*2*pi*N*eps (~3e-10 at N=4096). Our four-step uses
        // exact integer turn reduction, so the reference is the accuracy floor
        // here — scale the tol linearly in N to track it. The tight, reference-free
        // correctness signal is the roundtrip identity below.
        const T tol = eps * T(N) * (std::log2(T(N)) + T(1)) * T(64);
        for (std::size_t k = 0; k < N; ++k) {
            REQUIRE_THAT(out_re[k], Catch::Matchers::WithinAbs(ref[k].real(), tol * (T(1) + std::abs(ref[k].real()))));
            REQUIRE_THAT(out_im[k], Catch::Matchers::WithinAbs(ref[k].imag(), tol * (T(1) + std::abs(ref[k].imag()))));
        }
    }

    // Reference-free roundtrip: forward then inverse must recover x (up to 1/N).
    // No large-angle reference involved, so this is a tight check on the math.
    run(true);
    std::vector<T> f_re = out_re, f_im = out_im;
    for (std::size_t i = 0; i < N; ++i) { in_re[i] = f_re[i]; in_im[i] = f_im[i]; }
    run(false);
    const T eps = std::numeric_limits<T>::epsilon();
    const T rtol = eps * std::sqrt(T(N)) * (std::log2(T(N)) + T(1)) * T(64);
    for (std::size_t i = 0; i < N; ++i) {
        REQUIRE_THAT(out_re[i] / T(N), Catch::Matchers::WithinAbs(x[i].real(), rtol * (T(1) + std::abs(x[i].real()))));
        REQUIRE_THAT(out_im[i] / T(N), Catch::Matchers::WithinAbs(x[i].imag(), rtol * (T(1) + std::abs(x[i].imag()))));
    }
}

TEST_CASE("batched four-step matches reference DFT (composition of kernels, N>64)", "[codelet][fourstep]") {
    check_four_step<16, 16, double>();   // 256
    check_four_step<32, 32, double>();   // 1024
    check_four_step<64, 64, double>();   // 4096
    check_four_step<32, 64, double>();   // 2048 (unequal factors)
    check_four_step<16, 16, float>();    // 256
    check_four_step<32, 32, float>();    // 1024
    check_four_step<32, 64, float>();    // 2048
}

TEST_CASE("codelet forward then inverse is identity (up to 1/N)", "[codelet]") {
    auto roundtrip = [](auto Nc) {
        constexpr unsigned N = decltype(Nc)::value;
        using T = double;
        const auto x = make_input<T>(N);
        std::vector<T> xre(N), xim(N), fre(N), fim(N), gre(N), gim(N);
        for (std::size_t i = 0; i < N; ++i) { xre[i] = x[i].real(); xim[i] = x[i].imag(); }

        kernel<N, T, true>::apply(xre.data(), xim.data(), 1, fre.data(), fim.data());
        kernel<N, T, false>::apply(fre.data(), fim.data(), 1, gre.data(), gim.data());

        const T tol = std::numeric_limits<T>::epsilon() * std::sqrt(T(N)) * (std::log2(T(N)) + T(1)) * T(64);
        for (std::size_t i = 0; i < N; ++i) {
            REQUIRE_THAT(gre[i] / T(N), Catch::Matchers::WithinAbs(x[i].real(), tol * (T(1) + std::abs(x[i].real()))));
            REQUIRE_THAT(gim[i] / T(N), Catch::Matchers::WithinAbs(x[i].imag(), tol * (T(1) + std::abs(x[i].imag()))));
        }
    };
    roundtrip(std::integral_constant<unsigned, 12>{});
    roundtrip(std::integral_constant<unsigned, 60>{});
    roundtrip(std::integral_constant<unsigned, 64>{});
}
