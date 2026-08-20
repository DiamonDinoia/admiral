#pragma once

// What a test compares against, and how close is close enough. One definition
// for the whole suite; per-file copies drift.

// A -ffast-math oracle is no more exact than what it judges, so test TUs must
// not inherit it (admiral_fast_math_flags is PRIVATE). The compile-time check
// is the only place a linkage regression is visible. __FINITE_MATH_ONLY__ is
// separate: it does not define __FAST_MATH__, yet alone lets the compiler
// assume no NaN/Inf in the reference accumulation.
#ifdef __FAST_MATH__
#error "test TUs must not be compiled with -ffast-math (see admiral_fast_math_flags)"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "test TUs must not be compiled with -ffinite-math-only (see admiral_fast_math_flags)"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <functional>
#include <iterator>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <span>
#include <type_traits>
#include <vector>

// Element count of a shape, over any range of extents: what the shared error budget
// takes as Ntot for a separable N-D transform.
std::size_t shape_product(const auto& shape) {
    return std::reduce(std::begin(shape), std::end(shape), std::size_t{1}, std::multiplies<>{});
}

// Normwise relative L2 distance ||a-b||_2 / ||a||_2, with `a` the reference. L2
// rather than per-element so a near-zero spectral bin cannot inflate the ratio.
// Elements may be real or complex; the accumulators are double so a 2^20-point
// float vector does not lose the error being measured.
double relerrtwonorm(const auto& a, const auto& b) {
    static_assert(requires { { std::norm(a[0] - b[0]) } -> std::convertible_to<double>; },
                  "relerrtwonorm: element types must subtract to something normable");
    // A reference may carry a WIDER element type than the result it judges, and the
    // subtraction then promotes `b[m]`. Widen it explicitly: clang reports the implicit
    // promotion under -Wdouble-promotion, which the suite builds with -Werror.
    using Ref = std::remove_cvref_t<decltype(a[0])>;
    double err = 0.0, nrm = 0.0;
    for (std::size_t m = 0; m < std::size(a); ++m) {
        nrm += static_cast<double>(std::norm(a[m]));
        err += static_cast<double>(std::norm(a[m] - static_cast<Ref>(b[m])));
    }
    // An all-zero reference has no scale to be relative to; err/nrm would be NaN, and
    // `NaN <= tol` fails with a message that says nothing. Fall back to the absolute norm.
    return nrm > 0.0 ? std::sqrt(err / nrm) : std::sqrt(err);
}

// Relative-L2 budget: a flat multiple of eps, no N term. The scaled DFT is
// unitary (kappa_2 == 1), so a small multiple of eps suffices; an N-dependent
// bound like sqrt(N)*log2(N) cannot fail at large N and is the trap to avoid.
// `scale` compensates tests that compound several transforms.
template<typename T>
constexpr double fft_tol(double scale = 1.0) {
    return 32.0 * scale * static_cast<double>(std::numeric_limits<T>::epsilon());
}

// Catch2's WithinAbs takes double. Handing it float arguments promotes them
// implicitly, which -Wdouble-promotion flags at every call site; convert once here.
auto WithinAbsT(const auto target, const auto margin) {
    return Catch::Matchers::WithinAbs(static_cast<double>(target), static_cast<double>(margin));
}

// The suite's one array comparison. Reports the measured error, not just "false":
// a bare predicate inside a REQUIRE says nothing about how far off the result was.
void require_close(const auto& got, const auto& ref, double tol) {
    REQUIRE(std::size(got) == std::size(ref));
    const double err = relerrtwonorm(ref, got);
    INFO("relative L2 error " << err << " (tol " << tol << ")");
    REQUIRE(err <= tol);
}

// The same check over the C API's structs, which have no operator-. Casting rather than
// re-deriving the comparison: adm_complex is layout-compatible with std::complex by
// documented contract (admiral.h), so this is also the conversion a C caller performs.
template <typename C>
void require_close_c(const C* got, const C* ref, std::size_t size, double tol) {
    using T = decltype(C::real);
    require_close(std::span(reinterpret_cast<const std::complex<T>*>(got), size),
                  std::span(reinterpret_cast<const std::complex<T>*>(ref), size), tol);
}

// Neumaier-compensated energy sum: `long double` is `double` on Apple Silicon, so a
// wider accumulator alone does not clear the transform's own error. Terms are
// std::norm, hence non-negative; Neumaier's magnitude test reduces to `s >= t`.
long double energy(const auto& v) {
    long double s = 0, c = 0;
    for (const auto& e : v) {
        const long double t = static_cast<long double>(std::norm(e));
        const long double n = s + t;
        c += s >= t ? (s - n) + t : (t - n) + s;
        s = n;
    }
    return s + c;
}

// Parseval for the unnormalized forward: ||X||^2 == Ntot * ||x||^2. Energies are
// squares, so an amplitude relative error e shows up here as ~2e, hence the scale.
template<typename T>
void require_parseval(const auto& x, const auto& X, std::size_t ntot, double scale = 4.0) {
    const double want = static_cast<double>(energy(x) * static_cast<long double>(ntot));
    const double got = static_cast<double>(energy(X));
    INFO("Parseval: ntot=" << ntot << " got " << got << " want " << want);
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(want, fft_tol<T>(scale)));
}

// Fraction of a turn of a DFT tone, K*n/N reduced exactly by integer mod first.
// Forming K*n/N in T instead loses the low bits of a large angle: at N=2^20 that
// is a 4e-10 relative error in the *generated input*, which then reads as a
// library failure. K*n <= N^2, so size_t covers every N a test builds.
constexpr long double turn_fraction(std::size_t K, std::size_t n, std::size_t N) {
    return static_cast<long double>((K * n) % N) / static_cast<long double>(N);
}

// exp(+2*pi*i*turns), rounded to T once at the end. `turns` may exceed 1.
template<typename T>
std::complex<T> unit_phasor(long double turns) {
    const long double a = 2.0L * std::numbers::pi_v<long double> * std::fmod(turns, 1.0L);
    return {static_cast<T>(std::cos(a)), static_cast<T>(std::sin(a))};
}

// Direct O(N^2) DFT reference with long double accumulation. Forward uses
// exp(-2*pi*i*k*n/N) (this library's sign convention), unnormalized. The N-entry
// root table keeps the inner loop table-lookup rather than N^2 libm calls;
// `Out` widens where the caller judges something narrowed to its own type.
template<typename T, typename Out = T>
std::vector<std::complex<Out>> reference_dft(const std::vector<std::complex<T>>& x, bool forward) {
    const std::size_t N = x.size();
    const long double sign = forward ? -1.0L : 1.0L;
    std::vector<std::complex<long double>> w(N);
    for (std::size_t j = 0; j < N; ++j) {
        const long double ang = sign * 2.0L * std::numbers::pi_v<long double>
                                * static_cast<long double>(j) / static_cast<long double>(N);
        w[j] = {std::cos(ang), std::sin(ang)};
    }
    std::vector<std::complex<Out>> out(N);
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<long double> acc{0.0L, 0.0L};
        for (std::size_t n = 0; n < N; ++n)
            acc += std::complex<long double>(x[n]) * w[(k * n) % N];
        out[k] = {static_cast<Out>(acc.real()), static_cast<Out>(acc.imag())};
    }
    return out;
}

template<typename T>
std::vector<std::complex<T>> make_signal(std::size_t N) {
    std::vector<std::complex<T>> v(N);
    for (std::size_t i = 0; i < N; ++i)
        v[i] = std::complex<T>(std::cos(T(0.3) * T(i) + T(1)), std::sin(T(0.17) * T(i)));
    return v;
}

// Deterministic but non-trivial input.
template<typename T>
std::vector<std::complex<T>> make_input(std::size_t N) {
    std::vector<std::complex<T>> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                               std::cos(T(1.1) * T(n) - T(0.2)));
    return x;
}

// Deterministic pseudo-random input; seed keeps a failure reproducible.
template<typename T>
std::vector<std::complex<T>> make_input(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<T>> v(n);
    for (auto& z : v) z = std::complex<T>(static_cast<T>(dist(gen)), static_cast<T>(dist(gen)));
    return v;
}

template<typename T>
std::vector<T> make_real_input(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T> v(n);
    for (auto& x : v) x = static_cast<T>(dist(gen));
    return v;
}
