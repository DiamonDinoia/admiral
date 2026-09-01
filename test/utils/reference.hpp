#pragma once

#ifdef __FAST_MATH__
#error "test TUs must not be compiled with -ffast-math (see admiral_fast_math_flags)"
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "test TUs must not be compiled with -ffinite-math-only (see admiral_fast_math_flags)"
#endif

#include <catch2/catch_test_macros.hpp>

#include "admiral/detail/cxx_compat.hpp"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <complex>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

template <typename Shape>
std::size_t shape_product(const Shape& shape) {
    return std::reduce(std::begin(shape), std::end(shape), std::size_t{1}, std::multiplies<>{});
}

template <typename A, typename B>
double relerrtwonorm(const A& a, const B& b) {
    static_assert(std::is_convertible_v<decltype(std::norm(a[0] - b[0])), double>,
                  "relerrtwonorm: element types must subtract to something normable");
    using Ref = admiral::detail::remove_cvref_t<decltype(a[0])>;
    double err = 0.0, nrm = 0.0;
    for (std::size_t m = 0; m < std::size(a); ++m) {
        nrm += static_cast<double>(std::norm(a[m]));
        err += static_cast<double>(std::norm(a[m] - static_cast<Ref>(b[m])));
    }
    return nrm > 0.0 ? std::sqrt(err / nrm) : std::sqrt(err);
}

template<typename T>
constexpr double fft_tol(double scale = 1.0) {
    return 32.0 * scale * static_cast<double>(std::numeric_limits<T>::epsilon());
}

template <typename A, typename B>
auto WithinAbsT(const A target, const B margin) {
    return Catch::Matchers::WithinAbs(static_cast<double>(target), static_cast<double>(margin));
}

template <typename A, typename B>
void require_close(const A& got, const B& ref, double tol) {
    REQUIRE(std::size(got) == std::size(ref));
    const double err = relerrtwonorm(ref, got);
    INFO("relative L2 error " << err << " (tol " << tol << ")");
    REQUIRE(err <= tol);
}

template <typename C>
void require_close_c(const C* got, const C* ref, std::size_t size, double tol) {
    using T = decltype(C::real);
    require_close(admiral::span(reinterpret_cast<const std::complex<T>*>(got), size),
                  admiral::span(reinterpret_cast<const std::complex<T>*>(ref), size), tol);
}

template <typename V>
long double energy(const V& v) {
    long double s = 0, c = 0;
    for (const auto& e : v) {
        const long double t = static_cast<long double>(std::norm(e));
        const long double n = s + t;
        c += s >= t ? (s - n) + t : (t - n) + s;
        s = n;
    }
    return s + c;
}

template<typename T, typename Xv, typename Xo>
void require_parseval(const Xv& x, const Xo& X, std::size_t ntot, double scale = 4.0) {
    const double want = static_cast<double>(energy(x) * static_cast<long double>(ntot));
    const double got = static_cast<double>(energy(X));
    INFO("Parseval: ntot=" << ntot << " got " << got << " want " << want);
    REQUIRE_THAT(got, Catch::Matchers::WithinRel(want, fft_tol<T>(scale)));
}

constexpr long double turn_fraction(std::size_t K, std::size_t n, std::size_t N) {
    return static_cast<long double>((K * n) % N) / static_cast<long double>(N);
}

template<typename T>
std::complex<T> unit_phasor(long double turns) {
    const long double a =
        2.0L * admiral::detail::numbers::pi_v<long double> * std::fmod(turns, 1.0L);
    return {static_cast<T>(std::cos(a)), static_cast<T>(std::sin(a))};
}

template<typename T, typename Out = T>
std::vector<std::complex<Out>> reference_dft(const std::vector<std::complex<T>>& x, bool forward) {
    const std::size_t N = x.size();
    const long double sign = forward ? -1.0L : 1.0L;
    std::vector<std::complex<long double>> w(N);
    for (std::size_t j = 0; j < N; ++j) {
        const long double ang = sign * 2.0L * admiral::detail::numbers::pi_v<long double>
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

template<typename T>
std::vector<std::complex<T>> make_input(std::size_t N) {
    std::vector<std::complex<T>> x(N);
    for (std::size_t n = 0; n < N; ++n)
        x[n] = std::complex<T>(std::sin(T(0.7) * T(n) + T(0.3)),
                               std::cos(T(1.1) * T(n) - T(0.2)));
    return x;
}

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
