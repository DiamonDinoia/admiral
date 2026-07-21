// Explicit output-scale (`fct`) parameter, ducc0-style. Every transform takes an
// optional fct: nullopt = the convention default (forward unscaled, inverse 1/N),
// any value = that exact output scale. These tests pin three properties:
//   1. fct == convention-default is byte-for-byte the nullopt path (same result).
//   2. fct linearly scales the output (result(fct) == fct * result(1)).
//   3. unnormalized round-trips compose: fwd(fct=1) then inv(fct=1/N) == identity,
//      and fwd(fct=1) then inv(fct=1) == N * input.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <cstddef>
#include <admiral/admiral.hpp>
#include <numbers>
#include <span>
#include <vector>

namespace {

template<typename T>
std::vector<std::complex<T>> make_signal(std::size_t N) {
    std::vector<std::complex<T>> v(N);
    for (std::size_t i = 0; i < N; ++i)
        v[i] = std::complex<T>(std::sin(T(2) * std::numbers::pi_v<T> * T(i) / T(N)),
                               T(0.5) * std::cos(T(3) * std::numbers::pi_v<T> * T(i) / T(N)));
    return v;
}

template<typename T>
T tol(std::size_t N, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    return eps * std::sqrt(T(N)) * (std::log2(T(N)) + T(1)) * T(64) * scale;
}

template<typename T>
void require_close(const std::vector<std::complex<T>>& a, const std::vector<std::complex<T>>& b, T eps) {
    REQUIRE(a.size() == b.size());
    for (std::size_t i = 0; i < a.size(); ++i) REQUIRE(std::abs(a[i] - b[i]) <= eps);
}

}  // namespace

TEMPLATE_TEST_CASE("fct: default value equals nullopt path", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {8u, 15u, 32u, 63u, 120u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);

        std::vector<std::complex<T>> a = in, b = in;
        p.forward(std::span(a));                       // nullopt -> unscaled
        p.forward(std::span(b), T(1));                 // explicit default
        require_close(a, b, tol<T>(N));

        a = in; b = in;
        p.inverse(std::span(a));                       // nullopt -> 1/N
        p.inverse(std::span(b), T(1) / T(N));          // explicit default
        require_close(a, b, tol<T>(N));
    }
}

TEMPLATE_TEST_CASE("fct: linear output scaling", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {16u, 24u, 31u, 64u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);
        const T k = T(2.5);

        std::vector<std::complex<T>> base = in, scaled = in;
        p.forward(std::span(base), T(1));
        p.forward(std::span(scaled), k);
        for (auto& x : base) x *= k;
        require_close(base, scaled, tol<T>(N, k));
    }
}

TEMPLATE_TEST_CASE("fct: unnormalized round-trip composes", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {8u, 30u, 64u, 127u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);

        // fwd(1) then inv(1/N) recovers the input.
        std::vector<std::complex<T>> rt = in;
        p.forward(std::span(rt), T(1));
        p.inverse(std::span(rt), T(1) / T(N));
        require_close(rt, in, tol<T>(N, T(2)));

        // fwd(1) then inv(1) yields N * input (both directions unscaled).
        std::vector<std::complex<T>> nrt = in;
        p.forward(std::span(nrt), T(1));
        p.inverse(std::span(nrt), T(1));
        auto scaled_in = in;
        for (auto& x : scaled_in) x *= T(N);
        require_close(nrt, scaled_in, tol<T>(N, T(2) * T(N)));
    }
}

TEMPLATE_TEST_CASE("fct: N-D custom scale round-trip", "[fct][nd]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{6, 10};
    const std::size_t Ntot = shape[0] * shape[1];
    std::vector<std::complex<T>> in(Ntot);
    for (std::size_t i = 0; i < Ntot; ++i) in[i] = std::complex<T>(T(i % 7), T(i % 3));

    admiral::plan<T> p(shape);
    std::vector<std::complex<T>> rt = in;
    p.forward(std::span(rt), T(1));
    p.inverse(std::span(rt), T(1) / T(Ntot));
    require_close(rt, in, tol<T>(Ntot, T(2)));
}

TEMPLATE_TEST_CASE("fct: r2c/c2r custom scale round-trip", "[fct][r2c]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{4, 8};
    admiral::plan_r2c<T> p(shape);
    const std::size_t Nr = p.real_size();

    std::vector<T> in(Nr);
    for (std::size_t i = 0; i < Nr; ++i) in[i] = std::sin(T(i));
    std::vector<std::complex<T>> spec(p.cplx_size());
    std::vector<T> out(Nr);

    // r2c unscaled, c2r with 1/Nr -> identity.
    p.forward(in.data(), spec.data(), T(1));
    p.inverse(spec.data(), out.data(), T(1) / T(Nr));
    for (std::size_t i = 0; i < Nr; ++i) REQUIRE(std::abs(out[i] - in[i]) <= tol<T>(Nr, T(2)));
}
