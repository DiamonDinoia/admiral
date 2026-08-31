// Explicit output-scale `fct` parameter, ducc0-style. Every transform takes an
// optional `fct`: `nullopt` is the convention default (forward unscaled, inverse
// 1/N), any value is that exact output scale. These tests pin three properties:
//   1. `fct` == the convention default is byte-for-byte the `nullopt` path.
//   2. `fct` scales the output linearly: result(`fct`) == `fct` * result(1).
//   3. Unnormalized round trips compose: fwd(`fct`=1) then inv(`fct`=1/N) is the
//      identity, and fwd(`fct`=1) then inv(`fct`=1) is N * input.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <vector>

TEMPLATE_TEST_CASE("fct: default value equals nullopt path", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {8u, 15u, 32u, 63u, 120u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);

        std::vector<std::complex<T>> a = in, b = in;
        p.forward(admiral::span(a));                       // `nullopt` -> unscaled
        p.forward(admiral::span(b), T(1));                 // explicit default
        require_close(a, b, fft_tol<T>());

        a = in; b = in;
        p.inverse(admiral::span(a));                       // `nullopt` -> 1/N
        p.inverse(admiral::span(b), T(1) / T(N));          // explicit default
        require_close(a, b, fft_tol<T>());
    }
}

TEMPLATE_TEST_CASE("fct: linear output scaling", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {16u, 24u, 31u, 64u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);
        const T k = T(2.5);

        std::vector<std::complex<T>> base = in, scaled = in;
        p.forward(admiral::span(base), T(1));
        p.forward(admiral::span(scaled), k);
        for (auto& x : base) x *= k;
        require_close(base, scaled, fft_tol<T>());
    }
}

TEMPLATE_TEST_CASE("fct: unnormalized round-trip composes", "[fct]", float, double) {
    using T = TestType;
    for (std::size_t N : {8u, 30u, 64u, 127u}) {
        const auto in = make_signal<T>(N);
        admiral::plan<T> p(N);

        // fwd(1) then inv(1/N) recovers the input.
        std::vector<std::complex<T>> rt = in;
        p.forward(admiral::span(rt), T(1));
        p.inverse(admiral::span(rt), T(1) / T(N));
        require_close(rt, in, fft_tol<T>(2));

        // fwd(1) then inv(1) yields N * input (both directions unscaled).
        std::vector<std::complex<T>> nrt = in;
        p.forward(admiral::span(nrt), T(1));
        p.inverse(admiral::span(nrt), T(1));
        auto scaled_in = in;
        for (auto& x : scaled_in) x *= T(N);
        require_close(nrt, scaled_in, fft_tol<T>(2));
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
    p.forward(admiral::span(rt), T(1));
    p.inverse(admiral::span(rt), T(1) / T(Ntot));
    require_close(rt, in, fft_tol<T>(2));
}

// A degenerate tensor (no extent > 1) has no axis to fold a custom `fct` into.
// The N-D driver applies the scale as a standalone post-pass. The round-trip tests
// above reach the in-place overload's copy of that branch but not the (src, dst)
// overload's copy. In the (src, dst) copy, the scale is the ONLY thing the call
// does, so a dropped scale would still look like a correct identity.
TEMPLATE_TEST_CASE("fct: degenerate tensor scales out-of-place", "[fct][nd]",
                   float, double) {
    using T = TestType;
    const std::complex<T> v{T(3), T(-5)};
    for (std::size_t rank = 1; rank <= 3; ++rank) {
        const std::vector<std::size_t> shape(rank, 1);   // {1}, {1,1}, {1,1,1}
        CAPTURE(rank);
        admiral::plan<T> p(admiral::span<const std::size_t>(shape.data(), shape.size()));
        REQUIRE(p.size() == 1);
        std::complex<T> src = v, dst{};
        p.forward(&src, &dst, T(2));
        REQUIRE(dst == v * T(2));
        // The inverse's 1/N is 1 here, so a custom `fct` is the only visible scale.
        p.inverse(&src, &dst, T(4));
        REQUIRE(dst == v * T(4));
    }
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

    // A "custom" scale has to DIFFER from the default, or neither scaling loop
    // runs. The r2c pass skips the loop when `fct` == 1, and the c2r pass skips the
    // loop when `fct` == 1/`Nr`. Passing either default tests nothing. Both custom
    // values are powers of two, so the scale itself is exact and only the transform
    // carries error.
    std::vector<std::complex<T>> unscaled(p.cplx_size());
    p.forward(in.data(), unscaled.data());              // default r2c scale (1)
    p.forward(in.data(), spec.data(), T(2));
    std::vector<std::complex<T>> twice = unscaled;
    for (auto& z : twice) z *= T(2);
    // The comparison must be able to FAIL: if `fct` were ignored, `spec` would
    // equal `unscaled`, so `unscaled` and `twice` have to be far apart.
    T mag = 0;
    for (const auto& z : unscaled) mag = std::max(mag, std::abs(z));
    REQUIRE(mag > T(1e-3));
    require_close(spec, twice, fft_tol<T>(2));           // forward applied the scale

    // c2r with 1/(2*`Nr`) undoes both the forward 2 and the transform's own `Nr`.
    p.inverse(spec.data(), out.data(), T(1) / (T(2) * T(Nr)));
    require_close(out, in, fft_tol<T>(2));
}

TEMPLATE_TEST_CASE("plan_r2c survives a move", "[r2c]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 2> shape{4, 8};
    admiral::plan_r2c<T> ref(shape);
    const std::size_t Nr = ref.real_size();

    std::vector<T> in(Nr);
    for (std::size_t i = 0; i < Nr; ++i) in[i] = std::sin(T(i));
    std::vector<std::complex<T>> want(ref.cplx_size()), got = want, got2 = want;
    ref.forward(in.data(), want.data(), T(1));

    // A defaulted move on the pimpl is only correct while every member moves along.
    // The moved-to plan must still transform, not only report its sizes.
    admiral::plan_r2c<T> src(shape);
    admiral::plan_r2c<T> moved(std::move(src));
    moved.forward(in.data(), got.data(), T(1));
    REQUIRE(got == want);

    admiral::plan_r2c<T> dst(std::array<std::size_t, 2>{2, 4});
    dst = std::move(moved);
    REQUIRE(dst.real_size() == Nr);
    dst.forward(in.data(), got2.data(), T(1));
    REQUIRE(got2 == want);
}

// `fct` through `four_step_large` (inverse default = 1/`n2` in P2, custom `fct` all
// in P4): the same three properties at a `four_step_large`-routed f64 size.
TEST_CASE("fct: scale folding through four_step_large (double)", "[fct][fourstep]") {
    constexpr std::size_t N = 1048576;
    const auto in = make_signal<double>(N);
    admiral::plan<double> p(N);

    std::vector<std::complex<double>> a = in, b = in;
    p.inverse(admiral::span(a));                     // `nullopt` -> 1/N, folded into P2/P4
    p.inverse(admiral::span(b), 1.0 / double(N));    // explicit default, same fold site
    for (std::size_t i = 0; i < N; ++i) REQUIRE(a[i] == b[i]);

    std::vector<std::complex<double>> base = in, scaled = in;
    p.forward(admiral::span(base), 1.0);
    p.forward(admiral::span(scaled), 2.0);
    for (auto& x : base) x *= 2.0;
    require_close(base, scaled, fft_tol<double>(2));
}
