// DCT/DST (`plan_r2r`) against long-double direct sums of FFTW's four r2r kinds.
// Both directions plus the round trip per kind: a wrong index map is symmetric
// (DCT-III is DCT-II read backwards) and can round-trip while both directions
// are wrong.
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <admiral/admiral.hpp>
#include "utils/reference.hpp"

namespace {

// FFTW's unnormalized definitions, verbatim: REDFT10, REDFT01, RODFT10, RODFT01.
template<typename T>
std::vector<long double> oracle(admiral::r2r_kind kind, const std::vector<T>& x) {
    const std::size_t N = x.size();
    const long double pi = admiral::detail::numbers::pi_v<long double>;
    const auto ld = [](auto v) { return static_cast<long double>(v); };
    std::vector<long double> y(N);
    for (std::size_t k = 0; k < N; ++k) {
        long double s = 0;
        switch (kind) {
            case admiral::r2r_kind::dct2:
                for (std::size_t n = 0; n < N; ++n)
                    s += 2 * ld(x[n]) * std::cos(pi * ld(k) * ld(2 * n + 1) / ld(2 * N));
                break;
            case admiral::r2r_kind::dct3:  // x[0] + 2*sum_{n>=1}
                s = ld(x[0]);
                for (std::size_t n = 1; n < N; ++n)
                    s += 2 * ld(x[n]) * std::cos(pi * ld(n) * ld(2 * k + 1) / ld(2 * N));
                break;
            case admiral::r2r_kind::dst2:
                for (std::size_t n = 0; n < N; ++n)
                    s += 2 * ld(x[n]) * std::sin(pi * ld(k + 1) * ld(2 * n + 1) / ld(2 * N));
                break;
            case admiral::r2r_kind::dst3:  // (-1)^k x[N-1] + 2*sum_{n<N-1}
                s = (k % 2 == 0 ? 1.0L : -1.0L) * ld(x[N - 1]);
                for (std::size_t n = 0; n + 1 < N; ++n)
                    s += 2 * ld(x[n]) * std::sin(pi * ld(n + 1) * ld(2 * k + 1) / ld(2 * N));
                break;
        }
        y[k] = s;
    }
    return y;
}

template<typename T>
constexpr double r2r_tol() {
    return std::is_same_v<T, float> ? 2e-4 : 1e-11;
}

}  // namespace

TEMPLATE_TEST_CASE("DCT/DST match the direct sum in both directions", "[r2r]", float, double) {
    using admiral::r2r_kind;
    const r2r_kind kinds[] = {r2r_kind::dct2, r2r_kind::dct3, r2r_kind::dst2, r2r_kind::dst3};

    // A sweep, not a hand-picked list. The index maps of the four kinds differ at
    // the parity boundary and at N == 1. At the boundary, k == N/2 is the one index
    // where the two writes per k collide. Every small N is its own case. The larger
    // sizes cover routes a small N never reaches: prime (251), prime square (121),
    // pentanomial (100, 1000), 7-smooth or pow2 chains (96, 128, 210, 256, 360,
    // 512), and N/2 prime > 11 (26, 46), which forces the scalar r2c even path
    // instead of the batched one.
    for (std::size_t N = 1; N <= 64; ++N) {
        for (const r2r_kind kind : kinds) {
            const auto x = make_real_input<TestType>(N, 12345u + static_cast<unsigned>(N));
            admiral::plan_r2r<TestType> p(N, kind);
            std::vector<TestType> y(N), rt(N);
            p.forward(x.data(), y.data());
            p.inverse(y.data(), rt.data());
            CAPTURE(N, static_cast<int>(kind));
            CHECK(relerrtwonorm(oracle(kind, x), y) < r2r_tol<TestType>());
            CHECK(relerrtwonorm(x, rt) < r2r_tol<TestType>());
        }
    }
    for (const std::size_t N : {26u, 46u, 96u, 100u, 121u, 128u, 210u, 251u, 256u, 360u,
                                512u, 1000u}) {
        for (const r2r_kind kind : kinds) {
            const auto x = make_real_input<TestType>(N, 999u + static_cast<unsigned>(N));
            admiral::plan_r2r<TestType> p(N, kind);
            std::vector<TestType> y(N), rt(N);
            p.forward(admiral::span<const TestType>(x), admiral::span<TestType>(y));
            p.inverse(admiral::span<const TestType>(y), admiral::span<TestType>(rt));
            CAPTURE(N, static_cast<int>(kind));
            CHECK(relerrtwonorm(oracle(kind, x), y) < r2r_tol<TestType>());
            CHECK(relerrtwonorm(x, rt) < r2r_tol<TestType>());
        }
    }
}

TEMPLATE_TEST_CASE("Batched r2r rows equal the same rows one at a time", "[r2r]", float, double) {
    using admiral::r2r_kind;
    // Not bitwise: r2c has a batched tile path that `rows == 1` never takes, so the
    // batched and single-row runs legitimately round differently. An indexing bug in
    // the batch path is orders of magnitude larger, and the loose bound catches it.
    for (const std::size_t N : {5u, 8u, 12u, 17u, 64u}) {
        for (const std::size_t rows : {2u, 3u, 9u}) {
            for (const r2r_kind kind : {r2r_kind::dct2, r2r_kind::dct3, r2r_kind::dst2,
                                        r2r_kind::dst3}) {
                const auto x =
                    make_real_input<TestType>(N * rows, 7u + static_cast<unsigned>(N * rows));
                admiral::plan_r2r<TestType> batched(N, kind, rows);
                admiral::plan_r2r<TestType> single(N, kind);
                std::vector<TestType> got(N * rows), want(N * rows);
                batched.forward(x.data(), got.data());
                for (std::size_t r = 0; r < rows; ++r)
                    single.forward(x.data() + r * N, want.data() + r * N);
                CAPTURE(N, rows, static_cast<int>(kind));
                CHECK(relerrtwonorm(want, got) < r2r_tol<TestType>());
            }
        }
    }
}

TEMPLATE_TEST_CASE("Threaded r2r equals the serial plan", "[r2r]", float, double) {
    using admiral::r2r_kind;
    // `nthreads > 1` with `rows > 1` is the only configuration that constructs the
    // pool at all. The pool threads only the r2c/c2r tile loop over rows. The
    // comparison against the serial plan of the same shape is not bitwise, because
    // the tile loop partitions differently per thread count.
    constexpr std::size_t N = 40, rows = 12;
    for (const r2r_kind kind : {r2r_kind::dct2, r2r_kind::dct3, r2r_kind::dst2, r2r_kind::dst3}) {
        const auto x = make_real_input<TestType>(N * rows, 88u + static_cast<unsigned>(kind));
        admiral::plan_r2r<TestType> mt(N, kind, rows, {4});
        admiral::plan_r2r<TestType> st(N, kind, rows);
        std::vector<TestType> got(N * rows), want(N * rows), rt(N * rows);
        mt.forward(x.data(), got.data());
        st.forward(x.data(), want.data());
        mt.inverse(got.data(), rt.data());
        CAPTURE(static_cast<int>(kind));
        CHECK(relerrtwonorm(want, got) < r2r_tol<TestType>());
        CHECK(relerrtwonorm(x, rt) < r2r_tol<TestType>());
    }
}

TEMPLATE_TEST_CASE("r2r fct replaces the default scale", "[r2r]", float, double) {
    constexpr std::size_t N = 24;
    const auto x = make_real_input<TestType>(N, 4242u);
    std::vector<TestType> a(N), b(N);

    // `fct` is an exact output scale: the value REPLACES the default, so the output
    // scales linearly. `fct` == the default reproduces the default path bit for bit.
    admiral::plan_r2r<TestType> p2(N, admiral::r2r_kind::dct2);
    p2.forward(x.data(), a.data());
    p2.forward(x.data(), b.data(), TestType(1));   // `dct2`'s default already is 1
    CHECK(a == b);
    p2.forward(x.data(), b.data(), TestType(3));
    for (std::size_t i = 0; i < N; ++i) CHECK(b[i] == TestType(3) * a[i]);

    // The 2N that separates the library's convention from FFTW's is the type-3
    // forward's default, nothing else. At `fct` == 1, that direction is the type-2
    // kind's exact inverse instead.
    admiral::plan_r2r<TestType> p3(N, admiral::r2r_kind::dct3);
    p3.forward(x.data(), a.data());
    p3.forward(x.data(), b.data(), TestType(1));
    for (std::size_t i = 0; i < N; ++i) CHECK(a[i] == static_cast<TestType>(2 * N) * b[i]);
}

TEMPLATE_TEST_CASE("r2r runs in place", "[r2r]", float, double) {
    using admiral::r2r_kind;
    // Both cores (`dct2_rows`, `dct3_rows`) stage the whole input into scratch
    // before storing anything, so out == in is legal at any `rows`. The comparison
    // is bitwise: in-place and out-of-place run the same code. A pass-interleaving
    // reorder would corrupt the in-place run and nothing else.
    for (const std::size_t N : {7u, 16u, 45u}) {
        for (const r2r_kind kind : {r2r_kind::dct2, r2r_kind::dct3, r2r_kind::dst2,
                                    r2r_kind::dst3}) {
            const auto x = make_real_input<TestType>(N * 3, 31u + static_cast<unsigned>(N));
            auto inplace = x;
            std::vector<TestType> oop(N * 3);
            admiral::plan_r2r<TestType> p(N, kind, 3);
            p.forward(x.data(), oop.data());
            p.forward(inplace.data(), inplace.data());
            CAPTURE(N, static_cast<int>(kind));
            CHECK(inplace == oop);
        }
    }
}

TEST_CASE("plan_r2r rejects empty sizes and mismatched spans", "[r2r]") {
    CHECK_THROWS_AS(admiral::plan_r2r<double>(0, admiral::r2r_kind::dct2), admiral::size_error);
    CHECK_THROWS_AS(admiral::plan_r2r<double>(8, admiral::r2r_kind::dct2, 0),
                    admiral::size_error);
    admiral::plan_r2r<double> p(8, admiral::r2r_kind::dct2, 2);
    CHECK(p.size() == 16);
    std::vector<double> a(16), b(8);
    CHECK_THROWS_AS(p.forward(admiral::span<const double>(a), admiral::span<double>(b)),
                    admiral::size_error);
}
