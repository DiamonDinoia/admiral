#include <admiral/admiral.hpp>
#include <admiral/detail/cxx_compat.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

namespace cx = admiral::detail;

TEST_CASE("compat: span parity at this standard", "[kernels][compat]") {
    std::vector<double> v{1.0, 2.0, 3.0, 4.0};

    admiral::span a(v);
    admiral::span<const double> b(v);
    admiral::span<const double> c = a;
    REQUIRE(a.size() == 4);
    REQUIRE(a.data() == v.data());
    REQUIRE(b[2] == 3.0);
    REQUIRE(!c.empty());

    double x[3] = {5, 6, 7};
    admiral::span d(x);
    admiral::span e(x, 2);
    admiral::span f(d.begin(), d.size());
    REQUIRE(d.size() == 3);
    REQUIRE(e[1] == 6.0);
    REQUIRE(f.size() == 3);

    double s = 0.0;
    for (const double t : a) s += t;
    REQUIRE(s == 10.0);
}

TEST_CASE("compat: numbers constants match double-precision expectations", "[kernels][compat]") {
    REQUIRE(cx::numbers::pi == 3.141592653589793238462643383279502884);
    REQUIRE(cx::numbers::pi_v<double> == cx::numbers::pi);
    if constexpr (std::numeric_limits<long double>::digits >
                  std::numeric_limits<double>::digits)
        REQUIRE(cx::numbers::pi_v<long double> - static_cast<long double>(cx::numbers::pi)
                != 0.0L);
    else
        REQUIRE(cx::numbers::pi_v<long double> == static_cast<long double>(cx::numbers::pi));
}

TEST_CASE("compat: bit helpers agree with the shift-count definitions", "[kernels][compat]") {
    REQUIRE(cx::has_single_bit(8u));
    REQUIRE(!cx::has_single_bit(12u));
    REQUIRE(cx::bit_floor(13u) == 8u);
    REQUIRE(cx::bit_ceil(13u) == 16u);
    REQUIRE(cx::bit_width(13u) == 4);
    REQUIRE(cx::countr_zero(40u) == 3);
    const std::array hay{1, 2, 3};
    REQUIRE(cx::const_find(hay.begin(), hay.end(), 7) == hay.end());
    REQUIRE(cx::const_find(hay.begin(), hay.end(), 2) == hay.begin() + 1);
}

TEST_CASE("compat: span binds a temporary only through const elements", "[kernels][compat]") {
    STATIC_REQUIRE(std::is_constructible_v<admiral::span<const int>, std::vector<int>&&>);
    STATIC_REQUIRE(!std::is_constructible_v<admiral::span<int>, std::vector<int>&&>);
    STATIC_REQUIRE(std::is_constructible_v<admiral::span<int>, std::vector<int>&>);
    STATIC_REQUIRE(!std::is_constructible_v<admiral::span<int>, const std::vector<int>&>);
}
