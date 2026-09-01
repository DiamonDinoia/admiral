#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include <poet/poet.hpp>
#include "cxx_compat.hpp"

namespace admiral {
namespace detail {
namespace portable_trig {

using detail::numbers::pi;

inline constexpr std::array<double, 6> sin_coeffs = {
    0x1.5e0f86a545d1fp-33, -0x1.ae6015d2aa2ap-26, 0x1.71de379c19d39p-19,
    -0x1.a01a019e7d0c4p-13, 0x1.1111111110afep-7, -0x1.5555555555554p-3,
};
inline constexpr std::array<double, 6> cos_coeffs = {
    0x1.1c0948cd3683ep-29,  -0x1.27e0fe56c4828p-22, 0x1.a019fc8d5ba89p-16,
    -0x1.6c16c1692bdf3p-10, 0x1.5555555554198p-5,   -0x1.ffffffffffffap-2,
};

constexpr std::pair<double, double> reduced_sincos(double t) noexcept {
    const double t2 = t * t;
    const double t3 = t2 * t;
    double sp = sin_coeffs[0];
    double cp = cos_coeffs[0];
    poet::static_for<1, 6>([&](auto i) {
        sp = sp * t2 + sin_coeffs[i];
        cp = cp * t2 + cos_coeffs[i];
    });
    return {sp * t3 + t, cp * t2 + 1.0};
}

constexpr std::pair<double, double> sincos_reduced_turns(std::size_t num, std::size_t den) noexcept {
    const std::size_t four_num = 4 * num;
    const std::size_t q = (four_num + den / 2) / den;

    const double residual = (2.0 * pi) *
                            (static_cast<double>(four_num) - static_cast<double>(q * den)) /
                            static_cast<double>(4 * den);
    const auto sc = reduced_sincos(residual);

    switch (q & 3) {
        case 0:  return {sc.first, sc.second};
        case 1:  return {sc.second, -sc.first};
        case 2:  return {-sc.first, -sc.second};
        default: return {-sc.second, sc.first};
    }
}

template<bool Forward>
constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den) noexcept {
    num %= den;
    if constexpr (Forward) {
        if (num != 0) num = den - num;
    }
    return sincos_reduced_turns(num, den);
}

constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den,
                                                 bool forward) noexcept {
    return forward ? sincos_turns<true>(num, den) : sincos_turns<false>(num, den);
}

}
}
}
