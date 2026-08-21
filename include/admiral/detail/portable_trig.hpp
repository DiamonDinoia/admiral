#pragma once

// ----------------------------------------------------------------------------
// portable_trig: sincos for rational turn fractions, at plan construction.
//
// FFT twiddles are exp(+/-2*pi*i*num/den) with INTEGER num,den. Argument
// reduction is exact integer arithmetic (reduce mod den, round to nearest
// quadrant) and leaves a residual in [-pi/4,pi/4] for the polynomial. No
// Payne-Hanek needed, and more accurate than it for large arguments.
// Polynomial kernel (Horner in u=t^2 on [0,(pi/4)^2]) vendored from
// DiamonDinoia/polyfit (examples/portable_trig.hpp).
//
// Compile-time twiddles use ct_sincos_turns in ct_math.hpp.
// ----------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <numbers>
#include <utility>

#include <poet/poet.hpp>

namespace admiral {
namespace detail {
namespace portable_trig {

using std::numbers::pi;

// Minimax Horner coefficients in u=t^2, valid |t|<=pi/4.
inline constexpr std::array<double, 6> sin_coeffs = {
    0x1.5e0f86a545d1fp-33, -0x1.ae6015d2aa2ap-26, 0x1.71de379c19d39p-19,
    -0x1.a01a019e7d0c4p-13, 0x1.1111111110afep-7, -0x1.5555555555554p-3,
};
inline constexpr std::array<double, 6> cos_coeffs = {
    0x1.1c0948cd3683ep-29,  -0x1.27e0fe56c4828p-22, 0x1.a019fc8d5ba89p-16,
    -0x1.6c16c1692bdf3p-10, 0x1.5555555554198p-5,   -0x1.ffffffffffffap-2,
};

// {sin(t), cos(t)} for reduced angle t in [-pi/4, pi/4].
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

// {sin(theta), cos(theta)} for theta = 2*pi * num/den; exact integer reduction.
// den > 0, num in [0, den).
constexpr std::pair<double, double> sincos_reduced_turns(std::size_t num, std::size_t den) noexcept {
    // Nearest quadrant q; residual = num/den - q/4 turns, in [-den/2, den/2].
    const std::size_t four_num = 4 * num;
    const std::size_t q = (four_num + den / 2) / den;   // round-to-nearest in [0, 4]

    // Residual angle = 2*pi * (four_num - q*den)/(4*den) in [-pi/4, pi/4]. The
    // numerator is signed, so the code forms it in double: both operands are integers
    // below 2^53, hence the subtraction is exact.
    const double residual = (2.0 * pi) *
                            (static_cast<double>(four_num) - static_cast<double>(q * den)) /
                            static_cast<double>(4 * den);
    const auto sc = reduced_sincos(residual);

    // Quadrant remap: rotate (sin,cos) by q*pi/2.
    switch (q & 3) {
        case 0:  return {sc.first, sc.second};
        case 1:  return {sc.second, -sc.first};
        case 2:  return {-sc.first, -sc.second};
        default: return {-sc.second, sc.first};  // q == 3
    }
}

// {sin(theta), cos(theta)} for theta = (Forward ? -1 : +1) * 2*pi * num/den.
// num, den: non-negative integer magnitudes; sign from Forward. den > 0.
template<bool Forward>
constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den) noexcept {
    num %= den;                              // 0 <= num < den
    if constexpr (Forward) {
        if (num != 0) num = den - num;       // negate the turn fraction, keep it in [0, den)
    }
    return sincos_reduced_turns(num, den);
}

// Runtime-direction overload for non-template-param direction paths.
constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den,
                                                 bool forward) noexcept {
    return forward ? sincos_turns<true>(num, den) : sincos_turns<false>(num, den);
}

}  // namespace portable_trig
}  // namespace detail
}  // namespace admiral

