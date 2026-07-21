#pragma once

// ----------------------------------------------------------------------------
// portable_trig — minimal sincos for rational turn fractions.
//
// The polynomial kernel (Horner coefficients on u = t^2 over [0, (pi/4)^2]) is
// vendored from DiamonDinoia/polyfit (examples/portable_trig.hpp); all polyfit/
// poet/macro dependencies were purged.
//
// FFT twiddles are always exp(+/- 2*pi*i * num/den) with INTEGER num, den, so
// argument reduction is done in EXACT integer arithmetic (reduce num mod den,
// then round to the nearest quadrant), leaving a residual in [-pi/4, pi/4] that
// the polynomial evaluates at full precision. This avoids the floating-point
// Payne-Hanek range reduction entirely and is both faster and more accurate
// than feeding a large angle to a generic sincos.
//
// This is a runtime helper used during plan construction to populate twiddle
// tables. Compile-time codelet twiddles use ct_sincos_turns in fft_codelet.h.
// ----------------------------------------------------------------------------

#include <array>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <utility>

#include <poet/poet.hpp>

namespace admiral {
namespace detail {
namespace portable_trig {

using std::numbers::pi;

// Minimax-style Horner coefficients in u = t^2, valid for |t| <= pi/4.
inline constexpr std::array<double, 6> sin_coeffs = {
    0x1.5e0f86a545d1fp-33, -0x1.ae6015d2aa2ap-26, 0x1.71de379c19d39p-19,
    -0x1.a01a019e7d0c4p-13, 0x1.1111111110afep-7, -0x1.5555555555554p-3,
};
inline constexpr std::array<double, 6> cos_coeffs = {
    0x1.1c0948cd3683ep-29,  -0x1.27e0fe56c4828p-22, 0x1.a019fc8d5ba89p-16,
    -0x1.6c16c1692bdf3p-10, 0x1.5555555554198p-5,   -0x1.ffffffffffffap-2,
};

// {sin(t), cos(t)} for the reduced angle t in [-pi/4, pi/4].
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

// {sin(theta), cos(theta)} where theta = 2*pi * num/den, exact integer
// reduction; den must be > 0, num in [0, den).
constexpr std::pair<double, double> sincos_reduced_turns(long long num, long long den) noexcept {
    // Nearest quadrant q (q/4 of a turn). residual turns = num/den - q/4.
    const long long four_num = 4 * num;
    long long q = (four_num + den / 2) / den;        // round-to-nearest in [0, 4]
    const long long r = four_num - q * den;          // in [-den/2, den/2]
    q &= 3;

    // residual angle = 2*pi * r/(4*den) in [-pi/4, pi/4].
    const double residual = (2.0 * pi) *
                            (static_cast<double>(r) / static_cast<double>(4 * den));
    const auto sc = reduced_sincos(residual);  // {sin, cos}

    // Quadrant remap: rotate (sin,cos) by q*pi/2.
    switch (q) {
        case 0:  return {sc.first, sc.second};
        case 1:  return {sc.second, -sc.first};
        case 2:  return {-sc.first, -sc.second};
        default: return {-sc.second, sc.first};  // q == 3
    }
}

// {sin(theta), cos(theta)} where theta = (Forward ? -1 : +1) * 2*pi * num/den.
// num and den are non-negative integer magnitudes (the natural size_t products
// of FFT index arithmetic); the transform direction supplies the sign, so call
// sites need no casts. den must be > 0.
template<bool Forward, typename Int>
constexpr std::pair<double, double> sincos_turns(Int num, Int den) noexcept {
    static_assert(std::is_integral_v<Int>, "sincos_turns requires integral arguments");
    const long long d = static_cast<long long>(den);
    long long n = static_cast<long long>(num % den);  // 0 <= n < den
    if constexpr (Forward) {
        if (n != 0) n = d - n;  // negate the turn fraction, keep it in [0, den)
    }
    return sincos_reduced_turns(n, d);
}

// Runtime-direction overload for paths whose direction is not a template param.
template<typename Int>
constexpr std::pair<double, double> sincos_turns(Int num, Int den, bool forward) noexcept {
    return forward ? sincos_turns<true>(num, den) : sincos_turns<false>(num, den);
}

}  // namespace portable_trig
}  // namespace detail
}  // namespace admiral

