#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ct_math.hpp"

namespace admiral {
namespace detail {
namespace portable_trig {

// Runtime twin of the compile-time fold: roots of unity at ct_dd precision rounded once to
// double, so a table built here and a constant folded into a codelet are the same value.
//
// Plan tables call this per entry. Numerators that differ by a multiple of
// den / gcd(8, den) share one residual angle, and sincos_octant_mix turns one series
// evaluation into all eight octant outputs, so a whole table costs one series per residual
// class. The table-building code creates one turns_memo and passes it down: the memo lives
// for one table build and carries that sharing across its entries. The memo stores
// sincos_octant_mix's own results: the values stay bitwise identical to the fold's. long
// double is double or composite on MSVC and Apple, or degraded at run time (valgrind), where
// the fold route mixes differently; those keep the per-call fold.
// ponytail: capped at 4096 residual classes per den; larger dens (bluestein paddings) fall
// back to the per-call fold.

struct turns_memo {
    static constexpr std::size_t kCap = 4096;

    std::pair<double, double> get(std::size_t num, std::size_t d) {
        if (den != d) {
            den = d;
            period = d / std::gcd<std::size_t>(8, d);
            pow2 = d >= 8 && (d & (d - 1)) == 0;
            split = 0;
            for (std::size_t p = period; p > 1; p >>= 1) ++split;
            out = std::unique_ptr<std::array<double, 8>[]>(new std::array<double, 8>[period]);
            have.assign(period, 0);
        }
        const std::size_t oct = pow2 ? num >> split : (8 * num) / d;
        const std::size_t slot = pow2 ? num & (period - 1) : num % period;
        if (!have[slot]) {
            const std::size_t rem = 8 * num - oct * d;
            const long double res =
                static_cast<long double>(rem) *
                (ct_pi<long double> /
                 (static_cast<long double>(4) * static_cast<long double>(d)));
            const ct_sincos_v<long double> r = sincos_small_eval<long double>(res);
            // Octant j + 4 is exactly -octant j (both operands of the mix negate), so the top
            // half never needs storing or computing.
            for (std::size_t j = 0; j < 4; ++j) {
                const ct_sincos_v<long double> w = sincos_octant_mix<long double>(j, r);
                out[slot][2 * j] = static_cast<double>(w.s);
                out[slot][2 * j + 1] = static_cast<double>(w.c);
            }
            have[slot] = 1;
        }
        const std::size_t o = oct & 3;
        const double s = out[slot][2 * o], c = out[slot][2 * o + 1];
        return oct < 4 ? std::pair<double, double>{s, c} : std::pair<double, double>{-s, -c};
    }

   private:
    std::size_t den = 0;
    std::size_t period = 0;
    bool pow2 = false;
    unsigned split = 0;                             // pow2 den: oct = num >> split
    std::unique_ptr<std::array<double, 8>[]> out;   // per class: 4 octants x (s, c), filled under have
    std::vector<unsigned char> have{};              // per class
};

template<bool Forward>
inline std::pair<double, double> sincos_turns(std::size_t num, std::size_t den,
                                              turns_memo* memo = nullptr) noexcept {
    num %= den;
    if constexpr (Forward) {
        if (num != 0) num = den - num;
    }
    if constexpr (std::numeric_limits<long double>::is_iec559 &&
                  std::numeric_limits<long double>::digits > 53 &&
                  std::numeric_limits<long double>::max_exponent > 1024) {
        if (memo != nullptr && den <= turns_memo::kCap && ld_full_precision())
            return memo->get(num, den);
    }
    const ct_sincos_v<double> w = sincos_turns_eval<double>(false, num, den);
    return {w.s, w.c};
}

inline std::pair<double, double> sincos_turns(std::size_t num, std::size_t den, bool forward,
                                              turns_memo* memo = nullptr) noexcept {
    return forward ? sincos_turns<true>(num, den, memo) : sincos_turns<false>(num, den, memo);
}

}
}
}
