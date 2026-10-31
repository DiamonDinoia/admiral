#pragma once

#include <array>
#include <cfloat>
#include <cstddef>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include "ct_math.hpp"

// The circle matches the run-time fold bit for bit only where long double is the 64-bit
// mantissa format the generator folded at (x86 80-bit). The measured other formats keep
// the run-time fold: ld == double and the ppc64le double-double differ in 87448 of 262128
// cells, aarch64 binary128 in 752 (the sweep shape of test/kernels/test_twiddle_circle.cpp).
// The 3 MB table is parsed only where a plan reads it; MSVC's C1060 card limit never sees
// it. Consumers key the same switch on ADM_POW2_CIRCLE.
#if defined(LDBL_MANT_DIG) && LDBL_MANT_DIG == 64 && LDBL_MAX_EXP > 1024
#  include "twiddle_circle.hpp"
#  define ADM_POW2_CIRCLE 1
#endif

#include <version>
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
#  include <bit>
#  define ADM_CIRCLE_COUNTR_ZERO
#endif

// Local twin of macros.hpp's ADM_ALWAYS_INLINE: this header sits inside the include span
// of headers that pair macros.hpp with undef_macros.hpp, so it cannot join that pairing.
#if defined(_MSC_VER) && !defined(__clang__)
#  define ADM_QWAVE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_QWAVE_INLINE inline __attribute__((always_inline))
#else
#  define ADM_QWAVE_INLINE inline
#endif

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
            if (pow2)
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
ADM_QWAVE_INLINE std::pair<double, double> sincos_turns(std::size_t num, std::size_t den,
                                                        turns_memo* memo = nullptr) {
    if constexpr (std::numeric_limits<long double>::is_iec559) {
#ifdef ADM_POW2_CIRCLE
        // A power-of-two den reads the precomputed circle: one load replaces the series
        // fold and every run-time long double operation. The circle holds doubles folded
        // from 64-bit-mantissa long double values, so the ADM_POW2_CIRCLE switch keeps
        // the other long double formats (double, binary128, composite) on the run-time
        // fold, whose own bits differ there. The branch needs no ld_full_precision probe:
        // the read does no run-time long double arithmetic, and a degraded run-time long
        // double (valgrind, x87 precision control) leaves the committed bits alone, where
        // the memo and fold routes degrade to the double fold for the same dens. The
        // branch carries its own reduce and flip so the pow2 path never pays an idiv; the
        // generic reduce below repeats them for the rest.
        if (den >= 8 && den <= kPow2CircleDen && (den & (den - 1)) == 0) {
#ifdef ADM_CIRCLE_COUNTR_ZERO
            const unsigned l2 = static_cast<unsigned>(std::countr_zero(den));
#else
            unsigned l2 = 0;
            for (std::size_t d = den; d > 1; d >>= 1) ++l2;
#endif
            num &= den - 1;
            if constexpr (Forward) {
                if (num != 0) num = den - num;
            }
            const ct_sincos_v<double> w = pow2_circle[num << (16 - l2)];
            return {w.s, w.c};
        }
#endif
    }
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

[[nodiscard]] ADM_QWAVE_INLINE std::pair<double, double> sincos_turns(
    std::size_t num, std::size_t den, bool forward, turns_memo* memo = nullptr) {
    return forward ? sincos_turns<true>(num, den, memo) : sincos_turns<false>(num, den, memo);
}

}
}
}

#undef ADM_QWAVE_INLINE
#undef ADM_CIRCLE_COUNTR_ZERO
