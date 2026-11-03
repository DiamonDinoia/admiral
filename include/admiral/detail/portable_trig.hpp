#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#  include <intrin.h>
#endif

#include "ct_math.hpp"
#include "scratch.hpp"

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

// Runtime roots of unity for plan tables. Accuracy contract: the transform's error sits at
// machine precision in norm (the relative L2 transform error stays within 1.25x of stock
// pocketfft's and inside 4 * eps * log2(size), gated by test/plans/test_norm_gate.cpp).
// The old bit-identical-to-the-fold contract is dropped for these runtime twiddles: the fold
// kept about 64-bit precision per entry at x87-series cost (~40 ns per angle), and the gate
// only needs libm-grade accuracy.
//
// The memo is a two-level circle, the split pocketfft's sincos_2pibyn uses:
// w^(hi*q + lo) = w^(hi*q) * w^lo with q a power of two nearest sqrt(den). One table build
// costs 2*sqrt(den) libm sincos calls (~8 ns per pair), one lookup costs one complex product
// of two table entries (~2-3 ulp from the libm values). The table-building code creates one
// turns_memo and passes it down: the memo lives for one plan build, is rebuilt when the den
// changes, and holds at most 2*sqrt(2^24) doubles.

struct turns_memo {
    // Two-level lookup covers dens 8..2^24.
    static constexpr std::size_t kTwoLevelDenMax = std::size_t{1} << 24;

    // num % den without a hardware divide, with one exactness fixup: m = floor(2^W/den)+1,
    // W the word bits, makes floor(num*m/2^W) either the true quotient or one above it, and
    // never below the true quotient (m*den - 2^W < den, so q - q_true <= num/2^W < 1). The
    // clamp restores q <= q_true, leaving the remainder in [0, 2*den) for one final fixup.
    std::size_t reduce(std::size_t num, std::size_t den) {
        if (num < den) return num;
        if (den != mod_den_) {
            mod_den_ = den;
            // max/den lands one below floor(2^W/den) exactly where den is a power of two.
            mod_minv_ = ~std::size_t{0} / den + 1 + ((den & (den - 1)) == 0 ? 1u : 0u);
        }
        std::size_t q = mul_hi(num, mod_minv_);
        q -= (q != 0 && q * den > num);
        std::size_t r = num - q * den;
        if (r >= den) r -= den;
        return r;
    }

    std::pair<double, double> get_two(std::size_t num, std::size_t d) {
        // One table build also serves every den that divides it: w^(num/d) = w^(num*s/D)
        // with s = D/d. r2c pairs N with N/2, so a smooth plan builds its tables once. The
        // scale D/d is cached per den: every dif row of one pass calls with the same d, and
        // a hardware divide per call would cost as much as the product itself.
        std::size_t num_d = num;
        if (two_den != d) {
            if (d != scale_den_) {
                scale_map_ = two_den % d == 0 ? two_den / d : 0;
                scale_den_ = d;
            }
            if (scale_map_ != 0)
                num_d = num * scale_map_;
            else
                build_two(d);
        }
        // Quadrant folding: the table holds one quadrant (den dividable by 4) and the
        // axis swap/sign restores the full circle exactly; the libm build is 4x smaller.
        // quad is num_d / quarter with quarter = den/4: four range tests, no divide.
        std::size_t rem = num_d, quad = 0;
        if (two_fold != 0) {
            const std::size_t quarter = two_den >> 2;
            quad = static_cast<std::size_t>((num_d >= quarter) +
                   (num_d >= 2 * quarter) + (num_d >= 3 * quarter));
            rem = num_d - quad * quarter;
        }
        const std::size_t lo = rem & (two_q - 1);
        const std::size_t hi = rem >> two_shift;
        const double ls = two_lo_s[lo], lc = two_lo_c[lo];
        const double hs = two_hi_s[hi], hc = two_hi_c[hi];
        // One complex multiply per entry, pocketfft's sincos_2pibyn shape; the TwoProd tail
        // moved the entry from ~1.5-2 ulps to ~1 ulp and cost about 20 extra scalar ops in a
        // per-entry dependent chain, which dominated plan construction cost (ctor profile:
        // get_two 33%). The build path reads each entry once; the norm rule stays under
        // 40% of tolerance, so the extra absolute ulp is well below the accuracy gate.
        double s = ls * hc + lc * hs;
        double c = lc * hc - ls * hs;
        if (quad == 1 || quad == 3) {
            const double t = s;
            s = (quad == 1) ? c : -c;
            c = (quad == 1) ? -t : t;
        } else if (quad == 2) {
            s = -s;
            c = -c;
        }
        return {s, c};
    }

    // Snapshot for a vectorized one-row fill: every get_two-side table decision runs once
    // here (build on a miss, divisor scale otherwise), so the fill walks the reduced index
    // per block and reads plain tables. quarter is two_den when the den keeps the full
    // circle, which parks the fold compares at false.
    struct two_row_view {
        const double* lo_s;
        const double* lo_c;
        const double* hi_s;
        const double* hi_c;
        std::size_t den;      // index-space modulus (two_den)
        std::size_t qmask;    // two_q - 1
        unsigned shift;       // two_shift
        std::size_t quarter;  // fold span, or den when no fold
        std::size_t step_d;   // step*scale mod den
    };
    [[nodiscard]] two_row_view row_view(std::size_t step, std::size_t d) {
        std::size_t scale = 1;
        if (two_den != d) {
            if (d != scale_den_) {
                scale_map_ = two_den % d == 0 ? two_den / d : 0;
                scale_den_ = d;
            }
            if (scale_map_ != 0)
                scale = scale_map_;
            else
                build_two(d);
        }
        return {two_lo_s.data(), two_lo_c.data(), two_hi_s.data(), two_hi_c.data(),
                two_den, two_q - 1, two_shift,
                two_fold != 0 ? (two_den >> 2) : two_den, reduce(step * scale, two_den)};
    }

    // External backing the plan carves from its single block: table_elems(den) must cover the
    // largest den the memo will serve (this plan's top den; divisor dens reuse the build).
    void attach(double* base, std::size_t elems) {
        ext_base_ = base;
        ext_elems_ = elems;
        ext_off_ = 0;
    }
    [[nodiscard]] static std::size_t table_elems(std::size_t d) {
        if (d % 4u == 0) d /= 4u;
        std::size_t shift = 0;
        while ((std::size_t{1} << shift) * (std::size_t{1} << shift) < d) ++shift;
        const std::size_t q = std::size_t{1} << shift;
        return 2 * (q + (d + q - 1) / q);
    }

   private:
    // The exact word x word -> high-word product the reduce path needs, which ISO C++ does
    // not name at 64 bits: MSVC gets _umul128, the GNU compilers their 128-bit type
    // (cxx_compat.hpp's shape), and 32-bit words the long long product ISO C++ names.
    [[nodiscard]] static std::size_t mul_hi(std::size_t a, std::size_t b) {
#if defined(_MSC_VER) && !defined(__clang__)
        static_assert(sizeof(std::size_t) == 8, "_umul128 needs 64-bit words");
        unsigned long long hi = 0;
        [[maybe_unused]] const unsigned long long lo = _umul128(a, b, &hi);
        return static_cast<std::size_t>(hi);
#elif defined(__SIZEOF_INT128__)
        __extension__ typedef unsigned __int128 u128;
        return static_cast<std::size_t>((static_cast<u128>(a) * b) >> (sizeof(std::size_t) * 8));
#else
        static_assert(sizeof(std::size_t) <= 4, "64-bit words need __int128 or _umul128");
        return static_cast<std::size_t>(
            (static_cast<unsigned long long>(a) * b) >> (sizeof(std::size_t) * 8));
#endif
    }

    // sin/cos of a first-octant residual angle without the platform CRT: the Taylor tails of
    // the ct_dd fold (ct_math.hpp sincos_turns_fold_dbl) at double precision, alternating sums
    // of the term recurrence with reciprocal steps (one multiply per term, no divide). Table
    // accuracy then carries no libm rounding: the MSVC CRT's ~1 ulp error put the memo
    // contract at 2.5 ulp and the transform norm above tolerance (CI 125981420115). One
    // merged loop serves both tails; each tail stops at its own term threshold, so a small
    // angle pays only the terms it needs. The term sequences are bit-identical to the
    // separate-loop fold.
    static void series_sincos(double x, double& s, double& c) {
        static constexpr double kSinStep[] = {1.0 / 20.0, 1.0 / 42.0, 1.0 / 72.0,
                                              1.0 / 110.0, 1.0 / 156.0, 1.0 / 210.0,
                                              1.0 / 272.0, 1.0 / 342.0};
        static constexpr double kCosStep[] = {1.0 / 12.0, 1.0 / 30.0, 1.0 / 56.0,
                                              1.0 / 90.0, 1.0 / 132.0, 1.0 / 182.0,
                                              1.0 / 240.0, 1.0 / 306.0};
        const double x2 = x * x;
        double t = x * x2 * (1.0 / 6.0), ts = t;
        double u = x2 * 0.5, tu = u;
        for (unsigned k = 2; k <= 9 && (t > 0x1p-56 || u > 0x1p-56); ++k) {
            if (t > 0x1p-56) {
                t *= x2 * kSinStep[k - 2];
                ts = (k & 1) ? ts + t : ts - t;
            }
            if (u > 0x1p-56) {
                u *= x2 * kCosStep[k - 2];
                tu = (k & 1) ? tu + u : tu - u;
            }
        }
        s = x - ts;
        c = 1.0 - tu;
    }

    // Table of w^(k*step/d) for k = 0..count-1, evaluated like pocketfft's sincos_2pibyn:
    // every index maps to a first-octant residual angle below pi/4 and mirrors analytically,
    // so no libm call ever sees a big argument (a 2*pi-scale argument quantizes at ~1.6
    // absolute ulp and the transform norm feels it). The phase pi/(4d) is double-double:
    // a lone double ph rounds once and every base angle inherits that bias coherently, which
    // the transform reads as a systematic rotation. Base error is ~1 absolute ulp.
    void libm_base(std::size_t step, std::size_t count, std::size_t d, double* out_s,
                   double* out_c) {
        const ct_dd phdd = ct_pi<ct_dd> / ct_dd{4.0 * static_cast<double>(d), 0.0};
        const double phh = phdd.hi, phl = phdd.lo;
        // sqrt(1/2) as double-double: the diagonal mirror s2*t stays inside one ulp of the
        // plain double product (each mirror would otherwise carry s2's own rounding).
        constexpr double s2h = 0x1.6a09e667f3bcdp-1;
        constexpr double s2l = -4.8336466567264567e-17;
        const auto diag = [](double t) { return std::fma(s2h, t, s2l * t); };
        for (std::size_t k = 0; k < count; ++k) {
            const std::size_t m = k * step;
            const std::size_t oct = (8 * m) / d;
            const std::size_t rem = 8 * m - oct * d;
            double rs, rc;
            series_sincos(std::fma(static_cast<double>(rem), phl,
                                   static_cast<double>(rem) * phh),
                          rs, rc);
            double s, c;
            switch (oct & 7) {
                case 0: s = rs;                     c = rc;                     break;
                case 1: s = diag(rc + rs);          c = diag(rc - rs);          break;
                case 2: s = rc;                     c = -rs;                    break;
                case 3: s = diag(rc - rs);          c = -diag(rc + rs);         break;
                case 4: s = -rs;                    c = -rc;                    break;
                case 5: s = -diag(rc + rs);         c = -diag(rc - rs);         break;
                case 6: s = -rc;                    c = rs;                     break;
                default: s = -diag(rc - rs);        c = diag(rc + rs);          break;
            }
            out_s[k] = s;
            out_c[k] = c;
        }
    }

    void build_two(std::size_t d) {
        two_den = d;
        scale_den_ = 0;  // divisor cache tied to the old table den
        scale_map_ = 0;
        // Quadrant fold: dens dividable by 4 store one quadrant and rebuild the rest with an
        // exact axis swap/sign at read time. Odd or 2-mod-4 dens keep the full circle.
        std::size_t dd = d;
        two_fold = 0;
        if (d % 4u == 0) {
            dd = d / 4;
            two_fold = 1;
        }
        two_shift = 0;
        while ((std::size_t{1} << two_shift) * (std::size_t{1} << two_shift) < dd) ++two_shift;
        two_q = std::size_t{1} << two_shift;
        const std::size_t nh = (dd + two_q - 1) / two_q;
        // One block per table build, where the plan attached a region carved from its own
        // arena or from this unique_ptr fallback (as the earlier per-side vector storage).
        const std::size_t need = 2 * (two_q + nh);
        double* base;
        if (ext_base_ != nullptr && ext_off_ + need <= ext_elems_) {
            base = ext_base_ + ext_off_;
            ext_off_ += need;
        } else {
            owned_ = std::make_unique<double[]>(need);
            base = owned_.get();
        }
        two_lo_s = {base, two_q};
        two_lo_c = {base + two_q, two_q};
        two_hi_s = {base + 2 * two_q, nh};
        two_hi_c = {base + 2 * two_q + nh, nh};
        // Phase den stays d; the fold only truncates the table extent to one quadrant.
        libm_base(1, two_q, d, two_lo_s.data(), two_lo_c.data());      // w^(i/d),   i < q
        libm_base(two_q, nh, d, two_hi_s.data(), two_hi_c.data());     // w^(j*q/d), j < nh
    }

    std::size_t two_den = 0;
    unsigned two_shift = 0;
    std::size_t two_q = 0;
    unsigned two_fold = 0;
    std::size_t scale_den_ = 0;
    std::size_t scale_map_ = 0;
    std::size_t mod_den_ = 0;
    std::size_t mod_minv_ = 0;
    dif_buf<double> two_lo_s, two_lo_c, two_hi_s, two_hi_c;
    std::unique_ptr<double[]> owned_;
    double* ext_base_ = nullptr;
    std::size_t ext_elems_ = 0, ext_off_ = 0;
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
        // fold, whose own bits differ there. The ld_full_precision probe gates the read
        // for the same reason it gates the memo and the fold below: a degraded run-time
        // long double (valgrind, x87 precision control) runs the double fold for the same
        // dens, and a run-time result must equal the fold of the same run-time mode. The
        // branch carries its own reduce and flip so the pow2 path never pays an idiv; the
        // generic reduce below repeats them for the rest.
        if (den >= 8 && den <= kPow2CircleDen && (den & (den - 1)) == 0 &&
            ld_full_precision()) {
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
    if (memo != nullptr && den >= 8 && den <= turns_memo::kTwoLevelDenMax) {
        num = memo->reduce(num, den);
        if constexpr (Forward) {
            if (num != 0) num = den - num;
        }
        return memo->get_two(num, den);
    }
    num %= den;
    if constexpr (Forward) {
        if (num != 0) num = den - num;
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
