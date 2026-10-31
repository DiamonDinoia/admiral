// Bit test of the precomputed power-of-two circle: for every power-of-two den up to 2^16
// and every k, the circle twiddle must equal the run-time series fold bit for bit. The
// derivation below repeats the index math of sincos_turns's circle branch on purpose: an
// off-by-stride or a swapped sign must fail here, and so must one flipped table bit. This
// same TU also builds under the engine's fast-math flags (test_twiddle_circle_fast), where
// the fold must fold to the same bits. The circle holds the 64-bit-mantissa long double
// fold rounded once to double; the measured folds of the other formats differ (ld ==
// double and the ppc64le double-double: 87448 of 262128 cells; aarch64 binary128: 752),
// so the sweep runs only where portable_trig.hpp reads the circle (ADM_POW2_CIRCLE). The
// ld_full_precision probe skips the sweep where the run time long double degrades
// (valgrind, x87 precision control): the fold falls back to the double fold there.
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/portable_trig.hpp>

#include <cstddef>
#include <cstring>
#include <limits>
#include <utility>

using namespace admiral::detail;

#ifdef ADM_POW2_CIRCLE
static bool circle_gate() {
    return std::numeric_limits<long double>::is_iec559 && ld_full_precision();
}

static std::pair<double, double> circle_derive(bool forward, std::size_t num,
                                               std::size_t den) {
    num %= den;
    if (forward && num != 0) num = den - num;
    const ct_sincos_v<double> w =
        portable_trig::pow2_circle[num * (portable_trig::kPow2CircleDen / den)];
    return {w.s, w.c};
}
#endif

TEST_CASE("the power-of-two circle matches the fold bit for bit",
          "[portable_trig][numerics]") {
#ifndef ADM_POW2_CIRCLE
    SUCCEED("long double is not the circle's fold format: the circle is not read");
#else
    if (!circle_gate()) {
        SUCCEED("the run time long double degrades (valgrind, x87 precision control)");
        return;
    }
    for (std::size_t den = 8; den <= portable_trig::kPow2CircleDen; den <<= 1) {
        for (std::size_t k = 0; k < den; ++k) {
            for (const bool fwd : {false, true}) {
                const auto t = circle_derive(fwd, k, den);
                const auto p = portable_trig::sincos_turns(k, den, fwd);
                const std::size_t n = k % den;
                const std::size_t m = fwd && n != 0 ? den - n : n;
                const ct_sincos_v<double> f = sincos_turns_eval<double>(false, m, den);
                REQUIRE(std::memcmp(&t.first, &f.s, sizeof(double)) == 0);
                REQUIRE(std::memcmp(&t.second, &f.c, sizeof(double)) == 0);
                REQUIRE(std::memcmp(&p.first, &f.s, sizeof(double)) == 0);
                REQUIRE(std::memcmp(&p.second, &f.c, sizeof(double)) == 0);
            }
        }
    }
#endif
}
