// Bit test of the precomputed power-of-two circle: for every power-of-two den up to 2^16
// and every k, the circle twiddle must equal the run-time series fold bit for bit. The
// derivation below repeats the index math of sincos_turns's circle branch on purpose: an
// off-by-stride or a swapped sign must fail here, and so must one flipped table bit (the
// mutation control). This same TU also builds under the engine's fast-math flags
// (test_twiddle_circle_fast), where the fold must fold to the same bits. On a long double
// format other than 64-bit mantissa the circle is not the fold's value and sincos_turns
// does not read it, so the sweep is skipped.
#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/portable_trig.hpp>

#include <cstddef>
#include <cstring>
#include <limits>

using namespace admiral::detail;

static bool circle_gate() {
    return std::numeric_limits<long double>::is_iec559 &&
           std::numeric_limits<long double>::digits == 64 &&
           std::numeric_limits<long double>::max_exponent > 1024 && ld_full_precision();
}

static std::pair<double, double> circle_derive(bool forward, std::size_t num,
                                               std::size_t den) {
    num %= den;
    if (forward && num != 0) num = den - num;
    const ct_sincos_v<double> w =
        portable_trig::pow2_circle[num * (portable_trig::kPow2CircleDen / den)];
    return {w.s, w.c};
}

TEST_CASE("the power-of-two circle matches the fold bit for bit",
          "[portable_trig][numerics]") {
    if (!circle_gate()) {
        SUCCEED("long double is not the 64-bit-mantissa format: the circle is not read");
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
}
