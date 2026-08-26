#pragma once

// ============================================================================
// Compile-time math shared by kernel<N> and the runtime DIF chain:
//   ct_sincos_*                     : ADM_CONSTEVAL sin/cos of a rational turn fraction.
//   smallest_radix / codelet_radix  : radix to peel off N.
// Split from codelet.hpp so DIF pass headers (butterfly, twiddles) avoid
// re-instantiating kernel<N> in every consumer TU (it lives in admiral_codelets).
// ============================================================================

#include <cstddef>
#include <limits>
#include <numeric>  // std::gcd
#include <type_traits>
#include <utility>  // std::pair
#include "cxx_compat.hpp"  // ADM_CONSTEVAL, detail::has_single_bit, detail::numbers

namespace admiral {
namespace detail {

// ----------------------------------------------------------------------------
// Compile-time sin/cos of angle 2*pi*num/den. Turn fractions allow exact integer
// range-reduction before Taylor evaluation on a small residual.
//
// The fold runs in F. Every SIMD caller takes the default: double twiddles are
// below the rounding of both float and double. Only the scalar long double
// backend asks for more, and it must, because a double constant caps a transform
// at 2^-53 and long double's own tolerance sits 11 bits under that.
// ----------------------------------------------------------------------------

// Fold precision for element type T: double up to double, long double past it.
template<typename T>
using ct_real_t = std::conditional_t<(std::numeric_limits<T>::digits > 53), long double, double>;

template<typename F>
struct ct_sincos_v {
    F s;
    F c;
};
using ct_sincos_t = ct_sincos_v<double>;

// sin/cos of x for |x| <= pi/4, via Taylor series (full F precision there).
// consteval: only ever used to fold compile-time twiddles (never runtime codegen).
template<typename F = double>
[[nodiscard]] ADM_CONSTEVAL ct_sincos_v<F> ct_sincos_small(F x) {
    const F x2 = x * x;
    // cos: sum (-1)^k x^(2k)/(2k)!   ;  sin: sum (-1)^k x^(2k+1)/(2k+1)!
    // k counts in F: every factorial step below is an exact small integer.
    // The tail falls under eps(F) after 9 terms in double and 12 in long double.
    constexpr F kTerms = std::numeric_limits<F>::digits > 53 ? F(12) : F(9);
    F cterm = 1, csum = 1;
    F sterm = x, ssum = x;
    for (F k = 1; k <= kTerms; ++k) {
        cterm *= -x2 / ((2 * k - 1) * (2 * k));
        csum += cterm;
        sterm *= -x2 / ((2 * k) * (2 * k + 1));
        ssum += sterm;
    }
    return {ssum, csum};
}

// sin/cos of 2*pi*num/den, range-reduced by octant; `conjugate` negates the
// angle. Forward transforms use the conjugate (negative) exponent, so callers
// pass their Forward flag straight through and keep num/den unsigned. Every
// caller already holds unsigned values.
// consteval: folds kernel<N>/butterfly twiddles at compile time only.
template<typename F = double>
[[nodiscard]] ADM_CONSTEVAL ct_sincos_v<F> ct_sincos_turns(bool conjugate, std::size_t num,
                                                           std::size_t den) {
    // reduce num into [0, den), then reflect for the conjugate
    num %= den;
    if (conjugate && num != 0) num = den - num;
    // Octant and residual entirely in integers: 8*num = oct*den + rem with
    // rem in [0, den), so res = 2*pi*num/den - oct*pi/4 = (pi/4)*rem/den, exactly.
    // Subtracting the octant base in F instead cancels bits off the residual.
    const std::size_t oct = (8 * num) / den;
    const std::size_t rem = 8 * num - oct * den;
    const F res = static_cast<F>(rem) *
                  (detail::numbers::pi_v<F> / (F(4) * static_cast<F>(den)));  // [0, pi/4)
    const ct_sincos_v<F> r = ct_sincos_small<F>(res);
    // Rotate by base (multiple of pi/4); inv_sqrt2 = sqrt2/2 (exact, no inv_sqrt2 literal).
    constexpr F inv_sqrt2 = detail::numbers::sqrt2_v<F> / F(2);
    // C++17 constexpr functions may not declare an uninitialized variable, hence the 0s.
    F bc = 0, bs = 0;
    switch (oct) {
        case 0: bc = 1;            bs = 0;            break;
        case 1: bc = inv_sqrt2;    bs = inv_sqrt2;    break;
        case 2: bc = 0;            bs = 1;            break;
        case 3: bc = -inv_sqrt2;   bs = inv_sqrt2;    break;
        case 4: bc = -1;           bs = 0;            break;
        case 5: bc = -inv_sqrt2;   bs = -inv_sqrt2;   break;
        case 6: bc = 0;            bs = -1;           break;
        default: bc = inv_sqrt2;   bs = -inv_sqrt2;   break;  // case 7
    }
    // {s, c} = (sin(base+res), cos(base+res)): rotate base by res
    return {bs * r.c + bc * r.s, bc * r.c - bs * r.s};
}

// ----------------------------------------------------------------------------
// Compile-time factorization: radix to peel off N.
// Order: 4 (best work/twiddle ratio), 2, 3, 5, 7, 11. If none divide N, returns N
// (direct O(N²)); runtime driver routes large primes to Bluestein so kernel<N>
// is only instantiated for smooth N.
// ----------------------------------------------------------------------------

// constexpr (not consteval) so the runtime mixed-radix driver and the
// compile-time kernel<N> share one definition.
// Coprime factor pair of n, larger factor first, or {0,0} if n is a prime power
// (no PFA). Larger-first because stage B carries the emit: the biggest loop body
// shrinks when N2 is the smaller factor. Generic: every coprime-split size routes
// here, no special cases.
//
// The trial divisor steps by 1, not 2, so an EVEN n reaches its (odd, pow2)
// split. Pure powers of two are unaffected: every a dividing 2^m shares a factor
// with 2^m/a, so they still return {0,0} and stay on pow2_dif_butterfly.
[[nodiscard]] constexpr std::pair<std::size_t, std::size_t> coprime_split(std::size_t n) noexcept {
    for (std::size_t a = 2; a * a <= n; ++a)
        if (n % a == 0 && std::gcd(a, n / a) == 1) return {n / a, a};
    return {0, 0};
}

[[nodiscard]] constexpr std::size_t smallest_radix(std::size_t N) {
    if (N % 4 == 0) return 4;
    if (N % 2 == 0) return 2;
    if (N % 3 == 0) return 3;
    if (N % 5 == 0) return 5;
    if (N % 7 == 0) return 7;
    if (N % 11 == 0) return 11;  // 11-smooth composites (e.g. 121=11^2) avoid Bluestein
    return N;  // prime relative to the radix set: direct DFT
}

[[nodiscard]] constexpr std::size_t codelet_radix(std::size_t N) {
    if (N == 60) return 5;
    // N=32: peel r=8 (M=4) over r=4 (M=8). r=8 spills (peak_live 26 > 16) but its
    // arithmetic wins on wide out-of-order uarchs. N=64: r=8 is register-starved;
    // keeps r=4.
    if (N == 32) return 8;
    return smallest_radix(N);
}

// Like codelet_radix but N=54: f32 peels r=6 (M=9 odd, 6 sub-DFTs across 6/8
// ymm lanes). f64 must NOT (batch=4 < 6, falls scalar); f64 keeps r=2 where
// M=27 fills the full 2-wide (Wc==R) batch.
template<typename T>
[[nodiscard]] ADM_CONSTEVAL std::size_t codelet_radix_for(std::size_t N) {
    if (N == 54 && std::is_same_v<T, float>) return 6;
    return codelet_radix(N);
}

// ----------------------------------------------------------------------------
// Number theory for Rader's algorithm (see codelet.hpp).
// Rader converts prime kernel<p> into a length-(p-1) cyclic convolution, which
// replaces a spilling flat O(p²) DFT.
// ----------------------------------------------------------------------------

[[nodiscard]] constexpr bool ct_is_prime(std::size_t n) {
    if (n < 2) return false;
    for (std::size_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// (base^exp) mod m, exponentiation by squaring. Every value stays 64-bit: the
// squaring cannot overflow for any m below 2^32, which covers every FFT length.
[[nodiscard]] constexpr std::size_t ct_powmod(std::size_t base, std::size_t exp, std::size_t m) {
    std::size_t result = 1 % m;
    base %= m;
    for (; exp != 0; exp >>= 1) {
        if (exp & 1) result = result * base % m;
        base = base * base % m;
    }
    return result;
}

// Smallest primitive root g modulo prime p: the multiplicative order of g is
// exactly p-1, i.e. g^((p-1)/q) != 1 (mod p) for every prime factor q of p-1.
[[nodiscard]] constexpr std::size_t ct_primitive_root(std::size_t p) {
    const std::size_t phi = p - 1;
    for (std::size_t g = 2; g < p; ++g) {
        bool ok = true;
        for (std::size_t q = 2; q <= phi; ++q) {
            if (phi % q != 0 || !ct_is_prime(q)) continue;
            if (ct_powmod(g, phi / q, p) == 1) { ok = false; break; }
        }
        if (ok) return g;
    }
    return 0;  // unreachable for prime p
}

// Primes up to 13 have direct butterfly codelets; p>13 primes route to Rader to
// avoid the O(p²) unroll. 13 is flat because radix_sym_dft<13> costs less than
// Rader-13's two 12-point transforms plus rotations plus permutation gathers, and it
// emits less code; 17 is not, because the cofactor batch changes shape at 2*17 and up.
// The bound is a kernel<N> choice only. It must not widen is_codelet_supported:
// the DIF tape has no thunk for a prime above 11 in a first or last pass, and
// claiming one sends every two-factor size through an unbuildable chain.
// codelet_cost_cyc_f64/f32 price these leaves, so re-sweep both tables whenever
// this bound moves.
[[nodiscard]] constexpr bool is_rader_prime(std::size_t N) {
    return N > 13 && ct_is_prime(N);
}

// A radix whose 2*IP live batches do not fit the register file runs as two
// physical sweeps that reload their inputs (see dif_butterfly_wants_reload).
// The register count is a parameter so this header stays poet-free; callers
// pass poet::vector_register_count().
[[nodiscard]] constexpr bool butterfly_wants_reload(std::size_t IP, std::size_t vregs) {
    return detail::has_single_bit(IP) && IP >= 4 && 2u * IP >= vregs;
}

// A radix-IP butterfly holds 2*IP live batches plus its twiddles and addresses, so
// the file it can use is a fraction of the file it has. A fraction rather
// than a fixed headroom keeps the bound positive on a narrow-register ISA. Callers
// pass poet::vector_register_count(); at 32 registers this yields 24.
inline constexpr double kRegUsableFraction = 0.75;

[[nodiscard]] constexpr std::size_t usable_vector_regs(std::size_t vregs) {
    return static_cast<std::size_t>(static_cast<double>(vregs) * kRegUsableFraction);
}

} // namespace detail
} // namespace admiral

