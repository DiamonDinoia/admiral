#pragma once

// ============================================================================
// Compile-time math shared by the codelet kernel<N> AND the runtime DIF chain:
//   - ct_sincos_*  : consteval sin/cos of a rational turn fraction (for the
//                    constant-folded radix-r DFT matrix and kernel<N> twiddles).
//   - smallest_radix / codelet_radix : the radix to peel off N.
//
// Split out of codelet.hpp so the DIF pass headers (butterfly, twiddles)
// can pull in just this lightweight, header-only math without dragging in the
// heavy kernel<N> template — which would otherwise be re-instantiated in every
// consumer TU. kernel<N> now lives only in the compiled fft_codelets library.
// ============================================================================

#include <numbers>
#include <type_traits>

namespace admiral {
namespace detail {

using std::numbers::pi;  // double; the compile-time twiddle math below is double

// ----------------------------------------------------------------------------
// Compile-time trigonometry (for the constant-folded radix-r DFT matrix).
// Computes sin/cos of an angle given as a fraction `num/den` of a full turn,
// i.e. angle = 2*pi*num/den. Working in turns lets us range-reduce exactly with
// integer arithmetic before evaluating a Taylor series on a small residual.
// ----------------------------------------------------------------------------

// Compile-time sin/cos result. s/c are ALWAYS double regardless of the FFT's
// value type T: twiddles are computed at full double precision and cast to T at
// the use site (more accurate than computing them in float). Do not template
// this on T — the fixed double precision is intentional.
struct ct_sincos_t {
    double s;
    double c;
};

// sin/cos of x for |x| <= pi/4, via Taylor series (full double precision there).
// consteval: only ever used to fold compile-time twiddles (never runtime codegen).
[[nodiscard]] consteval ct_sincos_t ct_sincos_small(double x) {
    const double x2 = x * x;
    // cos: sum (-1)^k x^(2k)/(2k)!   ;  sin: sum (-1)^k x^(2k+1)/(2k+1)!
    double cterm = 1.0, csum = 1.0;
    double sterm = x, ssum = x;
    for (int k = 1; k <= 9; ++k) {
        cterm *= -x2 / static_cast<double>((2 * k - 1) * (2 * k));
        csum += cterm;
        sterm *= -x2 / static_cast<double>((2 * k) * (2 * k + 1));
        ssum += sterm;
    }
    return {ssum, csum};
}

// sin/cos of 2*pi*num/den, range-reduced by octant. Exact integer reduction of
// the turn fraction keeps accuracy independent of how large num/den is.
// consteval: every call site folds a compile-time twiddle (kernel<N> / butterfly
// matrix); it is never emitted as runtime code.
[[nodiscard]] consteval ct_sincos_t ct_sincos_turns(long long num, long long den) {
    // reduce num into [0, den)
    num %= den;
    if (num < 0) num += den;
    // octant: which eighth of the turn. frac in [0,1).
    // angle = 2*pi*frac. Reduce to residual within +-pi/4 around an axis.
    const double frac = static_cast<double>(num) / static_cast<double>(den);
    // octant index 0..7
    const double scaled = frac * 8.0;
    long long oct = static_cast<long long>(scaled);  // floor for non-negative
    oct %= 8;
    // residual angle from the octant centre axis (multiples of pi/4)
    const double base = static_cast<double>(oct) * (pi / 4.0);
    const double res = (2.0 * pi * frac) - base;  // in [-?]; |res| <= pi/8 < pi/4
    const ct_sincos_t r = ct_sincos_small(res);
    // rotate (cos base, sin base) where base is a multiple of pi/4 -- exact-ish.
    // 1/sqrt(2) = sqrt(2)/2; dividing by 2 is exact, so this is bit-identical to
    // the literal it replaces (std::numbers has sqrt2 but no inv_sqrt2).
    constexpr double inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    double bc, bs;
    switch (oct) {
        case 0: bc = 1.0;          bs = 0.0;          break;
        case 1: bc = inv_sqrt2;    bs = inv_sqrt2;    break;
        case 2: bc = 0.0;          bs = 1.0;          break;
        case 3: bc = -inv_sqrt2;   bs = inv_sqrt2;    break;
        case 4: bc = -1.0;         bs = 0.0;          break;
        case 5: bc = -inv_sqrt2;   bs = -inv_sqrt2;   break;
        case 6: bc = 0.0;          bs = -1.0;         break;
        default: bc = inv_sqrt2;   bs = -inv_sqrt2;   break;  // case 7
    }
    // (cos(base+res), sin(base+res)) = rotate base by res
    return {bs * r.c + bc * r.s, bc * r.c - bs * r.s};
}

// ----------------------------------------------------------------------------
// Compile-time factorization: pick the radix to peel off N.
// Preference order mirrors pocketfft: 4 first (best work/twiddle ratio), then
// 2, 3, 5, 7. If none divide N, N is "prime to our radix set" and is handled as
// a single direct radix-N DFT (correct but O(N^2)); the runtime driver routes
// large such N to Bluestein instead so kernel<N> is only instantiated for N
// built from small factors.
// ----------------------------------------------------------------------------

// constexpr (not consteval) so the runtime mixed-radix driver and the
// compile-time kernel<N> share one definition.
[[nodiscard]] constexpr unsigned smallest_radix(unsigned N) {
    if (N % 4 == 0) return 4;
    if (N % 2 == 0) return 2;
    if (N % 3 == 0) return 3;
    if (N % 5 == 0) return 5;
    if (N % 7 == 0) return 7;
    if (N % 11 == 0) return 11;  // 11-smooth composites (e.g. 121=11^2) avoid Bluestein
    return N;  // prime-to-our-set: direct DFT
}

[[nodiscard]] constexpr unsigned codelet_radix(unsigned N) {
    if (N == 60) return 5;
    // N=32: peel radix-8 (M=4) instead of the smallest-radix 4 (M=8). The radix-8
    // split-radix leaf emits ~2x instructions and spills past the no-spill register
    // model (peak_live 26 > 16), but the better arithmetic structure wins on wide
    // OoO uarchs. N=64 (M=8) does NOT: the radix-8 leaf is register-starved there,
    // so 64 keeps the default r=4.
    if (N == 32) return 8;
    return smallest_radix(N);
}

// Precision-aware radix for the compile-time kernel<N, T> leaf. Identical to
// codelet_radix except for N=54: f32 peels r=6 (cofactor M=9 is odd, so the 6
// sub-DFTs batch across 6 of 8 ymm lanes — the cofactor-SIMD path). f64 must
// NOT: its native batch is 4 lanes (< 6), so r=6 would fall to scalar; f64 keeps
// r=2 whose M=27 cofactor fills the full 2-wide (Wc==R) batch.
template<typename T>
[[nodiscard]] consteval unsigned codelet_radix_for(unsigned N) {
    if (N == 54u && std::is_same_v<T, float>) return 6u;
    return codelet_radix(N);
}

// ----------------------------------------------------------------------------
// Number theory for Rader's algorithm (prime codelets, see codelet.hpp).
// A prime p has no small-radix factorization, so kernel<p> would otherwise be a
// giant flat radix-p DFT that spills. Rader turns it into a length-(p-1) cyclic
// convolution evaluated with the existing kernel<p-1> codelets.
// ----------------------------------------------------------------------------

[[nodiscard]] constexpr bool ct_is_prime(unsigned n) {
    if (n < 2) return false;
    for (unsigned d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

// (base^exp) mod m, exponentiation by squaring (consteval-safe, no overflow for
// the codelet primes: products stay < m^2 <= 64^2).
[[nodiscard]] constexpr unsigned ct_powmod(unsigned base, unsigned exp, unsigned m) {
    unsigned result = 1 % m;
    base %= m;
    while (exp > 0) {
        if (exp & 1u) result = static_cast<unsigned>(static_cast<unsigned long long>(result) * base % m);
        base = static_cast<unsigned>(static_cast<unsigned long long>(base) * base % m);
        exp >>= 1;
    }
    return result;
}

// Smallest primitive root g modulo prime p: the multiplicative order of g is
// exactly p-1, i.e. g^((p-1)/q) != 1 (mod p) for every prime factor q of p-1.
[[nodiscard]] constexpr unsigned ct_primitive_root(unsigned p) {
    const unsigned phi = p - 1;
    for (unsigned g = 2; g < p; ++g) {
        bool ok = true;
        for (unsigned q = 2; q <= phi; ++q) {
            if (phi % q != 0 || !ct_is_prime(q)) continue;
            if (ct_powmod(g, phi / q, p) == 1) { ok = false; break; }
        }
        if (ok) return g;
    }
    return 0;  // unreachable for prime p
}

// A size routes through the Rader prime codelet when it is a prime > 11 (primes
// 2,3,5,7,11 are handled directly by the symmetric radix butterfly; 13 and up
// would otherwise be a flat O(p^2) unroll).
[[nodiscard]] constexpr bool is_rader_prime(unsigned N) {
    return N > 11 && ct_is_prime(N);
}

} // namespace detail
} // namespace admiral

