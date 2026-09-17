#pragma once

#include <cstddef>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include "cxx_compat.hpp"

namespace admiral {
namespace detail {

template<typename T>
using ct_real_t = std::conditional_t<(std::numeric_limits<T>::digits > 53), long double, double>;

template<typename F>
struct ct_sincos_v {
    F s;
    F c;
};
using ct_sincos_t = ct_sincos_v<double>;

template<typename F = double>
[[nodiscard]] ADM_CONSTEVAL ct_sincos_v<F> ct_sincos_small(F x) {
    const F x2 = x * x;
    const F tiny = std::numeric_limits<F>::epsilon() / F(2);
    F cterm = 1, csum = 1;
    F sterm = x, ssum = x;
    // The term index counts in integers, so `2k-1` and `2k+1` are exact by construction and the
    // loop cannot drift; 15 iterations covers binary128, the widest `F` any target carries.
    for (unsigned i = 1;; ++i) {
        const F k = static_cast<F>(i);
        cterm *= -x2 / ((2 * k - 1) * (2 * k));
        csum += cterm;
        sterm *= -x2 / ((2 * k) * (2 * k + 1));
        ssum += sterm;
        const F ca = cterm < 0 ? -cterm : cterm;
        const F sa = sterm < 0 ? -sterm : sterm;
        if (ca <= tiny && sa <= tiny) break;
    }
    return {ssum, csum};
}

template<typename F = double>
[[nodiscard]] ADM_CONSTEVAL ct_sincos_v<F> ct_sincos_turns(bool conjugate, std::size_t num,
                                                           std::size_t den) {
    num %= den;
    if (conjugate && num != 0) num = den - num;
    const std::size_t oct = (8 * num) / den;
    const std::size_t rem = 8 * num - oct * den;
    const F res = static_cast<F>(rem) *
                  (detail::numbers::pi_v<F> / (F(4) * static_cast<F>(den)));
    const ct_sincos_v<F> r = ct_sincos_small<F>(res);
    constexpr F inv_sqrt2 = detail::numbers::sqrt2_v<F> / F(2);
    F bc = 0, bs = 0;
    switch (oct) {
        case 0: bc = 1;            bs = 0;            break;
        case 1: bc = inv_sqrt2;    bs = inv_sqrt2;    break;
        case 2: bc = 0;            bs = 1;            break;
        case 3: bc = -inv_sqrt2;   bs = inv_sqrt2;    break;
        case 4: bc = -1;           bs = 0;            break;
        case 5: bc = -inv_sqrt2;   bs = -inv_sqrt2;   break;
        case 6: bc = 0;            bs = -1;           break;
        default: bc = inv_sqrt2;   bs = -inv_sqrt2;   break;
    }
    return {bs * r.c + bc * r.s, bc * r.c - bs * r.s};
}

// Exact root-of-unity classification of the stage-twiddle constants. Every twiddle the
// butterfly and granule arms special-case is one of the eight octant constants (rem == 0:
// the folded (s, c) is an exact {0, +-1, +-sqrt2/2} pair at every precision) or a generic
// angle (rem != 0). ct_root_form recomputes ct_sincos_turns' (oct, rem) split on integers,
// so the form is a pure function of the reduced (num, den); fold_form re-derives the same
// form from the folded pair through the arms' own FP conditions, in the arms' own order, so
// the two spellings agree on every root the fold can produce. test/kernels/test_ct_root.cpp
// sweeps the lattice over every shipped denominator; the consumers key their arms on the
// descriptor and carry no per-instantiation assert (a consumer-side static_assert pin was
// measured to renumber gcc rodata, so the lattice test is the pin).
enum class root_form { one, neg_one, neg_i, pos_i, diag, anti_diag, generic };

[[nodiscard]] constexpr root_form ct_root_form(std::size_t num, std::size_t den, bool conj) {
    num %= den;
    if (conj && num != 0) num = den - num;
    const std::size_t oct = (8 * num) / den;
    const std::size_t rem = 8 * num - oct * den;
    if (rem != 0) return root_form::generic;
    root_form form = root_form::generic;
    switch (oct) {
        case 0: form = root_form::one; break;
        case 1: form = root_form::diag; break;
        case 2: form = root_form::pos_i; break;
        case 3: form = root_form::anti_diag; break;
        case 4: form = root_form::neg_one; break;
        case 5: form = root_form::diag; break;
        case 6: form = root_form::neg_i; break;
        default: form = root_form::anti_diag; break;
    }
#ifdef ADM_CT_ROOT_POISON
    // Positive-control seam for the WILL_FAIL twin (kernels/test_ct_root.cpp): swaps the two
    // diagonal classes, so the lattice's fold-agreement check must fail. Never defined in a
    // shipped build.
    if (form == root_form::diag) return root_form::anti_diag;
    if (form == root_form::anti_diag) return root_form::diag;
#endif
    return form;
}

template<std::size_t Num, std::size_t Den, bool Conj>
[[nodiscard]] ADM_CONSTEVAL root_form ct_root() {
    return ct_root_form(Num, Den, Conj);
}

template<typename F>
[[nodiscard]] constexpr root_form fold_form(ct_sincos_v<F> w) {
    if (w.s == 0 && w.c == 1) return root_form::one;
    if (w.s == 0 && w.c == -1) return root_form::neg_one;
    if (w.c == 0 && w.s == -1) return root_form::neg_i;
    if (w.c == 0 && w.s == 1) return root_form::pos_i;
    if (w.c == w.s) return root_form::diag;
    if (w.c == -w.s) return root_form::anti_diag;
    return root_form::generic;
}

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
    if (N % 11 == 0) return 11;
    return N;
}

[[nodiscard]] constexpr std::size_t codelet_radix(std::size_t N) {
    if (N == 60) return 5;
    if (N == 32) return 8;
    return smallest_radix(N);
}

template<typename T>
[[nodiscard]] ADM_CONSTEVAL std::size_t codelet_radix_for(std::size_t N) {
    if (N == 54 && std::is_same_v<T, float>) return 6;
    return codelet_radix(N);
}

[[nodiscard]] constexpr bool ct_is_prime(std::size_t n) {
    if (n < 2) return false;
    for (std::size_t d = 2; d * d <= n; ++d)
        if (n % d == 0) return false;
    return true;
}

[[nodiscard]] constexpr std::size_t ct_powmod(std::size_t base, std::size_t exp, std::size_t m) {
    std::size_t result = 1 % m;
    base %= m;
    for (; exp != 0; exp >>= 1) {
        if (exp & 1) result = result * base % m;
        base = base * base % m;
    }
    return result;
}

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
    return 0;
}

[[nodiscard]] constexpr bool is_rader_prime(std::size_t N) {
    return N > 13 && ct_is_prime(N);
}

[[nodiscard]] constexpr bool butterfly_wants_reload(std::size_t IP, std::size_t vregs) {
    return detail::has_single_bit(IP) && IP >= 4 && 2u * IP >= vregs;
}

inline constexpr double kRegUsableFraction = 0.75;

[[nodiscard]] constexpr std::size_t usable_vector_regs(std::size_t vregs) {
    return static_cast<std::size_t>(static_cast<double>(vregs) * kRegUsableFraction);
}

}
}
