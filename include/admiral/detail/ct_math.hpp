#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include "cxx_compat.hpp"

namespace admiral {
namespace detail {

// A T wider than long double is its own fold type, ct_dd below included.
template<typename T>
using ct_real_t = std::conditional_t<
    (std::numeric_limits<T>::digits > 64), T,
    std::conditional_t<(std::numeric_limits<T>::digits > 53), long double, double>>;

// The circle constants of a fold type. A fold type wider than long double specializes them.
template<typename F>
inline constexpr F ct_pi = detail::numbers::pi_v<F>;
template<typename F>
inline constexpr F ct_sqrt2 = detail::numbers::sqrt2_v<F>;

template<typename F>
struct ct_sincos_v {
    F s;
    F c;
};
using ct_sincos_t = ct_sincos_v<double>;

// A double-double, hi + lo with |lo| <= ulp(hi) / 2: the fold type of the engine's constants
// (ct_real_t<ct_dd> is ct_dd). Every operation is an error-free transformation (Dekker, Numer.
// Math. 18 (1971) 224), so one text folds the butterfly constants at compile time and builds
// the twiddle tables at run time, both to about 2^-100.
struct ct_dd {
    double hi = 0, lo = 0;
    ct_dd() = default;
    constexpr ct_dd(double h, double l) : hi(h), lo(l) {}
    // Exact: the integers the series and the folds use stay below 2^53.
    template<typename A,
             std::enable_if_t<std::is_integral_v<A> || std::is_same_v<A, double>, int> = 0>
    constexpr ct_dd(A a) : hi(static_cast<double>(a)) {}
};

// The engine TUs compile every inline body below under a fast-math flag (ADM_USE_FAST_MATH:
// /fp:fast on cl, -ffast-math on clang-cl), where the compensated sums and the error-free
// transforms collapse; on COFF the linker keeps the first COMDAT or linkonce_odr copy it
// meets, so a test or consumer TU that calls one of these bodies can run the fast-math copy.
// One pragma region keeps every instantiation on the precise model, so the linker's pick no
// longer changes the result. Constant evaluation folds precisely under any flag; only the
// runtime bodies need the region. fp_contract needs no treatment: the Dekker split it could
// break runs in constant evaluation only, and the runtime product spells std::fma explicitly.
// _MSC_VER covers clang-cl too: its engine TUs get -ffast-math and it emits linkonce_odr
// copies, the same hazard class as cl. gcc/clang on ELF cannot hit it: their engine objects
// emit no out-of-line fold/dd copy, and an archive member only loads for an undefined symbol.
#if defined(_MSC_VER)
#  pragma float_control(precise, on, push)
#endif

constexpr ct_dd dd_two_sum(double a, double b) {
    const double s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}

constexpr ct_dd dd_fast_two_sum(double a, double b) {
    const double s = a + b;
    return {s, b - (s - a)};
}

// Dekker's split at compile time, one fma at run time: -ffp-contract=fast fuses the split's
// multiply into ta - a and breaks it. The run time callers are the dd engine, which needs kFusedFma.
constexpr ct_dd dd_two_prod(double a, double b) {
    if (!ADM_IS_CONSTANT_EVALUATED()) {
        const double p = a * b;
        return {p, std::fma(a, b, -p)};
    }
    constexpr double split = 0x1p27 + 1;
    const double ta = split * a, ah = ta - (ta - a), al = a - ah;
    const double tb = split * b, bh = tb - (tb - b), bl = b - bh;
    const double p = a * b;
    return {p, ((ah * bh - p) + ah * bl + al * bh) + al * bl};
}

constexpr ct_dd operator-(ct_dd a) { return {-a.hi, -a.lo}; }

constexpr ct_dd operator+(ct_dd a, ct_dd b) {
    const ct_dd s = dd_two_sum(a.hi, b.hi), t = dd_two_sum(a.lo, b.lo);
    const ct_dd u = dd_fast_two_sum(s.hi, s.lo + t.hi);
    return dd_fast_two_sum(u.hi, u.lo + t.lo);
}

constexpr ct_dd operator-(ct_dd a, ct_dd b) { return a + -b; }

constexpr ct_dd operator*(ct_dd a, ct_dd b) {
    const ct_dd p = dd_two_prod(a.hi, b.hi);
    return dd_fast_two_sum(p.hi, p.lo + (a.hi * b.lo + a.lo * b.hi));
}

constexpr ct_dd operator/(ct_dd a, ct_dd b) {
    const double q1 = a.hi / b.hi;
    const ct_dd r1 = a - b * q1;
    const double q2 = r1.hi / b.hi;
    const ct_dd r2 = r1 - b * q2;
    return dd_fast_two_sum(q1, q2) + r2.hi / b.hi;
}

constexpr ct_dd& operator+=(ct_dd& a, ct_dd b) { return a = a + b; }
constexpr ct_dd& operator*=(ct_dd& a, ct_dd b) { return a = a * b; }
constexpr bool operator<(ct_dd a, ct_dd b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
constexpr bool operator<=(ct_dd a, ct_dd b) { return !(b < a); }

template<>
inline constexpr ct_dd ct_pi<ct_dd>{0x1.921fb54442d18p+1, 0x1.1a62633145c07p-53};
template<>
inline constexpr ct_dd ct_sqrt2<ct_dd>{0x1.6a09e667f3bcdp+0, -0x1.bdd3413b26456p-54};

}
}

template<>
struct std::numeric_limits<admiral::detail::ct_dd> {
    static constexpr bool is_specialized = true;
    static constexpr int digits = 104;
    static constexpr admiral::detail::ct_dd epsilon() { return {0x1p-104, 0}; }
};

namespace admiral {
namespace detail {

// The series and the octant fold below run at compile time through the consteval names; the
// double-double engine also runs them to build its twiddle tables.
template<typename F>
[[nodiscard]] constexpr ct_sincos_v<F> sincos_small_eval(F x) {
    const F x2 = x * x;
    // The break test doubles the term instead of halving epsilon: gcc's composite long double
    // reports epsilon at the smallest subnormal and epsilon / 2 then fails to constant-fold.
    const F eps = std::numeric_limits<F>::epsilon();
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
        if (ca + ca <= eps && sa + sa <= eps) break;
    }
    return {ssum, csum};
}

// v to 106 bits. q is the 106-bit ulp of hi's binade: lo rounds to q, so F(hi) + F(lo) is exact.
// An F narrower than double rounds hi + lo to F. When the double sum drops lo entirely, one
// double bit flips toward lo first, so a value exactly on a float tie cannot round to even.
// ponytail: a value within one double ulp of a float tie, with the double sum different from hi,
// can still double-round; about one constant in 2^29. Round to odd on the double sum if that
// ever shows up in a failing constant.
template<typename F>
[[nodiscard]] constexpr F dd_round(ct_dd v) {
    if constexpr (std::numeric_limits<F>::digits < std::numeric_limits<double>::digits) {
        double sum = v.hi + v.lo;
        if (sum == v.hi && v.lo != 0) {
            sum += (v.lo > 0 ? 1.0 : -1.0) * (v.hi < 0 ? -v.hi : v.hi) * 0x1p-52;
        }
        return static_cast<F>(sum);
    } else {
        const double a = v.hi < 0 ? -v.hi : v.hi;
        double q = 0x1p-105;
        while (a != 0 && a < q * 0x1p105) q /= 2;
        while (a >= q * 0x1p106) q *= 2;
        const double t = v.lo / q, big = t < 0 ? -0x1p52 : 0x1p52;
        return F(v.hi) + F(((t + big) - big) * q);
    }
}

template <typename G>
[[nodiscard]] constexpr ct_sincos_v<G> sincos_turns_fold(bool conjugate, std::size_t num,
                                                         std::size_t den) {
    num %= den;
    if (conjugate && num != 0) num = den - num;
    const std::size_t oct = (8 * num) / den;
    const std::size_t rem = 8 * num - oct * den;
    const G res = static_cast<G>(rem) * (ct_pi<G> / (G(4) * static_cast<G>(den)));
    const ct_sincos_v<G> r = sincos_small_eval<G>(res);
    constexpr G inv_sqrt2 = ct_sqrt2<G> / G(2);
    G bc = 0, bs = 0;
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

// Fallback fold for a narrow F where long double adds no precision (long double is double on
// MSVC and clang-cl, composite double-double on ppc64le). The Taylor tails accumulate in
// double with a compensated sum, the residual angle folds at ct_dd, and the octant fold sums
// in ct_dd before the value rounds to F. Measured on clang-21 -mlong-double-64: one fold
// stays below 400 constexpr steps and make_rader_bhat<71U> below 200k after folding each
// twiddle once, against the default budget of 1048576. The ct_dd fold is tens of times that
// and busts the budget (the base commit fails the clang check at codelet.hpp:757).
// ponytail: the compensation folds away under -ffast-math or /fp:fast and the error then
// reaches 2.5 double ulp; the float_control region above pins the precise model on cl and
// clang-cl, the COFF compilers where a fast-math copy can out-select a test's own COMDAT copy.
template<typename F>
[[nodiscard]] constexpr ct_sincos_v<F> sincos_turns_fold_dbl(bool conjugate, std::size_t num,
                                                             std::size_t den) {
    num %= den;
    if (conjugate && num != 0) num = den - num;
    const std::size_t oct = (8 * num) / den;
    const std::size_t rem = 8 * num - oct * den;
    // The residual angle rounds once at ct_dd: a double-folded ratio loses bits of the angle,
    // which land 1:1 in the sine.
    const ct_dd ratio =
        ct_dd{static_cast<double>(rem), 0.0} / ct_dd{4.0 * static_cast<double>(den), 0.0};
    const double res = dd_round<double>(ratio * ct_pi<ct_dd>);
    const double x2 = res * res;
    const double tiny = std::numeric_limits<double>::epsilon() / 2.0;
    double cterm = 1, csum = 0, ccomp = 0;
    double sterm = res, ssum = 0, scomp = 0;
    for (unsigned i = 1;; ++i) {
        const double k = static_cast<double>(i);
        cterm *= -x2 / ((2 * k - 1) * (2 * k));
        double y = cterm - ccomp, t = csum + y;
        ccomp = (t - csum) - y;
        csum = t;
        sterm *= -x2 / ((2 * k) * (2 * k + 1));
        y = sterm - scomp;
        t = ssum + y;
        scomp = (t - ssum) - y;
        ssum = t;
        const double ca = cterm < 0 ? -cterm : cterm;
        const double sa = sterm < 0 ? -sterm : sterm;
        if (ca <= tiny && sa <= tiny) break;
    }
    // The compensated sums carry their pending term in the negated compensation: s - comp is
    // the exact accumulated value.
    const ct_dd c = dd_two_sum(1.0, csum) - ct_dd{ccomp, 0.0};
    const ct_dd s = dd_two_sum(res, ssum) - ct_dd{scomp, 0.0};
    if (oct == 0) return {dd_round<F>(s), dd_round<F>(c)};
    if (oct == 2) return {dd_round<F>(c), dd_round<F>(-s)};
    if (oct == 4) return {dd_round<F>(-s), dd_round<F>(-c)};
    if (oct == 6) return {dd_round<F>(-c), dd_round<F>(s)};
    constexpr ct_dd inv_sqrt2 = ct_sqrt2<ct_dd> * ct_dd{0.5, 0.0};
    const ct_dd u = c + s, v = c - s;
    const ct_dd pu = inv_sqrt2 * u, pv = inv_sqrt2 * v;
    // oct 1: {u, v}, oct 3: {v, -u}, oct 5: {-u, -v}, oct 7: {-v, u}, all by sqrt2/2.
    switch (oct) {
        case 1: return {dd_round<F>(pu), dd_round<F>(pv)};
        case 3: return {dd_round<F>(pv), dd_round<F>(-pu)};
        case 5: return {dd_round<F>(-pu), dd_round<F>(-pv)};
        default: return {dd_round<F>(-pv), dd_round<F>(pu)};
    }
}

// The run-time long double can be narrower than the compile-time format: valgrind emulates
// x87 at 53 bits, and an x87 precision-control word can do the same on real hardware. One
// volatile probe separates the two: 1 + 2^-60 is representable at 64 bits and rounds to 1 at
// 53. The volatile sum keeps -ffast-math from reducing the test to tail != 0. Stateless: the
// probe runs per call.
inline bool ld_full_precision() noexcept {
    volatile long double one = 1.0L;
    volatile long double tail = 0x1p-60L;
    volatile long double sum = one + tail;
    return sum != one;
}

template<typename F>
[[nodiscard]] constexpr ct_sincos_v<F> sincos_turns_eval(bool conjugate, std::size_t num,
                                                         std::size_t den) {
    // gcc folds an F that is a pair of doubles (IBM double-double) only where the exact result
    // fits 106 bits (fold-const.cc, MODE_COMPOSITE_P), so that F folds at ct_dd. A double or
    // float constant folds at long double and rounds once to F where long double is a wide
    // IEEE type (x86 80-bit, arm64 binary128): a double-folded series plus the octant fold
    // lands 2-3 F-ulp off the true value, and that error sits in every twiddle table and
    // codelet constant built from these values. Everywhere else (long double is double or
    // composite) the compensated double fold takes over. long double and ct_dd constants run
    // their own series at their own fold precision.
    if constexpr (std::numeric_limits<F>::digits == 2 * std::numeric_limits<double>::digits) {
        const ct_sincos_v<ct_dd> w = sincos_turns_fold<ct_dd>(conjugate, num, den);
        return {dd_round<F>(w.s), dd_round<F>(w.c)};
    } else if constexpr (std::numeric_limits<F>::digits < 64 &&
                         std::numeric_limits<long double>::is_iec559 &&
                         std::numeric_limits<long double>::digits > 53 &&
                         std::numeric_limits<long double>::max_exponent > 1024) {
        // The max_exponent gate keeps the IBM composite long double (ppc64le) out: it reports
        // is_iec559 with double's range and folds no better than the dd route below. The probe
        // keeps a degraded run-time long double on that same fallback, so the fold holds its
        // one-2^-53 contract wherever it runs.
        if (!ADM_IS_CONSTANT_EVALUATED() && !ld_full_precision())
            return sincos_turns_fold_dbl<F>(conjugate, num, den);
        const ct_sincos_v<long double> w = sincos_turns_fold<long double>(conjugate, num, den);
        return {static_cast<F>(w.s), static_cast<F>(w.c)};
    } else if constexpr (std::numeric_limits<F>::digits < 64) {
        return sincos_turns_fold_dbl<F>(conjugate, num, den);
    } else {
        return sincos_turns_fold<F>(conjugate, num, den);
    }
}

#if defined(_MSC_VER)
#  pragma float_control(pop)
#endif

template<typename F = double>
[[nodiscard]] ADM_CONSTEVAL ct_sincos_v<F> ct_sincos_turns(bool conjugate, std::size_t num,
                                                           std::size_t den) {
    return sincos_turns_eval<F>(conjugate, num, den);
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

[[nodiscard]] constexpr std::size_t ct_mul_mod(std::size_t a, std::size_t b, std::size_t m) {
    if constexpr (sizeof(std::size_t) <= 4) {
        // a * b is too large for a 32-bit size_t when m > 65536: the residues reach
        // m - 1 and (m - 1)^2 >= 2^32 (p = 65537 puts m - 1 at exactly 2^32 mod 2^32 = 0).
        // Multiply in 64 bits so the residue is correct.
        return width_cast<std::size_t>((std::uint64_t{a} * b) % m);
    }
    // Keep the old form on 64-bit targets: same value, and a constexpr evaluator
    // counts the same steps as before (apple-clang 16.4 hits its step limit in
    // make_rader_bhat<97U> when the op sequence changes).
    return (a * b) % m;
}

[[nodiscard]] constexpr std::size_t ct_powmod(std::size_t base, std::size_t exp, std::size_t m) {
    std::size_t result = 1 % m;
    base %= m;
    for (; exp != 0; exp >>= 1) {
        if (exp & 1) result = ct_mul_mod(result, base, m);
        base = ct_mul_mod(base, base, m);
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
