#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <complex>
#include <tuple>
#include <type_traits>
#include <utility>

#include "simd.hpp"

#include "macros.hpp"
#include "portable_trig.hpp"

namespace admiral {
namespace detail {

template<typename B>
ADM_ALWAYS_INLINE void cmul_ip(B& are, B& aim, const B& bre, const B& bim) {
    const B re = are * bre - aim * bim;
    aim = are * bim + aim * bre;
    are = re;
}

// get_two's per-entry value off the row view: reduce, divisor scale and the build check
// ran once at row_view, and each seed/anchor lookup below is one wrapped-free table read,
// bit-identical to get_two (same branches in the fold). The lookups stay independent of
// each other: a serial index walk with a branch-free fold measured slower than this shape.
template<bool Forward>
ADM_ALWAYS_INLINE std::pair<double, double>
geom_table_entry(portable_trig::turns_memo* memo,
                 const portable_trig::turns_memo::two_row_view& v, std::size_t idx) {
    std::size_t f = memo->reduce(idx * v.step_d, v.den);
    if constexpr (Forward) {
        if (f != 0) f = v.den - f;
    }
    const std::size_t quad = static_cast<std::size_t>(f >= v.quarter) +
                             static_cast<std::size_t>(f >= 2 * v.quarter) +
                             static_cast<std::size_t>(f >= 3 * v.quarter);
    const std::size_t rem = f - quad * v.quarter;
    const std::size_t lo = rem & v.qmask;
    const std::size_t hi = rem >> v.shift;
    const double ls = v.lo_s[lo], lc = v.lo_c[lo];
    const double hs = v.hi_s[hi], hc = v.hi_c[hi];
    double s = ls * hc + lc * hs;
    double c = lc * hc - ls * hs;
    // The branch-free fold (cmov swap + sign XORs) measured slower than these branches.
    if (quad == 1 || quad == 3) {
        const double t = s;
        s = (quad == 1) ? c : -c;
        c = (quad == 1) ? -t : t;
    } else if (quad == 2) {
        s = -s;
        c = -c;
    }
    return {c, s};
}

// One fill's per-entry value: the circle class is decided once per fill (pow2_circle_ok)
// because the mode probe is not free and cannot change within a fill.
template<bool Forward>
ADM_ALWAYS_INLINE std::pair<double, double>
fill_entry(std::size_t idx, std::size_t den, [[maybe_unused]] bool circle,
           portable_trig::turns_memo* memo) {
#ifdef ADM_POW2_CIRCLE
    if (circle) return portable_trig::sincos_turns_circle<Forward>(idx, den);
#endif
    return portable_trig::sincos_turns<Forward>(idx, den, memo);
}

// Hot pow2-circle fill: the circle is a pure index->value map, so the per-entry work is one
// running-index read. The seed/anchor fold matches the general path below at the same
// product rounding, and the whole loop body is inline (the general path's per-entry lambda
// call cost dominated small builds: c2c 256 filled rows at 45% of the one-shot).
#ifdef ADM_POW2_CIRCLE
template<typename T, bool Forward>
void geom_row_circle(std::size_t step, std::size_t den, std::size_t n, T* ADM_RESTRICT re,
                     T* ADM_RESTRICT im, std::size_t i0) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;
    std::size_t idx = step * i0;
    if (n < W) {
        for (std::size_t i = 0; i < n; ++i, idx += step) {
            const auto [sn, cs] = portable_trig::sincos_turns_circle<Forward>(idx, den);
            re[i] = static_cast<T>(cs);
            im[i] = static_cast<T>(sn);
        }
        return;
    }
    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) {
        const auto [sn, cs] = portable_trig::sincos_turns_circle<Forward>(step * l, den);
        seed_re[l] = static_cast<T>(cs);
        seed_im[l] = static_cast<T>(sn);
    }
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);
    const std::size_t blocks = n / W;
    const std::size_t step_w = step * W;
    for (std::size_t b = 0; b < blocks; ++b, idx += step_w) {
        const auto [a_im, a_re] = portable_trig::sincos_turns_circle<Forward>(idx, den);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(static_cast<T>(a_re)), B(static_cast<T>(a_im)));
        are.store_unaligned(re + b * W);
        aim.store_unaligned(im + b * W);
    }
    for (std::size_t i = blocks * W; i < n; ++i, idx += step) {
        const auto [sn, cs] = portable_trig::sincos_turns_circle<Forward>(idx, den);
        re[i] = static_cast<T>(cs);
        im[i] = static_cast<T>(sn);
    }
}
#endif

// w^(C - f) is an axis swap or a sign flip of w^f for one exact C per row. A row whose
// index window closes on that complement folds one entry of each pair and derives the
// mate from the folded entry at the same ulp. Closure needs the step to divide the den,
// so the mate index is k0 - idx over one table period and the fill carries no divides.
struct row_pair_map {
    std::size_t k0 = 0, per = 0;
    int kind = 0;
};

[[nodiscard]] ADM_ALWAYS_INLINE row_pair_map
geom_row_pairs(std::size_t step_d, std::size_t den, std::size_t n, std::size_t i0) {
    row_pair_map pm;
    if (step_d == 0 || den % step_d != 0) return pm;
    const std::size_t per = den / step_d;
    if (i0 + n > per) return pm;
    std::size_t C = 0;
    if (den % 4 == 0 && (den / 4) % step_d == 0) { C = den / 4; pm.kind = 1; }
    else if (den % 2 == 0 && (den / 2) % step_d == 0) { C = den / 2; pm.kind = 2; }
    else { C = den; pm.kind = 3; }
    pm.per = per;
    pm.k0 = (C / step_d) % per;
    const std::size_t lo = 2 * i0, hi = 2 * (i0 + n - 1);
    const std::size_t last = i0 + n - 1;
    for (const std::size_t s : {pm.k0, pm.k0 + per}) {
        if (s < lo || s > hi) continue;
        const std::size_t a = (s > last) ? (s - last) : i0;
        const std::size_t b = (s < last + 1) ? (s - i0) : last;
        if (b > a) return pm;
    }
    pm.kind = 0;
    return pm;
}

template<bool Forward>
[[nodiscard]] ADM_ALWAYS_INLINE std::pair<double, double>
geom_pair_mate(int kind, double c, double s) {
    if (kind == 1) {
        if constexpr (Forward) return {-s, -c};
        return {s, c};
    }
    if (kind == 2) return {-c, s};
    return {c, -s};
}

// One row of table values. An entry at index 0 is exactly w^0 = (1, 0), so it needs no
// circle state and a row of only such entries never builds a view.
template<typename T, bool Forward, class Store>
ADM_ALWAYS_INLINE void geom_row_small(portable_trig::turns_memo* memo, std::size_t step, std::size_t den,
                                      std::size_t n, std::size_t i0, Store&& store) {
    using two_row_view = portable_trig::turns_memo::two_row_view;
    two_row_view v{};
    row_pair_map pm;
    bool have = false;
    constexpr std::size_t W = xsimd::batch<T>::size;
    const bool want_pairs = (n >= W + W / 2) && (n + n <= W + W);
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t idx = i0 + i;
        if (idx == 0) {
            store(i, 1.0, 0.0);
            continue;
        }
        if (!have) {
            v = memo->row_view(step, den);
            have = true;
            if (want_pairs) pm = geom_row_pairs(v.step_d, v.den, n, i0);
        }
        const auto [c, s] = geom_table_entry<Forward>(memo, v, idx);
        store(i, c, s);
        if (pm.kind == 0) continue;
        const std::size_t pidx = (idx <= pm.k0) ? (pm.k0 - idx) : (pm.k0 + pm.per - idx);
        if (pidx == idx || pidx < idx || pidx >= i0 + n) continue;
        const std::size_t pl = pidx - i0;
        if (c == 0.0 || s == 0.0) {
            const auto [c2, s2] = geom_table_entry<Forward>(memo, v, pidx);
            store(pl, c2, s2);
        } else {
            const auto [c2, s2] = geom_pair_mate<Forward>(pm.kind, c, s);
            store(pl, c2, s2);
        }
    }
}

// One row of w^(step*(i0+i)/den) from the row view: the window offset rides the anchor
// leg only, like the wrapped rows. The fold runs in double and casts at store: the cast
// before the vector product keeps float rows at the exact bits the scalar fill carried.
template<typename T, bool Forward>
void geom_row_batched(portable_trig::turns_memo* memo, std::size_t step, std::size_t den,
                      std::size_t n, T* ADM_RESTRICT re, T* ADM_RESTRICT im,
                      std::size_t i0) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;
    if (n < W) {
        geom_row_small<T, Forward>(memo, step, den, n, i0, [&](std::size_t i, double c, double s) {
            re[i] = static_cast<T>(c);
            im[i] = static_cast<T>(s);
        });
        return;
    }
    const auto v = memo->row_view(step, den);
    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) {
        const auto [c, s] = geom_table_entry<Forward>(memo, v, l);
        seed_re[l] = static_cast<T>(c);
        seed_im[l] = static_cast<T>(s);
    }
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);
    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = geom_table_entry<Forward>(memo, v, i0 + b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(static_cast<T>(a_re)), B(static_cast<T>(a_im)));
        are.store_unaligned(re + b * W);
        aim.store_unaligned(im + b * W);
    }
    for (std::size_t i = blocks * W; i < n; ++i) {
        const auto [c, s] = geom_table_entry<Forward>(memo, v, i0 + i);
        re[i] = static_cast<T>(c);
        im[i] = static_cast<T>(s);
    }
}

// Same fill into interleaved complex storage.
template<typename T, bool Forward>
void geom_row_batched_packed(portable_trig::turns_memo* memo, std::size_t step,
                             std::size_t den, std::size_t n,
                             std::complex<T>* ADM_RESTRICT out) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;
    if (n < W) {
        geom_row_small<T, Forward>(memo, step, den, n, 0, [&](std::size_t i, double c, double s) {
            out[i] = std::complex<T>(static_cast<T>(c), static_cast<T>(s));
        });
        return;
    }
    const auto v = memo->row_view(step, den);
    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) {
        const auto [c, s] = geom_table_entry<Forward>(memo, v, l);
        seed_re[l] = static_cast<T>(c);
        seed_im[l] = static_cast<T>(s);
    }
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);
    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = geom_table_entry<Forward>(memo, v, b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(static_cast<T>(a_re)), B(static_cast<T>(a_im)));
        alignas(B::arch_type::alignment()) T re[W], im[W];
        are.store_aligned(re);
        aim.store_aligned(im);
        T* dst = reinterpret_cast<T*>(out + b * W);
        for (std::size_t i = 0; i < W; ++i) {
            dst[2 * i] = re[i];
            dst[2 * i + 1] = im[i];
        }
    }
    for (std::size_t i = blocks * W; i < n; ++i) {
        const auto [cre, cim] = geom_table_entry<Forward>(memo, v, i);
        out[i] = std::complex<T>(static_cast<T>(cre), static_cast<T>(cim));
    }
}

// One call per pass block; each row keeps the full wrapper routing (memo fold, baked
// circle, or fallback) exactly as a per-row build would.
template<typename T, bool Forward>
void geom_twiddle_row(std::size_t step, std::size_t den, std::size_t n,
                      T* ADM_RESTRICT re, T* ADM_RESTRICT im,
                      portable_trig::turns_memo* memo = nullptr, std::size_t i0 = 0);

template<typename T, bool Forward>
void geom_rows_batched(portable_trig::turns_memo* memo, std::size_t l1, std::size_t den,
                       std::size_t rows, std::size_t n,
                       T* ADM_RESTRICT re, T* ADM_RESTRICT im, std::size_t stride) {
    for (std::size_t j = 1; j <= rows; ++j) {
        geom_twiddle_row<T, Forward>(j * l1, den, n,
                                     re + (j - 1) * stride, im + (j - 1) * stride, memo);
    }
}


// Power-of-two dens (up to kPow2CircleDen) read the baked circle per entry instead of a
// per-plan memo build: keep the wrapped rows there. Guard mirrors sincos_turns' branch
// window; smells like duplication, but the wrapper gates on ld_full_precision() too and
// this path only orders the two.
[[nodiscard]] ADM_ALWAYS_INLINE bool geom_batched_ok(
    std::size_t den, const portable_trig::turns_memo* memo) {
    if (memo == nullptr || den < 8 || den > portable_trig::turns_memo::kTwoLevelDenMax)
        return false;
#ifdef ADM_POW2_CIRCLE
    if (den <= portable_trig::kPow2CircleDen && (den & (den - 1)) == 0) return false;
#endif
    return true;
}

template<typename T, bool Forward>
void geom_twiddle_row(std::size_t step, std::size_t den, std::size_t n,
                      T* ADM_RESTRICT re, T* ADM_RESTRICT im,
                      portable_trig::turns_memo* memo, std::size_t i0) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;

    if (geom_batched_ok(den, memo)) {
        geom_row_batched<T, Forward>(memo, step, den, n, re, im, i0);
        return;
    }

#ifdef ADM_POW2_CIRCLE
    if (portable_trig::pow2_circle_ok(den)) {
        geom_row_circle<T, Forward>(step, den, n, re, im, i0);
        return;
    }
#endif

    const bool circle = false;
    (void)circle;
    const auto exact = [&](std::size_t i) {
        const auto [sn, cs] = fill_entry<Forward>(step * (i0 + i), den, circle, memo);
        return std::pair<T, T>{static_cast<T>(cs), static_cast<T>(sn)};
    };

    if (n < W) {
        for (std::size_t i = 0; i < n; ++i) std::tie(re[i], im[i]) = exact(i);
        return;
    }

    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    // The window offset rides the anchor leg only: putting it on both legs would shift every
    // entry by step*i0 twice.
    for (std::size_t l = 0; l < W; ++l) {
        const auto [sn, cs] = fill_entry<Forward>(step * l, den, circle, memo);
        seed_re[l] = static_cast<T>(cs);
        seed_im[l] = static_cast<T>(sn);
    }
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);

    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = exact(b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(a_re), B(a_im));
        are.store_unaligned(re + b * W);
        aim.store_unaligned(im + b * W);
    }
    for (std::size_t i = blocks * W; i < n; ++i) std::tie(re[i], im[i]) = exact(i);
}

// Same geometry into interleaved storage, for tables held as complex<T>: one sincos call per
// batch instead of per entry, at the same product rounding the split row above carries.
template<typename T, bool Forward>
void geom_twiddle_row_packed(std::size_t step, std::size_t den, std::size_t n,
                             std::complex<T>* ADM_RESTRICT out,
                             portable_trig::turns_memo* memo = nullptr) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;

    if (geom_batched_ok(den, memo)) {
        geom_row_batched_packed<T, Forward>(memo, step, den, n, out);
        return;
    }

    const bool circle = portable_trig::pow2_circle_ok(den);
    const auto exact = [&](std::size_t i) {
        const auto [sn, cs] = fill_entry<Forward>(step * i, den, circle, memo);
        return std::pair<T, T>{static_cast<T>(cs), static_cast<T>(sn)};
    };

    if (n < W) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto [re, im] = exact(i);
            out[i] = std::complex<T>(re, im);
        }
        return;
    }

    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) std::tie(seed_re[l], seed_im[l]) = exact(l);
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);

    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = exact(b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(a_re), B(a_im));
        alignas(B::arch_type::alignment()) T re[W], im[W];
        are.store_aligned(re);
        aim.store_aligned(im);
        T* dst = reinterpret_cast<T*>(out + b * W);
        for (std::size_t i = 0; i < W; ++i) {
            dst[2 * i] = re[i];
            dst[2 * i + 1] = im[i];
        }
    }
    for (std::size_t i = blocks * W; i < n; ++i) {
        const auto [re, im] = exact(i);
        out[i] = std::complex<T>(re, im);
    }
}

}
}

#include "undef_macros.hpp"
