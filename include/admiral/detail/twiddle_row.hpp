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

// One row of w^(step*(i0+i)/den) from the row view: the window offset rides the anchor
// leg only, like the wrapped rows.
template<bool Forward>
void geom_row_batched(portable_trig::turns_memo* memo, std::size_t step, std::size_t den,
                      std::size_t n, double* ADM_RESTRICT re, double* ADM_RESTRICT im,
                      std::size_t i0) {
    using B = xsimd::batch<double>;
    constexpr std::size_t W = B::size;
    const auto v = memo->row_view(step, den);
    if (n < W) {
        for (std::size_t i = 0; i < n; ++i)
            std::tie(re[i], im[i]) = geom_table_entry<Forward>(memo, v, i0 + i);
        return;
    }
    alignas(B::arch_type::alignment()) double seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l)
        std::tie(seed_re[l], seed_im[l]) = geom_table_entry<Forward>(memo, v, l);
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);
    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = geom_table_entry<Forward>(memo, v, i0 + b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(a_re), B(a_im));
        are.store_unaligned(re + b * W);
        aim.store_unaligned(im + b * W);
    }
    for (std::size_t i = blocks * W; i < n; ++i)
        std::tie(re[i], im[i]) = geom_table_entry<Forward>(memo, v, i0 + i);
}

// Same fill into interleaved complex storage.
template<bool Forward>
void geom_row_batched_packed(portable_trig::turns_memo* memo, std::size_t step,
                             std::size_t den, std::size_t n,
                             std::complex<double>* ADM_RESTRICT out) {
    using B = xsimd::batch<double>;
    constexpr std::size_t W = B::size;
    const auto v = memo->row_view(step, den);
    if (n < W) {
        for (std::size_t i = 0; i < n; ++i) {
            const auto [cre, cim] = geom_table_entry<Forward>(memo, v, i);
            out[i] = std::complex<double>(cre, cim);
        }
        return;
    }
    alignas(B::arch_type::alignment()) double seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l)
        std::tie(seed_re[l], seed_im[l]) = geom_table_entry<Forward>(memo, v, l);
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);
    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        const auto [a_re, a_im] = geom_table_entry<Forward>(memo, v, b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(a_re), B(a_im));
        alignas(B::arch_type::alignment()) double re[W], im[W];
        are.store_aligned(re);
        aim.store_aligned(im);
        double* dst = reinterpret_cast<double*>(out + b * W);
        for (std::size_t i = 0; i < W; ++i) {
            dst[2 * i] = re[i];
            dst[2 * i + 1] = im[i];
        }
    }
    for (std::size_t i = blocks * W; i < n; ++i) {
        const auto [cre, cim] = geom_table_entry<Forward>(memo, v, i);
        out[i] = std::complex<double>(cre, cim);
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
                      portable_trig::turns_memo* memo = nullptr, std::size_t i0 = 0) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;

    if constexpr (std::is_same_v<T, double>) {
        if (geom_batched_ok(den, memo)) {
            geom_row_batched<Forward>(memo, step, den, n, re, im, i0);
            return;
        }
    }

    const auto exact = [&](std::size_t i) {
        const auto [sn, cs] = portable_trig::sincos_turns<Forward>(step * (i0 + i), den, memo);
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
        const auto [sn, cs] = portable_trig::sincos_turns<Forward>(step * l, den, memo);
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

    if constexpr (std::is_same_v<T, double>) {
        if (geom_batched_ok(den, memo)) {
            geom_row_batched_packed<Forward>(memo, step, den, n, out);
            return;
        }
    }

    const auto exact = [&](std::size_t i) {
        const auto [sn, cs] = portable_trig::sincos_turns<Forward>(step * i, den, memo);
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
