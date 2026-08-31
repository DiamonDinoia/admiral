#pragma once

// twiddle_row: one twiddle row, W entries at a time.
//
// Entry i of a row is r^i for r = W_den^{step}, i in [0, n). Two pieces fill
// a row W-wide:
//   SEED    one batch holding r^l in lane l, from W sincos calls per row.
//   ANCHOR  block b is seed * r^(b*W), with r^(b*W) itself from sincos, so
//           nothing accumulates across blocks and error stays at sincos's own
//           rounding.
// Per-entry sincos is the alternative. `sincos_turns` is divider-bound: two
// 64-bit integer divisions, one double division, a 4-way quadrant switch per
// entry. One W-wide complex multiply replaces W `sincos_turns` calls per row.
// The SEED is exact sincos, not a Kogge-Stone prefix-product scan. A scan
// chains log2(W)+1 complex multiplies, and the accumulated error outweighs
// the sincos calls saved.
// `test_ulp.cpp` probes the tables end-to-end; `test_iterative.cpp` checks the
// factored pass-0 row entry by entry.

#include <cstddef>
#include <tuple>
#include <utility>

#include "simd.hpp"

#include "macros.hpp"          // `ADM_ALWAYS_INLINE`, `ADM_RESTRICT`
#include "portable_trig.hpp"   // `sincos_turns` (exact anchors)

namespace admiral {
namespace detail {

// (are,aim) *= (bre,bim).
template<typename B>
ADM_ALWAYS_INLINE void cmul_ip(B& are, B& aim, const B& bre, const B& bim) {
    const B re = are * bre - aim * bim;
    aim = are * bim + aim * bre;
    are = re;
}

// `re[i]`, `im[i]` <- cos/sin of (`Forward` ? -1 : +1) * 2*pi * step*i / den, i in [0,n).
// Direct evaluation below one full batch, where the seed cannot amortise.
template<typename T, bool Forward>
void geom_twiddle_row(std::size_t step, std::size_t den, std::size_t n,
                      T* ADM_RESTRICT re, T* ADM_RESTRICT im) {
    using B = xsimd::batch<T>;
    constexpr std::size_t W = B::size;

    const auto exact = [&](std::size_t i) {
        const auto [sn, cs] = portable_trig::sincos_turns<Forward>(step * i, den);
        return std::pair<T, T>{static_cast<T>(cs), static_cast<T>(sn)};
    };

    if (n < W) {
        for (std::size_t i = 0; i < n; ++i) std::tie(re[i], im[i]) = exact(i);
        return;
    }

    // Seed: lane l holds r^l, from W exact sincos calls per row.
    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) std::tie(seed_re[l], seed_im[l]) = exact(l);
    const B sre = B::load_aligned(seed_re);
    const B sim = B::load_aligned(seed_im);

    const std::size_t blocks = n / W;
    for (std::size_t b = 0; b < blocks; ++b) {
        // Exact anchor per block: nothing accumulates, so there is no drift to correct.
        const auto [a_re, a_im] = exact(b * W);
        B are = sre, aim = sim;
        cmul_ip(are, aim, B(a_re), B(a_im));
        are.store_unaligned(re + b * W);
        aim.store_unaligned(im + b * W);
    }
    for (std::size_t i = blocks * W; i < n; ++i) std::tie(re[i], im[i]) = exact(i);
}

}  // namespace detail
}  // namespace admiral

#include "undef_macros.hpp"
