#pragma once

#include <cstddef>
#include <tuple>
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

    alignas(B::arch_type::alignment()) T seed_re[W], seed_im[W];
    for (std::size_t l = 0; l < W; ++l) std::tie(seed_re[l], seed_im[l]) = exact(l);
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

}
}

#include "undef_macros.hpp"
