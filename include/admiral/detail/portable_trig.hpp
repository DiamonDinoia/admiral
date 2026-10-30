#pragma once

#include <cstddef>
#include <utility>

#include "ct_math.hpp"

namespace admiral {
namespace detail {
namespace portable_trig {

// Runtime twin of the compile-time fold: roots of unity at ct_dd precision rounded once to
// double, so a table built here and a constant folded into a codelet are the same value.
template<bool Forward>
constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den) noexcept {
    num %= den;
    if constexpr (Forward) {
        if (num != 0) num = den - num;
    }
    const ct_sincos_v<double> w = sincos_turns_eval<double>(false, num, den);
    return {w.s, w.c};
}

constexpr std::pair<double, double> sincos_turns(std::size_t num, std::size_t den,
                                                 bool forward) noexcept {
    return forward ? sincos_turns<true>(num, den) : sincos_turns<false>(num, den);
}

}
}
}
