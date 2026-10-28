#pragma once

// The x86-64 tier map, widest first: v4 (AVX-512BW), v3 (AVX2 and FMA, xsimd's fma3<avx2>),
// the floor v1 (SSE2). ADM_TIER_TABLE feeds the test files (tag, arch); the alias below
// feeds the tier walk in src/api.cpp and lists the same archs in the same order. The pin
// probe holds on every x86 runner (SDE models the CPU).

#include "admiral/detail/simd.hpp"

#define ADM_TIER_TABLE(X)                                                            \
    X("v4", xsimd::avx512bw)                                                      \
    X("v3", xsimd::fma3<xsimd::avx2>)                                             \
    X("v1", xsimd::sse2)

#define ADM_TIER_MAP x86

namespace admiral::detail {
using tier_archs = xsimd::arch_list<xsimd::avx512bw, xsimd::fma3<xsimd::avx2>, xsimd::sse2>;
}
