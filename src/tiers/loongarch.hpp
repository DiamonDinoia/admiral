#pragma once

// The LoongArch tier map: the LSX floor (la64v1.0, numpy's baseline) and one tier, LASX.
// xsimd has no LASX FMA kernel, so the tier adds width only. Dropping it is one tag in
// cmake/tiers/loongarch.cmake; the LSX floor then ships as one build (ADM_DISPATCH=OFF).

#include "admiral/detail/simd.hpp"

#define ADM_TIER_TABLE(X)                                                            \
    X("lasx", xsimd::lasx)                                                        \
    X("lsx", xsimd::lsx)

#define ADM_TIER_MAP loongarch

namespace admiral::detail {
using tier_archs = xsimd::arch_list<xsimd::lasx, xsimd::lsx>;
}
