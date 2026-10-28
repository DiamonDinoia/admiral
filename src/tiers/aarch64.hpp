#pragma once

// The AArch64 tier map: the NEON floor (armv8-a) plus one fixed-width SVE tier, sve256.
// xsimd has one arch per SVE vector length and the length is fixed at compile time
// (-msve-vector-bits), so a fixed-width SVE build runs only at its own vector length:
// xsimd's run-time pick (hwcap + rdvl) takes sve<N> only at VL == N; any other VL falls
// back to NEON. The SVE tiers share one xsimd name(), so the tag is
// the type. sve512 stays out: gcc 14.2 ICEs ("unrecognizable insn", UNSPEC_TRN1_CONV on
// VNx4BI) in inst_gt_f.cpp at 512 bits, and 61 wrong results at VL 512 under qemu wait on
// job sve512-diag. Re-adding it is one tag in cmake/tiers/aarch64.cmake and one row here.

#include "admiral/detail/simd.hpp"

#define ADM_TIER_TABLE(X)                                                            \
    X("sve256", xsimd::detail::sve<256>)                                          \
    X("neon", xsimd::neon64)

#define ADM_TIER_MAP aarch64

namespace admiral::detail {
using tier_archs = xsimd::arch_list<xsimd::detail::sve<256>, xsimd::neon64>;
}
