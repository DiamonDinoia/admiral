#pragma once

// The riscv64 tier map, widest first. xsimd's rvv<N> needs vlenb * 8 >= N at run time, so the
// walk picks the widest RVV the core has; a CPU below 128 bits runs the rv64gc floor
// (emulated<128>): the dispatcher runs it without a check, and the floor objects hold no V
// instruction. The tier tag is the C++ type: every rvv<N> shares the name "riscv+rvv".
// The qemu legs build the tests at the rv64gc floor, where the arch list of the floor build
// holds only the emulated arch, so the pin probe disables itself (probe_ok in
// test/tier_select.cpp); the runner, not the host, picks the CPU.

#include "admiral/detail/simd.hpp"

#define ADM_TIER_TABLE(X)                                                            \
    X("rvv512", xsimd::detail::rvv<512>)                                          \
    X("rvv256", xsimd::detail::rvv<256>)                                          \
    X("rvv128", xsimd::detail::rvv<128>)                                          \
    X("rv64gc", xsimd::emulated<128>)

#define ADM_TIER_MAP riscv

namespace admiral::detail {
using tier_archs = xsimd::arch_list<xsimd::detail::rvv<512>, xsimd::detail::rvv<256>,
                                    xsimd::detail::rvv<128>, xsimd::emulated<128>>;
}
