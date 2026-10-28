#pragma once

// The one tier-map include chain of the dispatch build, shared by src/api.cpp and the tier
// test files. riscv first: before <xsimd/xsimd.hpp> the rv64gc floor must enable the
// emulated batch first, or xsimd loads with XSIMD_NO_SUPPORTED_ARCHITECTURE.
#if defined(__riscv)
#include "riscv.hpp"
#elif defined(__aarch64__)
#include "aarch64.hpp"
#elif defined(__loongarch__)
#include "loongarch.hpp"
#elif defined(__x86_64__) || defined(_M_X64)
#include "x86.hpp"
#endif
