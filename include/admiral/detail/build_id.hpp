#pragma once

// Identity of the current build, as the routing cost model keys builds.
//
// The key is (arch, compiler, major, W, regs, sizeof T), deliberately not an
// ISA revision. The same key works on NEON, SVE or RVV without a new
// enumeration. An unswept target finds no match and runs on the shared
// coefficients. The compiler sits in the key because the coefficients encode
// codegen quality, which does not port. The arch sits in the key because SVE
// at 512 bits has the same (W, regs) as AVX-512 and would otherwise inherit
// the x86 coefficients.

#include <cstddef>
#include <string_view>

#include <poet/poet.hpp>

#include "simd.hpp"

namespace admiral::detail {

inline constexpr std::string_view build_arch =
#if defined(__x86_64__) || defined(_M_X64)
    "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
    "aarch64";
#elif defined(__arm__)
    "arm";
#elif defined(__riscv)
    "riscv";
#elif defined(__powerpc64__)
    "ppc64";
#else
    "unknown";
#endif

inline constexpr std::string_view build_compiler =
#if defined(__INTEL_LLVM_COMPILER)
    "icx";
#elif defined(__clang__)
    "clang";
#elif defined(__GNUC__)
    "gcc";
#elif defined(_MSC_VER)
    "msvc";
#else
    "unknown";
#endif

inline constexpr int build_compiler_major =
#if defined(__INTEL_LLVM_COMPILER)
    __INTEL_LLVM_COMPILER / 10000;
#elif defined(__clang__)
    __clang_major__;
#elif defined(__GNUC__)
    __GNUC__;
#elif defined(_MSC_VER)
    _MSC_VER / 100;
#else
    0;
#endif

// Microarchitecture the codegen targets, part of a sweep receipt's filename.
// Two CPUs running identical binaries can disagree on the best route. Without
// the uarch key, both CPUs' measurements land in one file, pooled as if the
// CPUs agreed. "generic" = no uarch identity (`-march=x86-64-v*`).
inline constexpr std::string_view build_uarch =
#if defined(__znver5__)    // newer first: gcc/clang define exactly one macro
    "znver5";
#elif defined(__znver4__)
    "znver4";
#elif defined(__znver3__)
    "znver3";
#elif defined(__znver2__)
    "znver2";
#elif defined(__znver1__)
    "znver1";
#elif defined(__sapphirerapids__)
    "sapphirerapids";
#elif defined(__graniterapids__)
    "graniterapids";
#elif defined(__emeraldrapids__)
    "emeraldrapids";
#elif defined(__icelake_server__) || defined(__icelake_client__)
    "icelake";
#elif defined(__skylake__)
    "skylake";
#elif defined(__apple_m4__) || defined(__APPLE__) && defined(__aarch64__)
    "apple";            // apple-clang does not name the M-core via a macro
#elif defined(__aarch64__)
    "aarch64";          // `-mcpu`/`-march` macros vary by vendor; keep coarse
#else
    "generic";
#endif

inline constexpr std::size_t build_vector_regs = poet::vector_register_count();

template<typename T>
inline constexpr std::size_t build_width = xsimd::batch<T>::size;

}  // namespace admiral::detail
