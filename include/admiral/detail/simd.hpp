#pragma once

#include "cxx_compat.hpp"

#include <xsimd/config/xsimd_config.hpp>

#ifdef XSIMD_NO_SUPPORTED_ARCHITECTURE
// No SIMD arch at these flags: run on xsimd's emulated 128-bit batch, the state xsimd's config
// sets when XSIMD_WITH_EMULATED is on. So <xsimd/xsimd.hpp> must not precede this header.
#undef XSIMD_NO_SUPPORTED_ARCHITECTURE
#define XSIMD_WITH_EMULATED 1
#define XSIMD_DEFAULT_ARCH xsimd::emulated<128>
#include <xsimd/xsimd.hpp>
// make_sized_batch searches supported_architectures only, and that list never holds emulated<N>.
namespace xsimd {
template<> struct make_sized_batch<float, 4> { using type = batch<float>; };
template<> struct make_sized_batch<double, 2> { using type = batch<double>; };
}  // namespace xsimd
#else
#include <xsimd/xsimd.hpp>
#endif

// Under -ffast-math, xsimd_constants.hpp wraps its constants in
// `#pragma GCC optimize("signed-zeros")`. Each later declaration then carries its own
// optimization node, and gcc 16 does not inline across two nodes that differ. The reset puts
// back the command-line options.
ADM_RESET_FP_OPTIONS
