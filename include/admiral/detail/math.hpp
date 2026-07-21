#pragma once

// ============================================================================
// Small integer / size-class utilities shared across the FFT detail headers.
// Pure, header-only, no SIMD or twiddle dependencies.
// ============================================================================

#include <bit>
#include <complex>
#include <cstddef>

// CMake-generated single source of truth for the compiled codelet catalog,
// shared with the per-N codelet TUs in src/CMakeLists.txt.
#include <admiral/detail/codelet_max.hpp>

namespace admiral {
namespace detail {

// True iff N >= 1 and every prime factor of N is <= 11 (handled by the
// iterative DIF / codelet paths without Bluestein). radix_sym_dft/dif_butterfly
// are generic over odd radix, so radix-11 passes need no new kernel code.
[[nodiscard]] constexpr bool is_codelet_supported(std::size_t N) {
    if (N == 0) return false;
    for (unsigned p : {2u, 3u, 5u, 7u, 11u}) {
        while (N % p == 0) N /= p;
    }
    return N == 1;
}

// True iff N is in the compiled straight-line codelet catalog.
[[nodiscard]] constexpr bool is_codelet_catalog(std::size_t N) {
    for (int n : CODELET_CATALOG_SIZES) {
        if (N == static_cast<std::size_t>(n)) return true;
    }
    return false;
}

// Bluestein plan-cost model: pads to bit_ceil(2N-1) and runs ~3 pow2 FFTs of
// that size, modeled as 1.53 * pad * log2(pad) (fit to measurement). Shared by
// the four-step and Rader route cost gates.
[[nodiscard]] constexpr double bluestein_model_cost(std::size_t N) {
    const std::size_t pad = std::bit_ceil(2 * N - 1);
    return 1.53 * double(pad) * double(std::countr_zero(pad));
}

// ============================================================================
// Codelet catalog dispatch.
//
// The catalog is an explicit generated list. For these sizes the compile-time
// kernel<N> has all twiddles baked into .rodata (a prime N degenerates to a
// straight-line direct DFT) — no runtime twiddle table.
//
// To stop the heavy per-N kernel<N> bodies (radix-31/61 ≈ P² straight-line, ×4
// {float,double}×{fwd,inv}) from being re-instantiated in every consumer TU,
// the codelet path is compiled ONCE into the fft_codelets static library, one
// small TU per N.  Consumers see only the narrow codelet_dispatch<T>
// declaration below + the extern-template, so they instantiate nothing heavy.
// (Definition: src/codelet_apply.h, src/codelet_instance.cpp.in, codelet_dispatch.cpp.)
// ============================================================================

// AoS DFT of a catalog size, dispatched at runtime to the compiled
// compile-time kernel<N> codelet.  in == out is in-place; in != out reads `in`
// (preserved) and writes `out`.  UN-normalized: the caller applies 1/N for the
// inverse. Defined in the fft_codelets static library; declared-only here.
template<typename T, bool Forward>
void codelet_dispatch(const std::complex<T>* in, std::complex<T>* out, std::size_t N);

extern template void codelet_dispatch<float,  true >(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<float,  false>(const std::complex<float>*,  std::complex<float>*,  std::size_t);
extern template void codelet_dispatch<double, true >(const std::complex<double>*, std::complex<double>*, std::size_t);
extern template void codelet_dispatch<double, false>(const std::complex<double>*, std::complex<double>*, std::size_t);

} // namespace detail
} // namespace admiral

