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

namespace admiral {
namespace detail {

// float and double take xsimd's register barrier, and so does a batch of them. x87 long double
// has no "+x" or "+w" form, so it and any class keep "+m". An array of batches goes element by
// element. An array of scalars is a memory buffer: "+m" on the whole array emits no instruction,
// and a register form per element would add a load and a store for each one.
template <class T>
struct reg_barrier : std::bool_constant<std::is_same_v<T, float> || std::is_same_v<T, double>> {};
template <class T, class A>
struct reg_barrier<xsimd::batch<T, A>> : reg_barrier<T> {};

// Hides the value of v from the optimizer: v stays in its register and the compiler cannot
// reassociate across this point.
// dd_fft.hpp rounds hi = (fma(c, x, shift) - shift) to the grid. Under fast math the compiler
// folds that to c * x, and the error-free transform then returns a remainder of zero.
// The xsimd barrier is empty when the compiler may not reassociate. MSVC has no inline asm: v
// goes through a volatile pointer, and the compiler cannot know what the copy through that
// pointer reads.
template <class T>
ADM_COMPAT_INLINE void value_barrier(T& v) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (std::is_array_v<T> && std::is_class_v<std::remove_extent_t<T>>) {
        for (auto& e : v) value_barrier(e);
    } else if constexpr (reg_barrier<T>::value) {
        xsimd::kernel::detail::reassociation_barrier(v, "scaled loads and error-free transforms");
    } else {
        __asm__ __volatile__("" : "+m"(v));
    }
#else
    T* volatile p = &v;
    std::memmove(&v, p, sizeof(T));
#endif
}

}  // namespace detail
}  // namespace admiral
