#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>

#include "simd.hpp"

#include "cache.hpp"

namespace admiral {
namespace detail {

// S1b alignment-stabilization arm, sweep only. When ADM_ALIGN_STAB is set, the 1-D/axis
// engines' execute-time row workspaces (real_fft.hpp's make_unique_for_overwrite sites)
// allocate through make_aligned_buffer, pinning their base alignment to span_align<T> so the
// soa->aos store around the dif last pass sees one cache-line class for a given size,
// independent of malloc history. Default OFF: the call sites keep their master spelling.
#ifndef ADM_ALIGN_STAB
#define ADM_ALIGN_STAB 0
#endif

inline constexpr std::size_t SBO_MAX = 4096;

inline constexpr std::size_t SBO_PAD = 16;

// fix4 switch, sweep only. ADM_FIX4_PAD adds this many bytes, on top of SBO_PAD, between
// consecutive planes of every soa_scratch<T, K> (K > 1): the split re/im workspace planes of the
// DIF chain and every other soa_scratch caller. Must stay a multiple of 64 to keep every plane
// aligned. Default 0 is byte-identical to the source before this switch existed.
#ifndef ADM_FIX4_PAD
#define ADM_FIX4_PAD 0
#endif
static_assert(ADM_FIX4_PAD % 64u == 0u, "ADM_FIX4_PAD must be a multiple of 64 bytes");

static_assert(4 * (SBO_MAX + SBO_PAD + ADM_FIX4_PAD / sizeof(double)) * sizeof(double) <=
                  192u * 1024u,
              "soa_scratch<double,4> stack frame exceeds the 192 KB budget");

template<typename T>
inline constexpr std::size_t span_align =
    std::max(xsimd::batch<T>::arch_type::alignment(), kCacheLine);

template<typename T>
struct aligned_delete {
    void operator()(T* p) const noexcept {
        ::operator delete[](p, std::align_val_t{span_align<T>});
    }
};
template<typename T>
using aligned_buffer = std::unique_ptr<T[], aligned_delete<T>>;

template<typename T>
[[nodiscard]] aligned_buffer<T> make_aligned_buffer(std::size_t n) {
    return aligned_buffer<T>(static_cast<T*>(
        ::operator new[](n * sizeof(T), std::align_val_t{span_align<T>})));
}

template<typename T, std::size_t K>
struct soa_scratch {
    static_assert(K > 0, "K must be positive");

    explicit soa_scratch(std::size_t n) : m{} {
        m.stride = span_stride(n);
        if (n <= SBO_MAX) {
            m.ptr = m.stack_buf;
        } else {
            m.heap = make_aligned_buffer<T>(K * m.stride);
            m.ptr = m.heap.get();
        }
    }

    T* buf(std::size_t k) noexcept { return m.ptr + k * m.stride; }

    [[nodiscard]] std::size_t stride() const noexcept { return m.stride; }

private:
    static std::size_t span_stride(std::size_t n) noexcept {
        constexpr std::size_t l1_set_period_bytes = 4096;
        constexpr std::size_t critical = l1_set_period_bytes / (2 * sizeof(T));
        constexpr std::size_t lane = span_align<T> / sizeof(T);
        constexpr std::size_t fix4_pad_elems = ADM_FIX4_PAD / sizeof(T);
        const std::size_t s = (n % critical == 0) ? n + SBO_PAD : n;
        const std::size_t padded = s + fix4_pad_elems;
        return (padded + lane - 1) & ~(lane - 1);
    }

    struct M {
        M() {}
        std::size_t stride;
        T* ptr;
        alignas(span_align<T>) T stack_buf[K * (SBO_MAX + SBO_PAD + ADM_FIX4_PAD / sizeof(T))];
        aligned_buffer<T> heap;
    } m;
};

}
}
