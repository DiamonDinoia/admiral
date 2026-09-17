#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>

#include "simd.hpp"

#include "cache.hpp"

namespace admiral {
namespace detail {

inline constexpr std::size_t SBO_MAX = 4096;

inline constexpr std::size_t SBO_PAD = 16;

static_assert(4 * (SBO_MAX + SBO_PAD) * sizeof(double) <= 192u * 1024u,
              "soa_scratch<double,4> stack frame exceeds the 192 KB budget");

template<typename T>
inline constexpr std::size_t span_align =
    std::max(xsimd::batch<T>::arch_type::alignment(), kCacheLine);

// Defined in src/scratch_alloc.cpp, which is the only translation unit that names the
// allocator. Keeping it out of line keeps snmalloc's headers, and its C++20 and cmpxchg16b
// requirements, out of every TU that includes admiral.
void* scratch_alloc(std::size_t bytes, std::size_t align);
void scratch_free(void* p, std::size_t align) noexcept;
// Allocations made through the seam since process start. Test observable only; the version
// script keeps it out of the shipped ABI.
long scratch_alloc_count() noexcept;

template<typename T>
struct aligned_delete {
    void operator()(T* p) const noexcept { scratch_free(p, span_align<T>); }
};
template<typename T>
using aligned_buffer = std::unique_ptr<T[], aligned_delete<T>>;

template<typename T>
[[nodiscard]] aligned_buffer<T> make_aligned_buffer(std::size_t n) {
    return aligned_buffer<T>(static_cast<T*>(scratch_alloc(n * sizeof(T), span_align<T>)));
}

template<typename T, std::size_t K>
struct soa_scratch {
    static_assert(K > 0, "K must be positive");

    explicit soa_scratch(std::size_t n) : soa_scratch(n, nullptr, 0) {}

    // Elements the heap arm owns for `n` (K arms of one padded stride): the slice a plan-owned
    // arena must provide to back this scratch externally.
    [[nodiscard]] static std::size_t heap_elems(std::size_t n) noexcept {
        return K * span_stride(n);
    }

    // As the plain ctor, but views `ext` instead of touching the heap when the request is
    // heap-sized and `ext` covers it. An undersized or absent `ext` keeps the old behavior.
    soa_scratch(std::size_t n, T* ext, std::size_t ext_elems) : m{} {
        m.stride = span_stride(n);
        if (n <= SBO_MAX) {
            m.ptr = m.stack_buf;
        } else if (ext != nullptr && ext_elems >= heap_elems(n)) {
            m.ptr = ext;
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
        const std::size_t s = (n % critical == 0) ? n + SBO_PAD : n;
        const std::size_t padded = s;
        return (padded + lane - 1) & ~(lane - 1);
    }

    struct M {
        M() {}
        std::size_t stride;
        T* ptr;
        alignas(span_align<T>) T stack_buf[K * (SBO_MAX + SBO_PAD)];
        aligned_buffer<T> heap;
    } m;
};

}
}
