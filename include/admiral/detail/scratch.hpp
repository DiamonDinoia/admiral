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
        const std::size_t s = (n % critical == 0) ? n + SBO_PAD : n;
        return (s + lane - 1) & ~(lane - 1);
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
