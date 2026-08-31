#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>  // std::align_val_t, aligned operator new[]

#include "simd.hpp"

#include "cache.hpp"  // kCacheLine (the alignment floor below)

namespace admiral {
namespace detail {

// Small-buffer-optimized scratch: `K` contiguous planar buffers of length `N`.
// `N <= SBO_MAX`: aligned stack buffer. `N > SBO_MAX`: one heap block sliced
// into `K` spans. Neither path constructs `T`, so both buffers stay
// uninitialized.

// Stack-resident threshold. `K=4` f64 takes 128 KB stack at `N=4096`; `N=8192`
// would need 256 KB, hence the cap.
inline constexpr std::size_t SBO_MAX = 4096;

// Anti-alias padding (`T` elements) between planar spans. For power-of-2 `n`
// the span addresses sit on the 4 KB L1 set period and conflict-miss; a
// non-power-of-2 offset breaks that spacing.
inline constexpr std::size_t SBO_PAD = 16;

// Worst-case (`K=4` f64) stack frame within a 192 KB budget.
static_assert(4 * (SBO_MAX + SBO_PAD) * sizeof(double) <= 192u * 1024u,
              "soa_scratch<double,4> stack frame exceeds the 192 KB budget");

// Span alignment: what a SIMD load needs, floored at a cache line so every
// span base stays line-aligned. The value comes from the arch, not a literal
// 64: an RVV VLEN above 512 bits is wider than a line.
template<typename T>
inline constexpr std::size_t span_align =
    std::max(xsimd::batch<T>::arch_type::alignment(), kCacheLine);

// Aligned heap block, no value-init.
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

    // `stack_buf` left uninitialized: no per-execute memset of the 128 KB frame.
    explicit soa_scratch(std::size_t n) : m{} {
        m.stride = span_stride(n);
        if (n <= SBO_MAX) {
            m.ptr = m.stack_buf;
        } else {
            // `span_stride` rounds to a `span_align` multiple, so every
            // `k*stride` span is aligned and the bulk W-loads never split a
            // cache line.
            m.heap = make_aligned_buffer<T>(K * m.stride);
            m.ptr = m.heap.get();
        }
    }

    T* buf(std::size_t k) noexcept { return m.ptr + k * m.stride; }

    // Span stride. Callers that treat the spans as one allocation take the
    // stride from here. The DIF driver lays a generation out as 2N contiguous
    // across two planes. The difference of two `buf()` pointers is a valid
    // stride only within one object.
    [[nodiscard]] std::size_t stride() const noexcept { return m.stride; }

private:
    // Per-span stride: the anti-alias pad applies when `n` is L1-set critical:
    // the 2-span gap is a multiple of the 4 KB set period. Then round up to a
    // `span_align` multiple, so every span stays aligned on both paths.
    static std::size_t span_stride(std::size_t n) noexcept {
        constexpr std::size_t l1_set_period_bytes = 4096;
        constexpr std::size_t critical = l1_set_period_bytes / (2 * sizeof(T));
        constexpr std::size_t lane = span_align<T> / sizeof(T);
        const std::size_t s = (n % critical == 0) ? n + SBO_PAD : n;
        return (s + lane - 1) & ~(lane - 1);
    }

    struct M {
        // User-provided ctor (not = default) suppresses value-init on `m{}`:
        // no per-execute zero-init of the `stack_buf`.
        M() {}
        std::size_t stride;
        T* ptr;
        // `T[]`, not `char[]`: a `char` buffer would need a `reinterpret_cast`,
        // which riscv64 rejects under `-Werror=cast-align` (x86 has no such
        // rule and never warns).
        alignas(span_align<T>) T stack_buf[K * (SBO_MAX + SBO_PAD)];
        aligned_buffer<T> heap;
    } m;
};

} // namespace detail
} // namespace admiral

