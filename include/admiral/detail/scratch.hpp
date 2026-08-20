#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <new>  // std::align_val_t, aligned operator new[]

#include "simd.hpp"

#include "cache.hpp"  // kCacheLine (the alignment floor below)

namespace admiral {
namespace detail {

// ============================================================================
// Small-buffer-optimized scratch: K contiguous planar buffers of length N.
// N <= SBO_MAX: aligned stack buffer (no heap). N > SBO_MAX: single heap block
// sliced into K spans. Neither path constructs T: both are left uninitialized.
// ============================================================================

// Stack-resident threshold. K=4 f64: ~128 KB stack; keeps iterative-DIF scratch
// off the heap up to N=4096. Capped at 4096 (8192 needs 256 KB stack).
inline constexpr std::size_t SBO_MAX = 4096;

// Anti-alias padding (T elements) between planar spans. For power-of-2 n, spans
// at k*n are 4 KB-critical (e.g. n=4096 f64 → 32 KB apart), causing L1
// conflict misses between re/im and ping/pong buffers. Non-power-of-2 offset breaks it.
inline constexpr std::size_t SBO_PAD = 16;

// Sanity-check: K=4 f64 (worst case) stack frame within 192 KB budget.
static_assert(4 * (SBO_MAX + SBO_PAD) * sizeof(double) <= 192u * 1024u,
              "soa_scratch<double,4> stack frame exceeds the 192 KB budget");

// Span alignment: what a SIMD load needs, floored at a cache line so a padded row
// stride also keeps every span base line-aligned. Asked of the arch rather than
// written as 64: that literal is only right while the register is <= a line, and
// an RVV VLEN above 512 bits is not (every other buffer here already asks xsimd).
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

    // stack_buf left uninitialized: no per-execute memset of the 128 KB frame.
    explicit soa_scratch(std::size_t n) : m{} {
        m.stride = span_stride(n);
        if (n <= SBO_MAX) {
            m.ptr = m.stack_buf;
        } else {
            // No value-init: the FFT overwrites before reading. span_stride rounds to
            // a span_align multiple so every k*stride span is aligned too, and the
            // bulk W-loads never split a cache line.
            m.heap = make_aligned_buffer<T>(K * m.stride);
            m.ptr = m.heap.get();
        }
    }

    T* buf(std::size_t k) noexcept { return m.ptr + k * m.stride; }

    // Span stride. Callers that need to know the spans are one allocation, e.g. the DIF
    // driver's W-blocked SoA, which lays a generation out as 2N contiguous across what are
    // otherwise two planes, take it from here rather than differencing two buf() pointers,
    // which is only a valid stride when they came from the same object.
    [[nodiscard]] std::size_t stride() const noexcept { return m.stride; }

private:
    // Per-span stride with anti-alias padding when n is L1-set critical, then rounded
    // up to a span_align multiple so every k*stride span stays aligned on both paths.
    // Spans conflict-miss when their gap is a multiple of the 4 KB L1 set period.
    // Tightest K=4 trigger: 2-span gap → n*sizeof(T)*2 == 4096
    //   → n % (2048/sizeof(T)) == 0. Non-critical: no pad.
    static std::size_t span_stride(std::size_t n) noexcept {
        constexpr std::size_t l1_set_period_bytes = 4096;
        constexpr std::size_t critical = l1_set_period_bytes / (2 * sizeof(T));
        constexpr std::size_t lane = span_align<T> / sizeof(T);
        const std::size_t s = (n % critical == 0) ? n + SBO_PAD : n;
        return (s + lane - 1) & ~(lane - 1);
    }

    struct M {
        // User-provided ctor (not `= default`) suppresses value-init on `m{}`,
        // preventing a per-execute zero-init of the 128 KB stack_buf.
        M() {}
        std::size_t stride;
        T* ptr;
        // T[], not char[]: handing out a char buffer needs a reinterpret_cast, which
        // riscv64 rejects under -Werror=cast-align (x86 has no such rule and never warns).
        alignas(span_align<T>) T stack_buf[K * (SBO_MAX + SBO_PAD)];
        aligned_buffer<T> heap;
    } m;
};

} // namespace detail
} // namespace admiral

