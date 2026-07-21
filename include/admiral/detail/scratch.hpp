#pragma once

#include <cstddef>
#include <memory>

namespace admiral {
namespace detail {

// ============================================================================
// Small-buffer-optimized scratch helper.
//
// Owns K contiguous planar buffers of length N.  When N <= SBO_MAX the storage
// lives in an aligned stack byte buffer (no heap); otherwise a single
// std::vector is used.  One contiguous block is sliced into K equal spans.
//
// The stack buffer is raw bytes (no T construction until accessed) so there is
// no initialization cost for the heap path.
// ============================================================================

// Stack-resident scratch threshold. The stack buffer is K*(SBO_MAX+SBO_PAD)*
// sizeof(T) bytes: at K=4 that is ~128 KB for double and ~64 KB for float. This
// keeps the iterative-DIF scratch off the heap up to N=4096.
// Capped at 4096; 8192 would need 256 KB of stack frame (double).
inline constexpr std::size_t SBO_MAX = 4096;

// Anti-alias padding (in T elements) inserted between the K planar spans so
// their base addresses do not all collide in the same L1 cache sets. Without it,
// span k starts at k*n; for a power-of-2 n that is 4 KB-critical (e.g. n=4096
// f64 → spans 32 KB apart) the re/im and ping/pong buffers conflict-miss against
// each other. Offsetting each span by a non-power-of-2 lane count breaks the
// collision.
inline constexpr std::size_t SBO_PAD = 16;

// SBO_MAX trades stack frame for heap-alloc avoidance. The K=4 f64 stack buffer
// (the worst case) must stay within a sane per-frame budget. If you raise
// SBO_MAX, confirm the frame is still acceptable; 8192 would need 256 KB.
static_assert(4 * (SBO_MAX + SBO_PAD) * sizeof(double) <= 192u * 1024u,
              "soa_scratch<double,4> stack frame exceeds the 192 KB budget");

template<typename T, std::size_t K>
struct soa_scratch {
    static_assert(K > 0, "K must be positive");

    // stride/ptr/stack_buf are left uninitialized so the stack path pays no
    // per-alloc memset of the 128 KB stack_buf. Fields are assigned in the
    // body below.
    explicit soa_scratch(std::size_t n) : m{} {
        m.stride = span_stride(n);
        if (n <= SBO_MAX) {
            m.ptr = reinterpret_cast<T*>(m.stack_buf);
        } else {
            // make_unique_for_overwrite: no value-init of the heap buffer (the
            // FFT overwrites every element before reading), unlike std::vector's
            // resize which memsets.
            m.heap = std::make_unique_for_overwrite<T[]>(K * m.stride);
            m.ptr = m.heap.get();
        }
    }

    T* buf(std::size_t k) noexcept { return m.ptr + k * m.stride; }

private:
    // Per-span stride: n elements plus anti-alias padding when n is cache-set
    // critical. The K planar spans sit at byte offsets k*n*sizeof(T); two of them
    // conflict-miss when their gap is a multiple of the L1 4 KB set period. The
    // tightest trigger among K=4 spans is the 2-span gap, so the threshold is
    //   n*sizeof(T) * 2 == 4096  ->  n % (4096 / (2*sizeof(T))) == 0
    // i.e. 256 for f64 and 512 for f32. Non-critical sizes pay nothing.
    static std::size_t span_stride(std::size_t n) noexcept {
        constexpr std::size_t l1_set_period_bytes = 4096;
        constexpr std::size_t critical = l1_set_period_bytes / (2 * sizeof(T));
        return (n % critical == 0) ? n + SBO_PAD : n;
    }

    // All instance state in one internal `m`. stride/ptr/stack_buf are
    // deliberately left uninitialized — stack_buf is raw bytes, never memset.
    struct M {
        // User-provided (empty body, NOT `= default`): a defaulted default ctor is
        // not user-provided, so value-init (`m{}` in the soa_scratch ctor) would
        // zero-initialize all of M — a per-execute memset of the 128 KB stack_buf.
        // A user-provided ctor suppresses that zero-init.
        M() {}
        std::size_t stride;
        T* ptr;
        // Cache-line aligned: this container is SIMD-agnostic (no xsimd dep), and
        // 64 is a safe superset of every x86 SIMD load alignment.
        alignas(64) char stack_buf[K * (SBO_MAX + SBO_PAD) * sizeof(T)];
        std::unique_ptr<T[]> heap;
    } m;
};

} // namespace detail
} // namespace admiral

