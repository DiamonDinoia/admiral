// Global operator new/delete replacements live in their own binary. On ELF they cover the whole
// process through symbol interposition; on PE they cover this module only, which is what the
// allocator pairing below is about. Valgrind redirects the same symbols to its own allocator, so
// under memcheck the counter misses every allocation made inside libadmiral.so and the block
// families mismatch on free. The valgrind job skips this binary; every other configuration runs
// it.
#include <catch2/catch_test_macros.hpp>

#include <admiral/admiral.hpp>
#include <admiral/detail/scratch.hpp>

#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

// MSVC spells the noinline attribute differently, and an over-aligned block there must go back
// through `_aligned_free`.
#if defined(_MSC_VER)
#include <malloc.h>
#define COUNTED_NOINLINE __declspec(noinline)
#else
#define COUNTED_NOINLINE [[gnu::noinline]]
#endif

// Global replacements count every allocation, libadmiral included. `noinline` keeps free() out of
// the caller, where gcc's -Wmismatched-new-delete pairs it with the builtin operator new.
//
// The unaligned and over-aligned families allocate through DIFFERENT allocators and must free
// through the matching one. A replacement here is per-module on Windows, not process-wide: the
// CRT DLL allocates with plain `malloc` through its own operator new, and inlined code in this
// binary frees that block through the replacement below. Routing an unaligned block to
// `_aligned_free` therefore crashes the process, which is what the MSVC cells segfaulted on.
namespace {
std::atomic<long> g_alloc_count{0};

COUNTED_NOINLINE void* counted_alloc(std::size_t n) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
    void* p = std::malloc(n ? n : 1);
    if (p == nullptr) throw std::bad_alloc{};
    return p;
}
COUNTED_NOINLINE void counted_free(void* p) noexcept { std::free(p); }

COUNTED_NOINLINE void* counted_alloc_aligned(std::size_t n, std::size_t align) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
    void* p = ::_aligned_malloc(n ? n : 1, align);
#else
    void* p = nullptr;
    if (::posix_memalign(&p, align, n ? n : 1) != 0) p = nullptr;
#endif
    if (p == nullptr) throw std::bad_alloc{};
    return p;
}
COUNTED_NOINLINE void counted_free_aligned(void* p) noexcept {
#if defined(_MSC_VER)
    ::_aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n); }
void* operator new[](std::size_t n) { return counted_alloc(n); }
void* operator new(std::size_t n, std::align_val_t a) {
    return counted_alloc_aligned(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return counted_alloc_aligned(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }
void operator delete(void* p, std::align_val_t) noexcept { counted_free_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { counted_free_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { counted_free_aligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept {
    counted_free_aligned(p);
}

// Execute a plan<long double> at n <= SBO_MAX must not heap-allocate scratch;
// at n > SBO_MAX the soa_scratch heap path must fire.
//
// The two directions need two counters. The global replacements above see every allocation the
// process makes, which is what "small n allocates nothing at all" needs. They do NOT see the
// scratch itself: it goes through admiral::detail::scratch_alloc, which allocates from snmalloc
// and never calls operator new. So the large-n direction, the one that makes this a check rather
// than a tautology, reads the seam's own counter.
// plan<long double> routes exclusively through scalar_nd_c2c<long double>, which
// uses soa_scratch<std::complex<long double>, 1> in every execute path.
TEST_CASE("soa_scratch: small-n execute uses zero heap allocations; large-n uses nonzero",
          "[alloc]") {
    // Small: n=64 <= SBO_MAX=4096 — all scratch on stack.
    {
        INFO("small n=64");
        admiral::plan<long double> p({64});
        std::vector<std::complex<long double>> v(64, {1, 0});
        p.forward(v.data());  // warmup: flushes any lazy init
        const long before = g_alloc_count.load(std::memory_order_relaxed);
        const long before_scratch = admiral::detail::scratch_alloc_count();
        p.forward(v.data());
        const long after = g_alloc_count.load(std::memory_order_relaxed);
        REQUIRE(after == before);
        REQUIRE(admiral::detail::scratch_alloc_count() == before_scratch);
    }
    // Large: n=8192 > SBO_MAX=4096 — soa_scratch falls back to heap.
    {
        INFO("large n=8192");
        admiral::plan<long double> p({8192});
        std::vector<std::complex<long double>> v(8192, {1, 0});
        p.forward(v.data());  // warmup
        const long before = admiral::detail::scratch_alloc_count();
        p.forward(v.data());
        const long after = admiral::detail::scratch_alloc_count();
        REQUIRE(after > before);
    }
}

// The seam picks its allocator by request size, and `scratch_free` has to reach the same one for
// a block on either side of the line. A mispaired free is what this catches: glibc `free` aborts
// on a snmalloc block ("invalid pointer") and snmalloc aborts on a glibc one, so the case fails
// loudly in an ordinary build. The sanitizer arms cannot stand in for it -- asan compiles the
// snmalloc branch out through ADM_SCRATCH_SYSTEM_ALLOC and valgrind takes the same fallback at
// runtime, so under both of them every size here goes to ::operator new[] and the line is never
// crossed. This case has to run in a normal Release and Debug build to prove anything.
//
// Freeing the SMALLEST block first is load-bearing: snmalloc builds its pagemap on its first
// allocation, so a `scratch_free` that consults it before any snmalloc allocation has happened
// reads uninitialised state. That segfaulted during the gate's development.
//
// The sizes bracket the line rather than naming it, because the line lives in an anonymous
// namespace in the .cpp and a test that hardcoded one value would stop covering the far side
// the moment the constant moved. A block under it must round-trip, and so must one over it,
// wherever it sits.
TEST_CASE("scratch seam pairs alloc and free across the allocator line", "[alloc][scratch]") {
    constexpr std::size_t kAlign = admiral::detail::span_align<double>;
    for (std::size_t bytes : {std::size_t{1} << 20, std::size_t{8} << 20, std::size_t{32} << 20,
                              std::size_t{64} << 20, std::size_t{128} << 20}) {
        INFO("bytes=" << bytes);
        const long before = admiral::detail::scratch_alloc_count();
        void* p = admiral::detail::scratch_alloc(bytes, kAlign);
        REQUIRE(p != nullptr);
        REQUIRE(admiral::detail::scratch_alloc_count() == before + 1);
        REQUIRE(reinterpret_cast<std::uintptr_t>(p) % kAlign == 0);
        // Touch both ends: a gate that hands back a short block fails here and not in a kernel.
        auto* b = static_cast<unsigned char*>(p);
        b[0] = 0x5a;
        b[bytes - 1] = 0xa5;
        REQUIRE(b[0] == 0x5a);
        REQUIRE(b[bytes - 1] == 0xa5);
        admiral::detail::scratch_free(p, kAlign);
    }
}
