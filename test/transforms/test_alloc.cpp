// Global operator new/delete replacements live in their own binary. They are process-wide, and
// valgrind redirects the same symbols to its own allocator, so under memcheck the counter misses
// every allocation made inside libadmiral.so and the block families mismatch on free. The valgrind
// job skips this binary; every other configuration runs it.
#include <catch2/catch_test_macros.hpp>

#include <admiral/admiral.hpp>

#include <atomic>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

// MSVC spells both the attribute and the aligned allocator differently, and its blocks must go
// back through `_aligned_free`.
#if defined(_MSC_VER)
#include <malloc.h>
#define COUNTED_NOINLINE __declspec(noinline)
#else
#define COUNTED_NOINLINE [[gnu::noinline]]
#endif

// Global replacements count every allocation, libadmiral.so included. `noinline` keeps free()
// out of the caller, where gcc's -Wmismatched-new-delete pairs it with the builtin operator new.
namespace {
std::atomic<long> g_alloc_count{0};

COUNTED_NOINLINE void* counted_alloc(std::size_t n, std::size_t align) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
#if defined(_MSC_VER)
    void* p = ::_aligned_malloc(n ? n : 1, align);
    if (p == nullptr) throw std::bad_alloc{};
#else
    void* p = nullptr;
    if (::posix_memalign(&p, align, n ? n : 1) != 0) throw std::bad_alloc{};
#endif
    return p;
}
COUNTED_NOINLINE void counted_free(void* p) noexcept {
#if defined(_MSC_VER)
    ::_aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace

void* operator new(std::size_t n) { return counted_alloc(n, alignof(std::max_align_t)); }
void* operator new[](std::size_t n) { return counted_alloc(n, alignof(std::max_align_t)); }
void* operator new(std::size_t n, std::align_val_t a) {
    return counted_alloc(n, static_cast<std::size_t>(a));
}
void* operator new[](std::size_t n, std::align_val_t a) {
    return counted_alloc(n, static_cast<std::size_t>(a));
}
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }
void operator delete(void* p, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { counted_free(p); }

// Execute a plan<long double> at n <= SBO_MAX must not heap-allocate scratch;
// at n > SBO_MAX the soa_scratch heap path must fire.
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
        p.forward(v.data());
        const long after = g_alloc_count.load(std::memory_order_relaxed);
        REQUIRE(after == before);
    }
    // Large: n=8192 > SBO_MAX=4096 — soa_scratch falls back to heap.
    {
        INFO("large n=8192");
        admiral::plan<long double> p({8192});
        std::vector<std::complex<long double>> v(8192, {1, 0});
        p.forward(v.data());  // warmup
        const long before = g_alloc_count.load(std::memory_order_relaxed);
        p.forward(v.data());
        const long after = g_alloc_count.load(std::memory_order_relaxed);
        REQUIRE(after > before);
    }
}
