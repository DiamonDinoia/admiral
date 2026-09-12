// The engine's heap scratch, and the only place snmalloc is named.
//
// `soa_scratch` keeps small buffers on the stack and asks the heap for everything else, once
// per execute() on the iterative_dif route: 4*N*sizeof(T) bytes, 1 GiB at N = 2^25 f64. glibc
// unmaps a block that size on free, so the next execute re-faults every page of it, and even
// below the unmap threshold a fresh address surrenders the cache residency the previous call
// built. snmalloc returns the same block and releases only its pages, through the per-platform
// call that lets the OS reclaim them under pressure: MADV_FREE on Linux and the BSDs,
// MADV_FREE_REUSABLE on Apple, its own PAL on Windows.
//
// Measured on ccmlin075 (SPR) at 16 mod 64, iterative_dif, serial, against the shipped
// ::operator new[]: 0.879 at 2^23 f64, 0.885 at 2^24, 0.885 at 2^25, with minor faults per
// execute falling from 640-1024 to 0.5. four_step_large holds its buffers in the plan and is
// the negative control: it does not move on any size.
//
// snmalloc is reached ONLY from here. It does not replace the global operator new, so the
// caller's buffers and the rest of the process keep whatever allocator the application chose.

// AddressSanitizer instruments the global allocator, not snmalloc's own mmaps. Leaving the
// scratch with snmalloc under asan would drop the redzones from the largest buffers the engine
// owns, so a kernel overrunning one would stop being reported. The sanitized build therefore
// takes the global allocator: what has to be checked is admiral's use of the buffer, not the
// allocator that handed it over.
#if defined(__SANITIZE_ADDRESS__)
#define ADM_SCRATCH_SYSTEM_ALLOC 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ADM_SCRATCH_SYSTEM_ALLOC 1
#endif
#endif

#ifndef ADM_SCRATCH_SYSTEM_ALLOC
#include <snmalloc/snmalloc.h>

#include <cstdlib>
#include <cstring>
#endif

#include <atomic>
#include <cstddef>
#include <new>

namespace admiral {
namespace detail {
namespace {
// The seam is the only heap the engine reaches, so counting here is what lets a test assert
// that soa_scratch keeps small n on the stack and falls back for large n. A global
// operator new counter cannot: snmalloc does not go through it.
std::atomic<long> g_scratch_allocs{0};

#ifndef ADM_SCRATCH_SYSTEM_ALLOC
// valgrind grants no single mapping past 32 GiB (measured, valgrind 3.26.0, Linux 5.14), and
// snmalloc's pagemap reservation is larger, so snmalloc's init calls abort() and takes the whole
// process with it at the first large transform. memcheck does not intercept snmalloc's mappings
// either, so under valgrind the global allocator is both the only one that runs and the only one
// that reports an overrun of the scratch. valgrind announces itself by preloading vgpreload_core;
// a process that erases LD_PRELOAD before the first transform hides it and aborts as it would
// without this check. validate.sh's valgrind arm is the gate: break the fallback and every test
// aborts.
bool running_on_valgrind() noexcept {
    const char* preload = std::getenv("LD_PRELOAD");
    return preload != nullptr && std::strstr(preload, "vgpreload") != nullptr;
}

bool use_system_alloc() noexcept {
    static const bool yes = running_on_valgrind();
    return yes;
}
#endif
}  // namespace

long scratch_alloc_count() noexcept { return g_scratch_allocs.load(std::memory_order_relaxed); }

void* scratch_alloc(std::size_t bytes, std::size_t align) {
    g_scratch_allocs.fetch_add(1, std::memory_order_relaxed);
#ifndef ADM_SCRATCH_SYSTEM_ALLOC
    if (!use_system_alloc()) {
        return snmalloc::libc::memalign(align, bytes);
    }
#endif
    return ::operator new[](bytes, std::align_val_t{align});
}

// The alignment comes back in because an over-aligned ::operator new[] must be paired with the
// over-aligned ::operator delete[]. It is a compile-time constant at every call site.
void scratch_free(void* p, std::size_t align) noexcept {
#ifndef ADM_SCRATCH_SYSTEM_ALLOC
    if (!use_system_alloc()) {
        snmalloc::libc::free(p);
        return;
    }
#endif
    ::operator delete[](p, std::align_val_t{align});
}

}  // namespace detail
}  // namespace admiral
