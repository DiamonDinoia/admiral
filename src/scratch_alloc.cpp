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

// The line is glibc's own ceiling. glibc caps its dynamic mmap threshold at
// DEFAULT_MMAP_THRESHOLD_MAX, 32 MiB on 64-bit, and clamps an application's M_MMAP_THRESHOLD to the
// same value, so 32 MiB is the largest block glibc can ever recycle. At or above it every request
// is mmap'd and munmap'd and the next execute re-faults all of it, which is what snmalloc's
// page-release path avoids. Measured on ccmlin075 (SPR), glibc 2.34, alloc/touch/free to steady
// state: the flip sits between 31 MiB (0 faults per cycle, mallinfo2 hblks 0) and 32 MiB (faults
// every cycle, hblks 1). The threaded gate sweep reads the same crossover from timing.
//
// Below the line snmalloc loses, and NOT for want of something to remove: at 16 threads the 8 MiB
// per-worker blocks re-fault 8240 pages per execute under glibc and 0 under snmalloc, and snmalloc
// is still ~9% slower on that cell. Fewer faults, less kernel time, the same address returned with
// its contents intact, and a slower kernel. A base-offset sweep across the L2 set period and
// PR_SET_THP_DISABLE both fail to recover it. The mechanism is open; the line is where it stops
// mattering. Re-derive it with the threaded gate sweep, never by nudging the constant until one
// cell passes.
inline constexpr std::size_t kSnmallocMinBytes = std::size_t{32} << 20;

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

// snmalloc builds its pagemap on the first allocation, and is_owned() reads that pagemap, so a
// free that runs before any snmalloc allocation must not ask. The flag only ever goes false to
// true, and it is set before the pointer escapes, so every snmalloc pointer sees it set.
std::atomic<bool> g_snmalloc_live{false};
#endif
}  // namespace

long scratch_alloc_count() noexcept { return g_scratch_allocs.load(std::memory_order_relaxed); }

void* scratch_alloc(std::size_t bytes, std::size_t align) {
    g_scratch_allocs.fetch_add(1, std::memory_order_relaxed);
#ifndef ADM_SCRATCH_SYSTEM_ALLOC
    if (!use_system_alloc() && bytes >= kSnmallocMinBytes) {
        void* p = snmalloc::libc::memalign(align, bytes);
        if (p != nullptr) g_snmalloc_live.store(true, std::memory_order_release);
        return p;
    }
#endif
    return ::operator new[](bytes, std::align_val_t{align});
}

// The alignment comes back in because an over-aligned ::operator new[] must be paired with the
// over-aligned ::operator delete[]. It is a compile-time constant at every call site.
void scratch_free(void* p, std::size_t align) noexcept {
#ifndef ADM_SCRATCH_SYSTEM_ALLOC
    // Asking snmalloc who owns the pointer keeps the size out of the deleter, so aligned_buffer
    // stays one word. It is a single pagemap load, and it is on the free path only.
    if (p != nullptr && g_snmalloc_live.load(std::memory_order_acquire) && snmalloc::is_owned(p)) {
        snmalloc::libc::free(p);
        return;
    }
#endif
    ::operator delete[](p, std::align_val_t{align});
}

}  // namespace detail
}  // namespace admiral
