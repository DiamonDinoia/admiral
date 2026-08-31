#pragma once

// ============================================================================
// Fixed fork-join pool for N-D batch loops: equal units, static contiguous chunking.
// Dispatch protocol:
//   1. The dispatcher publishes `job_` and release-bumps the atomic `epoch_`.
//   2. Workers spin on `epoch_` with `cpu_relax` and park on a condvar after
//      `kSpinIters`.
//   3. Join: the caller spins on the `pending_` counter.
// `nthreads`-1 workers plus the caller run `nthreads` chunks of [0,n). Body signature:
// `body(begin, end, tid)`. Not re-entrant: one dispatcher at a time. If `nthreads`==1,
// the serial path runs with no threads. If `ADM_THREADS`==0, there is no pool:
// `resolve_nthreads` returns 1 and the class stays a complete type.
// ============================================================================
#include <admiral/detail/config.hpp>   // `ADM_THREADS`

#include "cache.hpp"                   // `kCacheLine` (false-sharing padding)
#include "cxx_compat.hpp"              // `ADM_CXX20`

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>     // `std::this_thread::yield` fallback in `cpu_relax` (non-x86/aarch64)
#include <type_traits>
#include <utility>

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}


#if ADM_THREADS
#include <atomic>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace admiral::detail {
// Body signature is `body(begin, end, tid)`; `is_chunk_body_v` enforces it at the call site.
template<typename F>
inline constexpr bool is_chunk_body_v =
    std::is_invocable_v<F&, std::size_t, std::size_t, std::size_t>;
#if ADM_CXX20
template<typename F>
concept ChunkBody = is_chunk_body_v<F>;
#endif
}  // namespace admiral::detail

namespace admiral {
namespace detail {

// Threading floor: dispatch cost dominates below `kThreadMinElems` elements.
inline constexpr std::size_t kThreadMinElems = std::size_t{1} << 15;

// Saturating product for total-element estimates: an overflowing shape resolves to the
// full auto count; the plan constructor reports the bad shape on its own path.
[[nodiscard]] inline std::size_t sat_elems(std::size_t a, std::size_t b) noexcept {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b
               ? std::numeric_limits<std::size_t>::max()
               : a * b;
}

#if ADM_THREADS

// Auto-selection law: serial below `kAutoSerialElems`; above it the count rises
// in log2(total) steps of a power of two, capped by the machine's allowed
// physical cores. The limiting cost is the pool's per-dispatch wake/join
// latency, which grows with the worker count and must amortize against
// per-thread work; the pow2 floor keeps the count a divisor of the passes'
// radix-structured unit counts, which load-imbalance the static chunks at
// off-divisor counts.
inline constexpr std::size_t kAutoSerialElems = std::size_t{1} << 15;

// Distinct physical cores among the CPUs this process may run on (Linux topology
// plus the affinity mask; the same method as finufft's getOptimalThreadCount).
// `hardware_concurrency` counts SMT siblings, which share execution resources: the
// wrong count for a spinning pool on unbalanced splits. The fallback is
// `hardware_concurrency` where topology is absent.
//
// `thread_siblings_list` names the core, not `core_id`: sysfs prints the mask in
// ascending order, so every sibling shares the first id in the list and the core
// counts once. `core_id` is -1 on arm64/riscv without firmware topology, which
// would fold the whole machine onto one core.
[[nodiscard]] inline std::size_t allowed_physical_cores() {
#if defined(__linux__)
    cpu_set_t aff;
    if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
        cpu_set_t cores;
        CPU_ZERO(&cores);
        // `std::size_t` index: glibc's `CPU_ISSET` converts its argument to
        // `std::size_t` internally.
        for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
            if (!CPU_ISSET(cpu, &aff)) continue;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                            "/topology/thread_siblings_list");
            long long first = -1;   // the read stops at the ',' or '-' ending the first id
            if (!(f >> first) || first < 0 || first >= CPU_SETSIZE)
                first = static_cast<long long>(cpu);
            CPU_SET(static_cast<std::size_t>(first), &cores);
        }
        if (const int n = CPU_COUNT(&cores); n > 0) return static_cast<std::size_t>(n);
    }
#endif
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : hc;
}

// 0 -> allowed physical cores (fallback 1). The count is cached in a static:
// affinity is fixed for the run in practice. No flat ceiling: the size-aware
// overload below keeps a small transform off a big machine, and a transform large
// enough to thread wants every core the process may use.
// >=1 returned verbatim; the caller decides whether to build a pool.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n) {
    if (n != 0) return n;
    static const std::size_t auto_n = allowed_physical_cores();
    return auto_n;
}

// Size-aware form: the fitted law for total elements, 1 meaning no pool. n != 0 verbatim.
// Ramp: 8 threads at [2^15, 2^17), +1 octave per octave to 16 at [2^17, 2^22),
// 32 at [2^22, 2^26), 64 above; the pow2 floor means only those steps are reachable.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total) {
    if (n != 0) return n;
    if (total < kAutoSerialElems) return 1;
    const std::size_t lg = static_cast<std::size_t>(bit_width(total)) - 1;
    const std::size_t ramp =
        lg < 17 ? 8 : lg < 22 ? 16 + (lg - 17) * 2 : 24 + (lg - 21) * 8;
    return std::min(bit_floor(ramp), resolve_nthreads(0));
}

// `kSpinIters` * pause latency must cover a typical inter-dispatch gap yet park quickly
// on an idle dispatch stream.
inline constexpr std::uint32_t kSpinIters = 2048;

class thread_pool {
public:
    explicit thread_pool(std::size_t nthreads) : nthreads_(std::max(nthreads, std::size_t{1})) {
        // `nthreads`-1 workers; the caller runs the last chunk. The bound is
        // `tid + 1 < nthreads_` so the clamp above keeps `nthreads` == 0 from wrapping to
        // `SIZE_MAX`. On spawn failure, join what started or the joinable threads
        // terminate.
        try {
            for (std::size_t tid = 0; tid + 1 < nthreads_; ++tid)
                workers_.emplace_back([this, tid] { worker_loop(tid); });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    ~thread_pool() { stop_and_join(); }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return nthreads_; }

    // Run `body(begin, end, tid)` over `nthreads_` contiguous chunks of [0,n).
    // The first exception wins; rethrown after join (no per-task futures).
#if ADM_CXX20
    template<ChunkBody F>
#else
    template<typename F, std::enable_if_t<is_chunk_body_v<F>, int> = 0>
#endif
    void parallel_for(std::size_t n, F&& body) {
        const std::size_t nt = nthreads_;
        const std::size_t chunk = (n + nt - 1) / nt;   // ceil, last chunk may be short/empty
        first_exc_ = nullptr;

        auto run_chunk = [&, chunk, n](std::size_t tid) {
            const std::size_t begin = std::min(tid * chunk, n);
            const std::size_t end   = std::min(begin + chunk, n);
            try {
                if (begin < end) body(begin, end, tid);
            } catch (...) {
                std::lock_guard lk(exc_mtx_);
                if (!first_exc_) first_exc_ = std::current_exception();
            }
        };

        // Publish the job, then release-bump `epoch_`. A worker acquire on `epoch_` sees
        // `job_` and `pending_`.
        job_ = std::cref(run_chunk);   // not copied by workers; alive until join below
        pending_.store(nt, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        // The empty lock ties the notify to the `epoch_` store; a worker mid-park
        // re-checks `epoch_` under `mtx_`.
        { std::lock_guard lk(mtx_); }
        cv_.notify_all();

        run_chunk(nt - 1);                                  // caller runs the last chunk
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        while (pending_.load(std::memory_order_acquire) != 0) cpu_relax();   // join
        if (first_exc_) std::rethrow_exception(first_exc_);
    }

private:
    void stop_and_join() {
        { std::lock_guard lk(mtx_); stopping_.store(true, std::memory_order_release); }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    void worker_loop(std::size_t tid) {
        std::uint64_t seen = 0;
        for (;;) {
            // Spin on `epoch_` for the common back-to-back case; park after `kSpinIters`.
            std::uint32_t spins = 0;
            while (epoch_.load(std::memory_order_acquire) == seen) {
                if (stopping_.load(std::memory_order_acquire)) return;
                if (++spins < kSpinIters) {
                    cpu_relax();
                } else {
                    std::unique_lock lk(mtx_);
                    cv_.wait(lk, [&] {
                        return epoch_.load(std::memory_order_acquire) != seen
                            || stopping_.load(std::memory_order_acquire);
                    });
                    break;   // `epoch_` changed or `stopping_` is set; re-evaluate at the top
                }
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            seen = epoch_.load(std::memory_order_acquire);
            job_(tid);                                         // call in place (no copy)
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    // False-sharing padding, not SIMD alignment; `cache.hpp` documents the constant.

    std::size_t nthreads_;
    // `std::thread`, not `std::jthread`: nothing uses `stop_token` and AppleClang's
    // libc++ lacks `jthread`.
    std::vector<std::thread> workers_;

    // Dispatch line, read-mostly. `job_` is set before the `epoch_` bump and read only after.
    alignas(kCacheLine) std::function<void(std::size_t)> job_;
    std::atomic<std::uint64_t> epoch_{0};    // bumped per dispatch; a bump wakes the workers
    std::atomic<bool> stopping_{false};

    // Own line: `fetch_sub` churn plus the join spin would bounce the dispatch line.
    alignas(kCacheLine) std::atomic<std::uint64_t> pending_{0};  // participants still running

    alignas(kCacheLine) std::mutex mtx_;     // only for the park/wake fallback
    std::condition_variable cv_;
    std::mutex exc_mtx_;
    std::exception_ptr first_exc_;
};

#else   // !ADM_THREADS: serial build, no worker pool.

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t) noexcept { return 1; }
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t, std::size_t) noexcept { return 1; }

// Kept complete (never instantiated: `resolve_nthreads` == 1) so `std::unique_ptr` stays valid.
class thread_pool {
public:
    explicit thread_pool(std::size_t) {}
    [[nodiscard]] std::size_t size() const noexcept { return 1; }
#if ADM_CXX20
    template<ChunkBody F>
#else
    template<typename F, std::enable_if_t<is_chunk_body_v<F>, int> = 0>
#endif
    void parallel_for(std::size_t n, F&& body) {
        body(std::size_t{0}, n, std::size_t{0});
    }
};

#endif  // ADM_THREADS

// Threads when `will_thread()` says so; otherwise one serial chunk, tid 0.
[[nodiscard]] inline std::size_t pool_size(const thread_pool* pool) {
    return pool ? pool->size() : 1;
}

// n >= 2, not 2*`size()`: thin shapes yield few units; surplus threads get empty chunks.
[[nodiscard]] inline bool will_thread(const thread_pool* pool, std::size_t n,
                                      std::size_t total_elems) {
    return pool_size(pool) > 1 && n >= 2 && total_elems >= kThreadMinElems;
}

#if ADM_CXX20
template<ChunkBody F>
#else
template<typename F, std::enable_if_t<is_chunk_body_v<F>, int> = 0>
#endif
inline void parallel_for(thread_pool* pool, std::size_t n, std::size_t total_elems, F&& body) {
    if (will_thread(pool, n, total_elems)) {
        pool->parallel_for(n, std::forward<F>(body));
    } else {
        body(std::size_t{0}, n, std::size_t{0});   // serial: one chunk, tid 0
    }
}

} // namespace detail
} // namespace admiral
