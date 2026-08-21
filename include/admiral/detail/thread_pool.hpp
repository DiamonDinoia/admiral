#pragma once

// ============================================================================
// Fixed fork-join thread pool for N-D batch loops.
//
// All work units are equal (rows/columns/tiles), so static contiguous chunking
// is optimal; a general task-graph scheduler adds nothing.
// ponytail: hand-rolled. Add a dependency only if profiling shows a win from
// work-stealing or irregular scheduling.
//
// Dispatch is a spin-then-park barrier (not mutex+condvar+latch):
//   * publish job, bump atomic epoch (release);
//   * workers spin on epoch (cpu_relax); park on condvar after ~kSpin iters;
//   * join = caller spins on atomic pending-counter (acquire).
// nthreads-1 persistent workers + calling thread run nthreads chunks of [0,n).
// body(begin, end, tid); callers allocate per-chunk scratch once per chunk.
// nthreads==1: byte-identical serial path, zero threads spawned.
// Not re-entrant; no concurrent parallel_for calls (one dispatcher).
//
// ADM_THREADS==0 (CMake -DADM_ENABLE_THREADS=OFF): no worker pool at all, and no
// <thread>/Threads dependency. resolve_nthreads collapses to 1 so callers never
// build a pool and parallel_for runs inline; the class stays a complete type.
// ============================================================================
#include <admiral/detail/config.hpp>   // ADM_THREADS

#include "cache.hpp"                   // kCacheLine (false-sharing padding)

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>     // std::this_thread::yield fallback in cpu_relax (non-x86/aarch64)
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
#include <set>
#include <string>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif
#endif

namespace admiral::detail {
// body(begin, end, tid): callers iterate [begin,end), allocate per-chunk scratch once.
// The concept makes a wrong lambda signature fail at the call site.
template<typename F>
concept ChunkBody = std::invocable<F&, std::size_t, std::size_t, std::size_t>;
}  // namespace admiral::detail

namespace admiral {
namespace detail {

// Threading floor: dispatch cost dominates below this element count.
inline constexpr std::size_t kThreadMinElems = std::size_t{1} << 15;

#if ADM_THREADS

// Ceiling for nthreads=0 auto-selection: tiny transforms never gain from
// oversubscribing a big machine, and the pool spins, so cap the auto count.
inline constexpr std::size_t kMaxAutoThreads = 16;

// Auto-selection heuristic: a pool costs more than it saves below
// kAutoSerialElems elements; above it, one worker per kAutoElemsPerThread,
// capped at the machine's allowed physical cores. Fitted from the threaded
// N-D sweep (benchmark/README.md conventions): threading is break-even to 1.1x
// up to ~25k elements and 2-4x at 32-36k.
inline constexpr std::size_t kAutoSerialElems = std::size_t{1} << 15;
inline constexpr std::size_t kAutoElemsPerThread = std::size_t{1} << 12;

// Distinct physical cores among the CPUs this process is allowed to run on
// (Linux topology + affinity mask; same method as finufft's getOptimalThreadCount).
// hardware_concurrency counts SMT siblings, which share execution resources: the
// wrong count for a spinning pool on unbalanced splits. Falls back to
// hardware_concurrency where topology is absent.
[[nodiscard]] inline std::size_t allowed_physical_cores() {
#if defined(__linux__)
    cpu_set_t aff;
    if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
        std::set<std::pair<std::uint32_t, std::uint32_t>> cores;  // (package id, core id)
        // size_t index: glibc's CPU_ISSET converts its argument to size_t internally.
        for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
            if (!CPU_ISSET(cpu, &aff)) continue;
            const std::string topo =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
            std::ifstream pkg(topo + "physical_package_id"), core(topo + "core_id");
            std::uint32_t p = 0, c = 0;
            if (pkg >> p && core >> c) cores.emplace(p, c);
        }
        if (!cores.empty()) return cores.size();
    }
#endif
    const unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 1 : hc;
}

// 0 -> allowed physical cores (fallback 1), capped at kMaxAutoThreads; cached,
// affinity is fixed for the run in practice.
// >=1 returned verbatim; caller decides whether to build a pool.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n) {
    if (n != 0) return n;
    static const std::size_t auto_n = std::min(allowed_physical_cores(), kMaxAutoThreads);
    return auto_n;
}

// Size-aware form for n == 0: the heuristic count for a transform of `total`
// elements, 1 meaning no pool at all. n != 0 returned verbatim.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total) {
    if (n != 0) return n;
    if (total < kAutoSerialElems) return 1;
    return std::clamp(total / kAutoElemsPerThread, std::size_t{1}, resolve_nthreads(0));
}

// Saturating product for total-element estimates: an overflowing shape resolves
// to the full auto count, and the plan constructor reports the bad shape on its
// own path.
[[nodiscard]] inline std::size_t sat_elems(std::size_t a, std::size_t b) noexcept {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b
               ? std::numeric_limits<std::size_t>::max()
               : a * b;
}

// ~kSpinIters * (pause latency) should exceed a typical inter-dispatch gap so
// back-to-back passes don't pay a wakeup; modest enough to park within ~tens of us.
inline constexpr std::uint32_t kSpinIters = 2048;

class thread_pool {
public:
    explicit thread_pool(std::size_t nthreads) : nthreads_(std::max(nthreads, std::size_t{1})) {
        // nthreads-1 workers (tids 0..nthreads-2); the caller runs the last chunk
        // itself. `tid + 1 < nthreads_`, not `tid < nthreads_ - 1`: the clamp above
        // is what keeps nthreads == 0 from wrapping the bound to SIZE_MAX, and the
        // two must not disagree.
        // A spawn failure here bypasses the dtor; stop and join what started or
        // the joinable threads terminate.
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

    // Run body(begin,end,tid) over nthreads contiguous chunks of [0,n).
    // First exception wins; rethrown after join (no per-task futures).
    template<ChunkBody F>
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

        // Publish job then release-bump epoch; worker acquire on epoch sees job_ and pending_.
        job_ = std::cref(run_chunk);   // NOT copied by workers; alive until join below
        pending_.store(nt, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        // Empty lock ties notify to epoch store; worker mid-park re-checks epoch under mtx_.
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
            // Spin on the epoch for the common back-to-back case; park after kSpin.
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
                    break;   // epoch changed or stopping, so re-evaluate at the top
                }
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            seen = epoch_.load(std::memory_order_acquire);
            job_(tid);                                         // call in place (no copy)
            pending_.fetch_sub(1, std::memory_order_acq_rel);  // signal completion
        }
    }

    // False-sharing padding, NOT SIMD alignment: the one documented constant
    // lives in cache.hpp.

    std::size_t nthreads_;
    // std::thread, not std::jthread: nothing here uses stop_token (the pool has its
    // own stopping_ flag), and AppleClang's libc++ does not ship jthread. The dtor
    // joins explicitly, which is all jthread would buy.
    std::vector<std::thread> workers_;

    // Dispatch line: read-mostly (epoch bumps once/dispatch, stopping once).
    // job_ set before epoch bump, read only after; shares this line safely.
    alignas(kCacheLine) std::function<void(std::size_t)> job_;  // set before epoch bump, read after
    std::atomic<std::uint64_t> epoch_{0};    // bumped per dispatch; workers run when it changes
    std::atomic<bool> stopping_{false};

    // Own line: every participant fetch_sub's it and the caller spins on it, so
    // exclusive churn would bounce the read-mostly dispatch line (false sharing).
    alignas(kCacheLine) std::atomic<std::uint64_t> pending_{0};  // participants still running

    alignas(kCacheLine) std::mutex mtx_;     // only for the park/wake fallback
    std::condition_variable cv_;
    std::mutex exc_mtx_;
    std::exception_ptr first_exc_;
};

#else   // !ADM_THREADS: serial build, no worker pool.

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t) noexcept { return 1; }
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t, std::size_t) noexcept { return 1; }

// Never instantiated (resolve_nthreads==1 => plans never build one); kept
// complete so unique_ptr<thread_pool> stays valid.
class thread_pool {
public:
    explicit thread_pool(std::size_t) {}
    [[nodiscard]] std::size_t size() const noexcept { return 1; }
    template<ChunkBody F>
    void parallel_for(std::size_t n, F&& body) { body(std::size_t{0}, n, std::size_t{0}); }
};

#endif  // ADM_THREADS

// parallel_for threads when will_thread() below says so; otherwise serial-inline
// (single chunk, tid 0).
[[nodiscard]] inline std::size_t pool_size(const thread_pool* pool) {
    return pool ? pool->size() : 1;
}

// n >= 2 rather than n >= 2*size(): thin shapes yield few units, and demanding two per
// thread would run them fully serial. Surplus threads just get empty chunks.
[[nodiscard]] inline bool will_thread(const thread_pool* pool, std::size_t n,
                                      std::size_t total_elems) {
    return pool_size(pool) > 1 && n >= 2 && total_elems >= kThreadMinElems;
}

template<ChunkBody F>
inline void parallel_for(thread_pool* pool, std::size_t n, std::size_t total_elems, F&& body) {
    if (will_thread(pool, n, total_elems)) {
        pool->parallel_for(n, std::forward<F>(body));
    } else {
        body(std::size_t{0}, n, std::size_t{0});   // serial: one chunk, tid 0
    }
}

} // namespace detail
} // namespace admiral
