#pragma once

// ============================================================================
// Fixed fork-join thread pool for the N-D batch loops.
//
// The parallel work is a *fixed fork-join over regular batch loops* — every row
// / column / tile costs the same — so static contiguous chunking is optimal and
// a general task-graph scheduler (BS::thread_pool et al.) buys nothing.
//
//   ponytail: hand-rolled static pool, ~90 LOC. Add a dependency only if a
//   profile ever shows a scheduling win (irregular work, work-stealing).
//
// Dispatch is a LOW-LATENCY spin-then-park barrier, not a mutex+condvar+latch
// round trip. A plan runs many transforms back-to-back; the fork-join sits on
// the hot path of each axis pass, so its cost must be ~microseconds, not tens.
//   * dispatch publishes the job and bumps an atomic epoch (release);
//   * workers spin on the epoch (cpu_relax) for the common back-to-back case and
//     only PARK on a condvar after ~kSpin idle iterations, so an idle plan does
//     not burn cores;
//   * the join is an atomic pending-counter the caller spins on (acquire).
// nthreads-1 persistent workers plus the calling thread run `nthreads` contiguous
// chunks of [0,n). Body signature is body(begin,end,tid) so callers allocate
// per-chunk scratch ONCE per chunk. The pool is only constructed when
// nthreads > 1; nthreads == 1 keeps the byte-identical serial path, zero threads.
//
// Not re-entrant / not safe for concurrent parallel_for calls (one dispatcher).
// ============================================================================

#include <algorithm>
#include <atomic>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace admiral::detail {
// A chunk body is invoked as body(begin, end, tid) — callers allocate per-chunk
// scratch once and iterate [begin,end). Constrains parallel_for so a wrong lambda
// signature is a clear compile error at the call site, not deep in the template.
template<typename F>
concept ChunkBody = std::invocable<F&, std::size_t, std::size_t, std::size_t>;
}  // namespace admiral::detail

namespace admiral {
namespace detail {

// Element-count floor below which threading is skipped (dispatch cost dominates).
// ponytail: fixed floor, tune if a profile says so.
inline constexpr std::size_t kThreadMinElems = std::size_t{1} << 15;

// Ceiling for nthreads=0 auto-selection: tiny transforms never gain from over
// subscribing a big machine, and the pool spins, so cap the auto count.
inline constexpr std::size_t kMaxAutoThreads = 16;

// Resolve a user-supplied thread count at the API boundary, before the pool is
// built (the pool ctor does nthreads-1, so 0 would underflow). 0 -> hardware
// concurrency (fallback 1) capped at kMaxAutoThreads; anything >=1 is returned
// verbatim (the caller still applies the >1 gate that decides whether to build a
// pool at all).
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n) noexcept {
    if (n != 0) return n;
    const unsigned hc = std::thread::hardware_concurrency();
    return std::min<std::size_t>(hc == 0 ? 1 : hc, kMaxAutoThreads);
}

// Spin iterations a parked-eligible worker burns on the epoch before falling back
// to the condvar. ~kSpin * (pause latency) should exceed a typical inter-dispatch
// gap so back-to-back passes never pay a wakeup; kept modest so an idle plan
// parks within tens of microseconds and stops burning cores.
inline constexpr int kSpinIters = 2048;

inline void cpu_relax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

class thread_pool {
public:
    explicit thread_pool(std::size_t nthreads) : nthreads_(nthreads) {
        // nthreads-1 workers; the caller runs the last chunk itself.
        for (std::size_t tid = 0; tid < nthreads_ - 1; ++tid)
            workers_.emplace_back([this, tid] { worker_loop(tid); });
    }

    ~thread_pool() {
        { std::lock_guard lk(mtx_); stopping_.store(true, std::memory_order_release); }
        cv_.notify_all();
        // std::jthread joins each worker on destruction (after this body runs).
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return nthreads_; }

    // Run body(begin, end, tid) over `nthreads` contiguous chunks of [0,n).
    // First exception from any chunk wins and is rethrown after the join
    // (ponytail: first-exception-wins, no per-task futures).
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

        // Publish the job, then release-bump the epoch: a worker that observes the
        // new epoch (acquire) is guaranteed to see this job_ and pending_.
        job_ = std::cref(run_chunk);   // NOT copied by workers; alive until join below
        pending_.store(nt, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        // Wake any PARKED workers. The empty lock ties the notify to the epoch
        // store so a worker mid-way to parking cannot miss it (it re-checks the
        // epoch under mtx_ in its wait predicate).
        { std::lock_guard lk(mtx_); }
        cv_.notify_all();

        run_chunk(nt - 1);                                  // caller runs the last chunk
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        while (pending_.load(std::memory_order_acquire) != 0) cpu_relax();   // join
        if (first_exc_) std::rethrow_exception(first_exc_);
    }

private:
    void worker_loop(std::size_t tid) {
        std::uint64_t seen = 0;
        for (;;) {
            // Spin on the epoch for the common back-to-back case; park after kSpin.
            int spins = 0;
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
                    break;   // epoch changed or stopping — re-evaluate at the top
                }
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            seen = epoch_.load(std::memory_order_acquire);
            job_(tid);                                         // call in place (no copy)
            pending_.fetch_sub(1, std::memory_order_acq_rel);  // signal completion
        }
    }

    // Assumed cache-line size for false-sharing padding. A plain constant (not
    // std::hardware_destructive_interference_size) to dodge the GCC ABI warning;
    // 64 B is correct on every ISA this library targets.
    static constexpr std::size_t kCacheLine = 64;

    std::size_t nthreads_;
    std::vector<std::jthread> workers_;

    // Dispatch line: READ-MOSTLY during the worker spin (epoch_ bumps once per
    // dispatch, stopping_ once at teardown), so every worker can hammer-load it
    // with no invalidations. job_ is set before the epoch bump and read only after,
    // so it shares this line harmlessly. alignas starts a fresh line here so the
    // vector/size above cannot bleed in.
    alignas(kCacheLine) std::function<void(std::size_t)> job_;  // set before epoch bump, read after
    std::atomic<std::uint64_t> epoch_{0};    // bumped per dispatch; workers run when it changes
    std::atomic<bool> stopping_{false};

    // Completion counter on its OWN line: every participant fetch_sub's it and the
    // caller spins loading it, so its exclusive-state churn would bounce the
    // read-mostly dispatch line above — classic false sharing. Isolate it.
    alignas(kCacheLine) std::atomic<std::uint64_t> pending_{0};  // participants still running

    alignas(kCacheLine) std::mutex mtx_;     // only for the park/wake fallback
    std::condition_variable cv_;
    std::mutex exc_mtx_;
    std::exception_ptr first_exc_;
};

// Free overload of parallel_for over a (possibly null) pool: run body(begin,end,
// tid) either serial-inline (single chunk) or on the pool. Threads only when a
// pool exists, there is enough work to amortize dispatch (>= 2*nthreads chunks)
// and the transform is large enough (>= floor). A null pool (nthreads==1) takes
// the byte-identical serial path with no dispatch overhead — which is why this is
// a free overload, not a member: the tuned single-thread path spawns and touches
// nothing.
// One-shot temp pool for the free-function transform surfaces: resolve nthreads
// (0 -> hardware_concurrency) then build a pool iff >1. Null == serial path.
[[nodiscard]] inline std::unique_ptr<thread_pool> make_temp_pool(std::size_t nthreads) {
    nthreads = resolve_nthreads(nthreads);
    return nthreads > 1 ? std::make_unique<thread_pool>(nthreads) : nullptr;
}

template<ChunkBody F>
inline void parallel_for(thread_pool* pool, std::size_t n, std::size_t total_elems, F&& body) {
    if (pool && n >= 2 * pool->size() && total_elems >= kThreadMinElems) {
        pool->parallel_for(n, std::forward<F>(body));
    } else {
        body(std::size_t{0}, n, std::size_t{0});   // serial: one chunk, tid 0
    }
}

} // namespace detail
} // namespace admiral
