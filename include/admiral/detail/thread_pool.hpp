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
#include <chrono>
#include <cmath>
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
// law below keeps a small transform off a big machine, and a transform large
// enough to thread wants every core the process may use.
// >=1 returned verbatim; the caller decides whether to build a pool.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n) {
    if (n != 0) return n;
    static const std::size_t auto_n = allowed_physical_cores();
    return auto_n;
}

// Physical cores per socket: distinct `physical_package_id` over the affinity mask,
// same sysfs idiom as `allowed_physical_cores` above. Fallback C0 = P when topology
// is absent, i.e. the machine reads as one socket.
[[nodiscard]] inline std::size_t cores_per_socket() {
#if defined(__linux__)
    static const std::size_t c0 = [] {
        cpu_set_t aff;
        if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
            long long pkgs[64]{};   // distinct package ids; > 64 sockets is not a host class
            std::size_t np = 0;
            for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
                if (!CPU_ISSET(cpu, &aff)) continue;
                std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                                "/topology/physical_package_id");
                long long id = -1;
                if (!(f >> id) || id < 0) continue;
                bool seen = false;
                for (std::size_t k = 0; k < np; ++k) seen |= pkgs[k] == id;
                if (!seen && np < 64) pkgs[np++] = id;
            }
            if (np > 0)
                return std::max<std::size_t>(1, allowed_physical_cores() / np);
        }
        return allowed_physical_cores();
    }();
    return c0;
#else
    return allowed_physical_cores();
#endif
}

// cycles -> ns for the law's work estimate: the effective core frequency, measured
// once (~1 ms) as a dependent-multiply latency chain against steady_clock. The chain
// runs at the IMUL r64 latency (3 cycles on every scored host), so 3*iters/elapsed is
// the core's cycles per ns at current boost. Not rdtsc: the invariant TSC ticks at
// base clock while an executing core boosts 1.3-1.6x above it, which would over-price
// W by that ratio and flip the law's picks toward knot on icelake/genoa.
[[nodiscard]] inline double core_cyc_per_ns() {
    static const double cyc_per_ns = [] {
        using clock = std::chrono::steady_clock;
        std::uint64_t x = 0x9E3779B97F4A7C15ull;
        std::uint64_t n  = 0;
        long long ns = 0;
        const auto t0 = clock::now();
        do {
            for (int i = 0; i < 64; ++i)
                x = x * 6364136223846793005ull + 1442695040888963407ull;
            n += 64;
            ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count();
        } while (ns < 1000000);
        volatile std::uint64_t sink = x;   // the chain must not fold away
        (void)sink;
        return static_cast<double>(3 * n) / static_cast<double>(ns);
    }();
    return cyc_per_ns;
}

// Auto-selection law v2 (2026-08-31 campaign, fi/mt t0-modeler-r2.md):
//   nt* = argmin over pow2 nt <= P of  T(nt) = W / min(nt, knee[cls]) + K * Dhat(nt, gap^),
// gap^ the self-consistent per-dispatch gap (two fixed-point iterations), W the serial
// work estimate in ns, K the dispatches per execute. The knee clamps the work term at
// the memory-fabric saturation width; Dhat prices the pool's per-dispatch wake/join.
// pow2 quantization stands: off-divisor counts load-imbalance the passes' static chunks.

// Pocket constants. G0: onset gap — the pair-RTT p50 stays hot at gap <= 30 us on all
// 15 class x family probe cells, so pools leave the hot band only past this column
// (fi/mt t0/topo-probe-results.md, 2026-08-31, jobs 6969062-64, table (a)).
inline constexpr double kPocketOnsetNs = 30e3;
// G1: parked pricing tier — the first pool-probe column in the parked regime (the
// pair-RTT leaves hot in (30, 300] us).
inline constexpr double kPocketTierNs = 100e3;

// One probed host class: thread counts, knee per engine class, and the wake grid.
// knee[cls]: 0 = 1-D four_step_large, 1 = 2-D lines, 2 = 3-D lines.
struct wake_family_row {
    std::size_t P;
    std::size_t C0;
    std::size_t knee[3];
    double dhat[12][6];   // mean_ns - 51 ns over nt grid x gap grid {0,30us,100us,300us,1ms,10ms}
};

// Dhat grids: `mt_scaling --mode=probe` reduced as mean_ns - 51 ns serial floor, the
// 3 ms column dropped (rome deep-C bimodal arm). fi/mt out/probe-<fam>.csv, 2026-08-31.
// Rows past a class's own P repeat its deepest probe row; the argmin never reads past P.
[[nodiscard]] inline const wake_family_row& wake_family_for(std::size_t P, std::size_t C0) {
    static constexpr wake_family_row rows[3] = {
        // icelake (2x32): knees (32,64,64); probe verdict CONFIRMED: sock0/L3 knee = 32
        // exact, node tier additive to 64 (fi/mt t0/topo-probe-results.md mem_scale).
        {64, 32, {32, 64, 64}, {
            {417.6, 2190.3, 3249.3, 44969.9, 69767.1, 71166.9},
            {659.4, 3325.1, 4036.6, 83426.6, 85974.7, 94599.9},
            {863.2, 4290.9, 67169.3, 103528.2, 107990.6, 105262.3},
            {994.6, 4697.1, 56299.3, 121667.0, 115548.8, 122090.8},
            {1285.1, 5051.3, 100853.2, 147251.1, 137009.1, 135459.1},
            {1572.6, 193924.6, 201834.1, 213123.9, 149816.5, 146465.7},
            {66344.2, 260911.9, 195271.4, 276801.4, 198020.8, 173496.2},
            {2629.2, 2608.2, 285792.7, 302605.0, 228081.9, 199131.2},
            {349718.6, 353449.7, 360336.7, 356031.5, 285706.6, 320509.6},
            {478692.6, 466653.6, 469364.1, 462019.9, 396584.9, 436007.0},
            {478692.6, 466653.6, 469364.1, 462019.9, 396584.9, 436007.0},
            {478692.6, 466653.6, 469364.1, 462019.9, 396584.9, 436007.0},
        }},
        // rome (2x64): knees (16,32,32); fsl = one NPS4 node width (probe plateau at
        // nt 16), nd = 2 nodes (probe verdicts + receipt runs fft/fft2-rome).
        {128, 64, {16, 32, 32}, {
            {845.5, 854.3, 4660.3, 5585.7, 12267.9, 29903.1},
            {1221.8, 1200.2, 5997.1, 7244.4, 15280.6, 32800.4},
            {1271.6, 1212.2, 6494.1, 8201.1, 19119.3, 34722.1},
            {1274.6, 1294.8, 6963.1, 8795.1, 20654.7, 113723.9},
            {1351.2, 1321.7, 9080.8, 11508.1, 20858.1, 219841.6},
            {1301.3, 1348.6, 10175.6, 13851.7, 23216.0, 446072.1},
            {7976.1, 2117.7, 15984.7, 21724.2, 32476.7, 456294.1},
            {30277.8, 2163.2, 21722.2, 27728.3, 36682.4, 475597.9},
            {47854.7, 3083.9, 29885.5, 41650.8, 55572.8, 541734.9},
            {80238.7, 3355.9, 48328.7, 54941.1, 64184.7, 553153.7},
            {44709.1, 134241.8, 165196.5, 212308.3, 209231.0, 651361.0},
            {279548.0, 291422.2, 290223.4, 287607.0, 291248.4, 810145.2},
        }},
        // genoa (2x48): knees (32,32,64); fsl/nd2 between the probe CCD knee (2-4) and
        // the socket tier (>48), nd3 = 64: 512^3 measured 439 GB/s aggregate > one
        // socket's 251 GB/s probed cap (receipt fft-genoa.csv; probe mem_scale sock0@48).
        {96, 48, {32, 32, 64}, {
            {803.6, 734.4, 5502.3, 6529.2, 7360.2, 21300.4},
            {1314.9, 1256.1, 7467.7, 8481.8, 9172.5, 23979.8},
            {1694.2, 1656.0, 10002.0, 10627.2, 12025.9, 26398.2},
            {2018.3, 2011.8, 12647.3, 14096.7, 13916.3, 28574.9},
            {2000.8, 1985.7, 17464.6, 17207.0, 18543.0, 31657.5},
            {2094.6, 2158.6, 21817.9, 20548.1, 21579.8, 35592.9},
            {2458.7, 2356.6, 30047.4, 28170.2, 29060.9, 40903.5},
            {2768.7, 2703.4, 41071.3, 35955.3, 37372.9, 47862.6},
            {3237.2, 3005.4, 76411.8, 72413.6, 62735.4, 66301.4},
            {73738.4, 4793.8, 119574.1, 141263.5, 142164.6, 137390.7},
            {261820.4, 231940.8, 236950.1, 275106.1, 296582.1, 266333.0},
            {261820.4, 231940.8, 236950.1, 275106.1, 296582.1, 266333.0},
        }},
    };
    for (const wake_family_row& r : rows)
        if (r.P == P && r.C0 == C0) return r;
    return rows[1];   // unknown host class: the rome grid (the most benign arm)
}

// Dhat(nt, gap_ns) in ns; bilinear over the family grid, gap clamped to the grid,
// result floored at 0 (the floor constant can exceed a mean on the hottest cells).
[[nodiscard]] inline double dhat_ns(const wake_family_row& fam, std::size_t nt, double gap_ns) {
    static constexpr std::size_t nts[12] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    static constexpr double gaps[6] = {0.0, 30e3, 100e3, 300e3, 1e6, 1e7};
    if (nt <= 1) return 0.0;
    if (nt > nts[11]) nt = nts[11];
    gap_ns = std::min(std::max(gap_ns, 0.0), gaps[5]);
    std::size_t hi = 1;
    while (hi < 11 && nts[hi] < nt) ++hi;   // first index with nts[hi] >= nt
    double row[6];
    if (nts[hi] == nt) {
        for (std::size_t j = 0; j < 6; ++j) row[j] = fam.dhat[hi][j];
    } else {
        const std::size_t lo = hi - 1;
        const double t =
            static_cast<double>(nt - nts[lo]) / static_cast<double>(nts[hi] - nts[lo]);
        for (std::size_t j = 0; j < 6; ++j)
            row[j] = (1.0 - t) * fam.dhat[lo][j] + t * fam.dhat[hi][j];
    }
    for (std::size_t i = 0; i + 1 < 6; ++i) {
        if (gap_ns <= gaps[i + 1]) {
            const double t = (gap_ns - gaps[i]) / (gaps[i + 1] - gaps[i]);
            return std::max((1.0 - t) * row[i] + t * row[i + 1], 0.0);
        }
    }
    return std::max(row[5], 0.0);
}

// Size-aware form: the law above, 1 meaning no pool. n != 0 is returned verbatim.
// `dispatch_k` is the engine's dispatches per execute (5 for 1-D four_step_large,
// rank for the N-D batch loops, 1..2 for the axis/strides loops, 0 where the route
// builds no pool). `work_ns` is the serial-work estimate in ns. `cls` indexes the
// knee row: 0 = 1-D four_step_large, 1 = 2-D lines, 2 = 3-D lines.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total,
                                                  std::size_t dispatch_k, double work_ns,
                                                  unsigned cls) {
    if (n != 0) return n;
    if (total < kAutoSerialElems || dispatch_k == 0) return 1;
    static const std::size_t P = resolve_nthreads(0);
    static const std::size_t C0 = cores_per_socket();
    static const wake_family_row& fam = wake_family_for(P, C0);
    const std::size_t knee = fam.knee[std::min(cls, 2u)];
    std::size_t best = 1;
    double best_t = work_ns;
    for (std::size_t nt = 2; nt <= P; nt <<= 1) {
        const double work = work_ns / static_cast<double>(std::min(nt, knee));
        const double gh = work / static_cast<double>(dispatch_k);
        // Pocket rule: a hot pass-gap below G0 at >= half a socket can land the pool
        // in the parked basin; price the dispatch at >= G1 so the argmin prefers a
        // cheaper arm. Probe: pair-RTT table (a) + pool-probe columns cited above.
        const bool pocket = gh < kPocketOnsetNs && nt >= C0 / 2;
        double gap = pocket ? std::max(gh, kPocketTierNs) : gh;
        double dh = dhat_ns(fam, nt, gap);
        for (int i = 0; i < 2; ++i) {   // gap^ fixed point: T(nt)/K is the real gap
            const double t = work + static_cast<double>(dispatch_k) * dh;
            gap = t / static_cast<double>(dispatch_k);
            if (pocket) gap = std::max(gap, kPocketTierNs);
            dh = dhat_ns(fam, nt, gap);
        }
        if (const double t = work + static_cast<double>(dispatch_k) * dh; t < best_t) {
            best_t = t;   // strict < : ties resolve toward the smaller count
            best = nt;
        }
    }
    return best;
}

// Route-blind compat form: one batched-lines dispatch, generic serial-work estimate
// = total * log2(total) * 1.669 cycles with u = codelet_cost_cyc_f64[64]/(64*6)
// (math.hpp, measured leaf price) at the probed clock. Route-aware plans call the
// (dispatch_k, work_ns, cls) form instead.
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total) {
    if (n != 0) return n;
    const double w_ns = static_cast<double>(total) * std::log2(static_cast<double>(total | 1)) *
                        1.669 / core_cyc_per_ns();
    return resolve_nthreads(n, total, 1, w_ns, 1);
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
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t, std::size_t, std::size_t, double,
                                                  unsigned) noexcept { return 1; }
[[nodiscard]] inline double core_cyc_per_ns() noexcept { return 1.0; }

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
