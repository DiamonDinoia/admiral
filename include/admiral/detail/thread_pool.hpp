#pragma once

#include <admiral/detail/config.hpp>

#include "cache.hpp"
#include "cxx_compat.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <thread>
#include <type_traits>
#include <utility>

#if ADM_THREADS
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
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
template<typename F>
inline constexpr bool is_chunk_body_v =
    std::is_invocable_v<F&, std::size_t, std::size_t, std::size_t>;
#if ADM_CXX20
template<typename F>
concept ChunkBody = is_chunk_body_v<F>;
#endif
}

namespace admiral {
namespace detail {

inline constexpr std::size_t kThreadMinElems = std::size_t{1} << 15;

[[nodiscard]] inline std::size_t sat_elems(std::size_t a, std::size_t b) noexcept {
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b
               ? std::numeric_limits<std::size_t>::max()
               : a * b;
}

#if ADM_THREADS

inline constexpr std::size_t kAutoSerialElems = std::size_t{1} << 15;

[[nodiscard]] inline std::size_t allowed_physical_cores() {
#if defined(__linux__)
    cpu_set_t aff;
    if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
        cpu_set_t cores;
        CPU_ZERO(&cores);
        for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
            if (!CPU_ISSET(cpu, &aff)) continue;
            std::ifstream f("/sys/devices/system/cpu/cpu" + std::to_string(cpu) +
                            "/topology/thread_siblings_list");
            long long first = -1;
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

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n) {
    if (n != 0) return n;
    static const std::size_t auto_n = allowed_physical_cores();
    return auto_n;
}

// (socket, core) names one physical core; SMT siblings share it.
struct pool_topo_row {
    std::size_t cpu;
    long long socket;
    long long core;
};

// One row per allowed cpu, core_id missing => that cpu drops out (untrustworthy row, never
// guessed). cores_per_socket() and options::pin_threads both key off this, so the tree walks
// /sys/.../topology exactly once.
[[nodiscard]] inline const std::vector<pool_topo_row>& topology_rows() {
    static const std::vector<pool_topo_row> rows = [] {
        std::vector<pool_topo_row> out;
#if defined(__linux__)
        cpu_set_t aff;
        if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
            for (std::size_t cpu = 0; cpu < static_cast<std::size_t>(CPU_SETSIZE); ++cpu) {
                if (!CPU_ISSET(cpu, &aff)) continue;
                const std::string dir =
                    "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
                long long core = -1;
                if (!(std::ifstream(dir + "core_id") >> core)) continue;
                long long pkg = 0;
                std::ifstream(dir + "physical_package_id") >> pkg;
                out.push_back({cpu, pkg, core});
            }
        }
#endif
        return out;
    }();
    return rows;
}

[[nodiscard]] inline std::size_t cores_per_socket() {
#if defined(__linux__)
    static const std::size_t c0 = [] {
        long long pkgs[64]{};
        std::size_t np = 0;
        for (const pool_topo_row& r : topology_rows()) {
            bool seen = false;
            for (std::size_t k = 0; k < np; ++k) seen |= pkgs[k] == r.socket;
            if (!seen && np < 64) pkgs[np++] = r.socket;
        }
        return np > 0 ? std::max<std::size_t>(1, allowed_physical_cores() / np)
                      : allowed_physical_cores();
    }();
    return c0;
#else
    return allowed_physical_cores();
#endif
}

// One pin per physical core: the lowest cpu of each (socket, core) pair (siblings collapse),
// ordered socket-major then cpu-ascending.
[[nodiscard]] inline std::vector<std::size_t> pool_pin_list(span<const pool_topo_row> rows) {
    std::vector<pool_topo_row> cores;
    for (const pool_topo_row& r : rows) {
        auto it = std::find_if(cores.begin(), cores.end(), [&](const pool_topo_row& c) {
            return c.socket == r.socket && c.core == r.core;
        });
        if (it == cores.end())
            cores.push_back(r);
        else if (r.cpu < it->cpu)
            it->cpu = r.cpu;
    }
    std::sort(cores.begin(), cores.end(), [](const pool_topo_row& a, const pool_topo_row& b) {
        return a.socket != b.socket ? a.socket < b.socket : a.cpu < b.cpu;
    });
    std::vector<std::size_t> out;
    out.reserve(cores.size());
    for (const pool_topo_row& c : cores) out.push_back(c.cpu);
    return out;
}

// Empty on an unreadable topology or off Linux.
[[nodiscard]] inline const std::vector<std::size_t>& pool_pin_cpus() {
    static const std::vector<std::size_t> pins = pool_pin_list(topology_rows());
    return pins;
}

// Fail-soft: a short mask or a rejected sched_setaffinity leaves the worker unpinned, with one
// stderr note per process.
inline void pool_pin_worker([[maybe_unused]] std::size_t tid,
                            [[maybe_unused]] std::size_t nthreads) noexcept {
#if defined(__linux__)
    static std::atomic<bool> noted{false};
    const std::vector<std::size_t>& pins = pool_pin_cpus();
    // Only nthreads-1 workers are ever spawned (the caller runs the last chunk unmoved), so
    // tid never exceeds nthreads-2 and the mask only needs to cover nthreads-1 cores.
    if (pins.size() < nthreads - 1) {
        if (!noted.exchange(true, std::memory_order_relaxed))
            std::fprintf(stderr, "admiral: thread pinning disabled, mask holds %zu cores for "
                                 "nthreads=%zu\n", pins.size(), nthreads);
        return;
    }
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(pins[tid], &one);
    if (sched_setaffinity(0, sizeof one, &one) != 0 &&
        !noted.exchange(true, std::memory_order_relaxed))
        std::fputs("admiral: thread pinning disabled, sched_setaffinity failed\n", stderr);
#endif
}

inline constexpr double kPocketOnsetCyc = 81e3;
inline constexpr double kPocketTierCyc = 270e3;

struct wake_family_row {
    std::size_t P;
    std::size_t C0;
    std::size_t knee[3][2];
    std::size_t gateMB[3];
    double dhat[12][6];
};

[[nodiscard]] inline const wake_family_row& wake_family_for(std::size_t P, std::size_t C0) {
    // Cycles. Fitted constant: each entry is the ns wake law of commit a4079596 times 2.70,
    // the median of 7 runs of the removed timed loop (2.69 to 2.76) on a Xeon w5-3435X.
    // No count replaces it. The ns table is a fit of measured wake times, not a count of
    // instructions, and the clock of a host changes the cycles of one ns.
    static constexpr wake_family_row rows[3] = {
        {64, 32, {{32, 32}, {32, 64}, {32, 64}}, {48, 48, 48}, {
            {1127.52, 5913.81, 8773.11, 121418.73, 188371.17, 192150.63},
            {1780.38, 8977.77, 10898.82, 225251.82, 232131.69, 255419.73},
            {2330.64, 11585.43, 181357.11, 279526.14, 291574.62, 284208.21},
            {2685.42, 12682.17, 152008.11, 328500.9, 311981.76, 329645.16},
            {3469.77, 13638.51, 272303.64, 397577.97, 369924.57, 365739.57},
            {4246.02, 523596.42, 544952.07, 575434.53, 404504.55, 395457.39},
            {179129.34, 704462.13, 527232.78, 747363.78, 534656.16, 468439.74},
            {7098.84, 7042.14, 771640.29, 817033.5, 615821.13, 537654.24},
            {944240.22, 954314.19, 972909.09, 961285.05, 771407.82, 865375.92},
            {1292470.02, 1259964.72, 1267283.07, 1247453.73, 1070779.23, 1177218.9},
            {1292470.02, 1259964.72, 1267283.07, 1247453.73, 1070779.23, 1177218.9},
            {1292470.02, 1259964.72, 1267283.07, 1247453.73, 1070779.23, 1177218.9},
        }},
        {128, 64, {{16, 32}, {16, 32}, {16, 32}}, {64, 256, 256}, {
            {2282.85, 2306.61, 12582.81, 15081.39, 33123.33, 80738.37},
            {3298.86, 3240.54, 16192.17, 19559.88, 41257.62, 88561.08},
            {3433.32, 3272.94, 17534.07, 22142.97, 51622.11, 93749.67},
            {3441.42, 3495.96, 18800.37, 23746.77, 55767.69, 307054.53},
            {3648.24, 3568.59, 24518.16, 31071.87, 56316.87, 593572.32},
            {3513.51, 3641.22, 27474.12, 37399.59, 62683.2, 1204394.67},
            {21535.47, 5717.79, 43158.69, 58655.34, 87687.09, 1231994.07},
            {81750.06, 5840.64, 58649.94, 74866.41, 99042.48, 1284114.33},
            {129207.69, 8326.53, 80690.85, 112457.16, 150046.56, 1462684.23},
            {216644.49, 9060.93, 130487.49, 148340.97, 173298.69, 1493514.99},
            {120714.57, 362452.86, 446030.55, 573232.41, 564923.7, 1758674.7},
            {754779.6, 786839.94, 783603.18, 776538.9, 786370.68, 2187392.04},
        }},
        {96, 48, {{32, 32}, {32, 64}, {32, 64}}, {512, 512, 512}, {
            {2169.72, 1982.88, 14856.21, 17628.84, 19872.54, 57511.08},
            {3550.23, 3391.47, 20162.79, 22900.86, 24765.75, 64745.46},
            {4574.34, 4471.2, 27005.4, 28693.44, 32469.93, 71275.14},
            {5449.41, 5431.86, 34147.71, 38061.09, 37574.01, 77152.23},
            {5402.16, 5361.39, 47154.42, 46458.9, 50066.1, 85475.25},
            {5655.42, 5828.22, 58908.33, 55479.87, 58265.46, 96100.83},
            {6638.49, 6362.82, 81127.98, 76059.54, 78464.43, 110439.45},
            {7475.49, 7299.18, 110892.51, 97079.31, 100906.83, 129229.02},
            {8740.44, 8114.58, 206311.86, 195516.72, 169385.58, 179013.78},
            {199093.68, 12943.26, 322850.07, 381411.45, 383844.42, 370954.89},
            {706915.08, 626240.16, 639765.27, 742786.47, 800771.67, 719099.1},
            {706915.08, 626240.16, 639765.27, 742786.47, 800771.67, 719099.1},
        }},
    };
    for (const wake_family_row& r : rows)
        if (r.P == P && r.C0 == C0) return r;
    return rows[1];
}

[[nodiscard]] inline double dhat_cyc(const wake_family_row& fam, std::size_t nt, double gap_cyc) {
    static constexpr std::size_t nts[12] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    static constexpr double gaps[6] = {0.0, 81e3, 270e3, 810e3, 2.7e6, 27e6};
    if (nt <= 1) return 0.0;
    if (nt > nts[11]) nt = nts[11];
    gap_cyc = std::min(std::max(gap_cyc, 0.0), gaps[5]);
    std::size_t row_i = 0;
    while (row_i < 11 && nts[row_i] < nt) ++row_i;
    if (nts[row_i] != nt && row_i > 0) --row_i;
    const double* row = fam.dhat[row_i];
    for (std::size_t i = 0; i + 1 < 6; ++i) {
        if (gap_cyc <= gaps[i + 1]) {
            const double t = (gap_cyc - gaps[i]) / (gaps[i + 1] - gaps[i]);
            return std::max((1.0 - t) * row[i] + t * row[i + 1], 0.0);
        }
    }
    return std::max(row[5], 0.0);
}

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total,
                                                  std::size_t dispatch_k, double work_cyc,
                                                  unsigned cls) {
    if (n != 0) return n;
    if (total < kAutoSerialElems || dispatch_k == 0) return 1;
    static const std::size_t P = resolve_nthreads(0);
    static const std::size_t C0 = cores_per_socket();
    static const wake_family_row& fam = wake_family_for(P, C0);
    const std::size_t cls_i = std::min(cls, 2u);
    const bool deep = static_cast<double>(total) * 16.0 >
                      static_cast<double>(fam.gateMB[cls_i]) * 1e6;
    const std::size_t knee = fam.knee[cls_i][deep];
    std::size_t best = 1;
    double best_t = work_cyc;
    for (std::size_t nt = 2; nt <= P; nt <<= 1) {
        const double work = work_cyc / static_cast<double>(std::min(nt, knee));
        const double gh = work / static_cast<double>(dispatch_k);
        const bool pocket = gh < kPocketOnsetCyc && nt >= C0 / 2;
        double gap = pocket ? std::max(gh, kPocketTierCyc) : gh;
        double dh = dhat_cyc(fam, nt, gap);
        for (int i = 0; i < 2; ++i) {
            const double t = work + static_cast<double>(dispatch_k) * dh;
            gap = t / static_cast<double>(dispatch_k);
            if (pocket) gap = std::max(gap, kPocketTierCyc);
            dh = dhat_cyc(fam, nt, gap);
        }
        if (const double t = work + static_cast<double>(dispatch_k) * dh; t < best_t) {
            best_t = t;
            best = nt;
        }
    }
    return best;
}

inline constexpr std::uint32_t kSpinIters = 2048;

class thread_pool {
public:
    // `pin` (options::pin_threads) pins each spawned worker to one physical core; the calling
    // thread (chunk nt-1) is never moved.
    explicit thread_pool(std::size_t nthreads, bool pin = false)
        : nthreads_(std::max(nthreads, std::size_t{1})) {
        try {
            for (std::size_t tid = 0; tid + 1 < nthreads_; ++tid)
                workers_.emplace_back([this, tid, pin] {
                    if (pin) pool_pin_worker(tid, nthreads_);
                    worker_loop(tid);
                });
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    ~thread_pool() { stop_and_join(); }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    [[nodiscard]] std::size_t size() const noexcept { return nthreads_; }

#if ADM_CXX20
    template<ChunkBody F>
#else
    template<typename F, std::enable_if_t<is_chunk_body_v<F>, int> = 0>
#endif
    void parallel_for(std::size_t n, F&& body) {
        std::lock_guard<std::mutex> dlk(dispatch_mtx_);
        const std::size_t nt = nthreads_;
        const std::size_t chunk = (n + nt - 1) / nt;
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

        // cppcheck-suppress danglingLifetime  // the spin-wait below joins every worker
        // before run_chunk dies, so job_ is never read past its referent's lifetime.
        job_ = std::cref(run_chunk);
        pending_.store(nt, std::memory_order_relaxed);
        epoch_.fetch_add(1, std::memory_order_release);
        { std::lock_guard lk(mtx_); }
        cv_.notify_all();

        run_chunk(nt - 1);
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        while (pending_.load(std::memory_order_acquire) != 0) cpu_relax();
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
                    break;
                }
            }
            if (stopping_.load(std::memory_order_acquire)) return;
            seen = epoch_.load(std::memory_order_acquire);
            job_(tid);
            pending_.fetch_sub(1, std::memory_order_acq_rel);
        }
    }

    std::size_t nthreads_;
    std::vector<std::thread> workers_;

    alignas(kCacheLine) std::function<void(std::size_t)> job_;
    std::atomic<std::uint64_t> epoch_{0};
    std::atomic<bool> stopping_{false};

    alignas(kCacheLine) std::atomic<std::uint64_t> pending_{0};

    alignas(kCacheLine) std::mutex mtx_;
    std::condition_variable cv_;
    std::mutex exc_mtx_;
    std::exception_ptr first_exc_;
    // Serializes callers to `parallel_for`; distinct from `mtx_` so workers
    // can block on `mtx_` inside `cv_.wait` without deadlocking.
    alignas(kCacheLine) std::mutex dispatch_mtx_;
};

#else

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t) noexcept { return 1; }
[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t, std::size_t, std::size_t, double,
                                                  unsigned) noexcept { return 1; }

class thread_pool {
public:
    explicit thread_pool(std::size_t, bool = false) {}
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

#endif

[[nodiscard]] inline std::size_t pool_size(const thread_pool* pool) {
    return pool ? pool->size() : 1;
}

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
        body(std::size_t{0}, n, std::size_t{0});
    }
}

}
}
