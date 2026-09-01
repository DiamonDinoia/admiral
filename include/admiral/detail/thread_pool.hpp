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

[[nodiscard]] inline std::size_t cores_per_socket() {
#if defined(__linux__)
    static const std::size_t c0 = [] {
        cpu_set_t aff;
        if (sched_getaffinity(0, sizeof(aff), &aff) == 0) {
            long long pkgs[64]{};
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
        [[maybe_unused]] volatile std::uint64_t sink = x;
        return static_cast<double>(3 * n) / static_cast<double>(ns);
    }();
    return cyc_per_ns;
}

inline constexpr double kPocketOnsetNs = 30e3;
inline constexpr double kPocketTierNs = 100e3;

struct wake_family_row {
    std::size_t P;
    std::size_t C0;
    std::size_t knee[3][2];
    std::size_t gateMB[3];
    double dhat[12][6];
};

[[nodiscard]] inline const wake_family_row& wake_family_for(std::size_t P, std::size_t C0) {
    static constexpr wake_family_row rows[3] = {
        {64, 32, {{32, 32}, {32, 64}, {32, 64}}, {48, 48, 48}, {
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
        {128, 64, {{16, 32}, {16, 32}, {16, 32}}, {64, 256, 256}, {
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
        {96, 48, {{32, 32}, {32, 64}, {32, 64}}, {512, 512, 512}, {
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
    return rows[1];
}

[[nodiscard]] inline double dhat_ns(const wake_family_row& fam, std::size_t nt, double gap_ns) {
    static constexpr std::size_t nts[12] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    static constexpr double gaps[6] = {0.0, 30e3, 100e3, 300e3, 1e6, 1e7};
    if (nt <= 1) return 0.0;
    if (nt > nts[11]) nt = nts[11];
    gap_ns = std::min(std::max(gap_ns, 0.0), gaps[5]);
    std::size_t row_i = 0;
    while (row_i < 11 && nts[row_i] < nt) ++row_i;
    if (nts[row_i] != nt && row_i > 0) --row_i;
    const double* row = fam.dhat[row_i];
    for (std::size_t i = 0; i + 1 < 6; ++i) {
        if (gap_ns <= gaps[i + 1]) {
            const double t = (gap_ns - gaps[i]) / (gaps[i + 1] - gaps[i]);
            return std::max((1.0 - t) * row[i] + t * row[i + 1], 0.0);
        }
    }
    return std::max(row[5], 0.0);
}

[[nodiscard]] inline std::size_t resolve_nthreads(std::size_t n, std::size_t total,
                                                  std::size_t dispatch_k, double work_ns,
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
    double best_t = work_ns;
    for (std::size_t nt = 2; nt <= P; nt <<= 1) {
        const double work = work_ns / static_cast<double>(std::min(nt, knee));
        const double gh = work / static_cast<double>(dispatch_k);
        const bool pocket = gh < kPocketOnsetNs && nt >= C0 / 2;
        double gap = pocket ? std::max(gh, kPocketTierNs) : gh;
        double dh = dhat_ns(fam, nt, gap);
        for (int i = 0; i < 2; ++i) {
            const double t = work + static_cast<double>(dispatch_k) * dh;
            gap = t / static_cast<double>(dispatch_k);
            if (pocket) gap = std::max(gap, kPocketTierNs);
            dh = dhat_ns(fam, nt, gap);
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
    explicit thread_pool(std::size_t nthreads) : nthreads_(std::max(nthreads, std::size_t{1})) {
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
[[nodiscard]] inline double core_cyc_per_ns() noexcept { return 1.0; }

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
