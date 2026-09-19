// options::pin_threads storage, pin-list derivation and application; contract in
// include/admiral/detail/thread_pool.hpp.
#include <admiral/detail/thread_pool.hpp>

#if ADM_THREADS

#include <atomic>
#include <iostream>

#if defined(__linux__)
#include <cstdio>
#include <fstream>
#include <sched.h>
#elif defined(__APPLE__)
#include <mach/thread_act.h>
#include <mach/thread_policy.h>
#include <pthread.h>
#endif

namespace admiral {
namespace detail {

namespace {

// One std::cerr line per process, whichever failure arrives first.
std::atomic<bool> g_warned{false};

void note_disabled(const char* reason) noexcept {
    bool expected = false;
    if (g_warned.compare_exchange_strong(expected, true, std::memory_order_relaxed))
        std::cerr << "admiral: " << reason << '\n';
}

}  // namespace

const std::vector<std::size_t>* pool_pin_cpus(std::size_t nthreads) {
    static const std::vector<std::size_t> pins = [] {
        std::vector<std::size_t> out;
#if defined(__linux__)
        cpu_set_t aff;
        if (sched_getaffinity(0, sizeof aff, &aff) != 0) return out;
        std::vector<std::size_t> allowed;
        std::vector<pool_topo_row> topo;
        for (std::size_t c = 0; c < static_cast<std::size_t>(CPU_SETSIZE); ++c) {
            if (!CPU_ISSET(c, &aff)) continue;
            const std::string dir =
                "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
            // Unreadable core_id => untrustworthy table: refuse to pin (never clamp wrong).
            // A missing package id reads as one socket.
            pool_topo_row r{c, 0, 0};
            if (!(std::ifstream(dir + "core_id") >> r.core)) return out;
            std::ifstream(dir + "physical_package_id") >> r.socket;
            allowed.push_back(c);
            topo.push_back(r);
        }
        out = pool_pin_list(span<const std::size_t>(allowed.data(), allowed.size()),
                            span<const pool_topo_row>(topo.data(), topo.size()));
#endif
        return out;
    }();
    return pins.size() >= nthreads ? &pins : nullptr;
}

void pool_pin_worker([[maybe_unused]] std::size_t tid,
                     [[maybe_unused]] std::size_t nthreads,
                     [[maybe_unused]] bool pin) noexcept {
    if (!pin) return;
#if defined(__linux__)
    if (const std::vector<std::size_t>* pins = pool_pin_cpus(nthreads)) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET((*pins)[tid], &one);
        if (sched_setaffinity(0, sizeof one, &one) != 0)
            note_disabled("pinpool: DISABLED sched_setaffinity failed");
    } else {
        // The width-0 list prints the raw count (0 = unreadable topology).
        char why[96];
        std::snprintf(why, sizeof why, "pinpool: DISABLED mask holds %zu cores, nt=%zu",
                      pool_pin_cpus(0)->size(), nthreads);
        note_disabled(why);
    }
#elif defined(__APPLE__)
    // Hw threads are not pinnable on macOS; a distinct tag per worker is the keep-apart hint.
    thread_affinity_policy_data_t tag = {static_cast<integer_t>(tid + 1)};
    thread_policy_set(pthread_mach_thread_np(pthread_self()), THREAD_AFFINITY_POLICY,
                      reinterpret_cast<thread_policy_t>(&tag), THREAD_AFFINITY_POLICY_COUNT);
#else
    if (tid == 0) note_disabled("pinpool: unsupported on this OS");
#endif
}

bool pool_pin_disabled_noted() noexcept { return g_warned.load(std::memory_order_relaxed); }

}  // namespace detail
}  // namespace admiral

#endif  // ADM_THREADS
