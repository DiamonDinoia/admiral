#pragma once

#include <cstddef>
#if defined(__APPLE__)
#include <cstdint>
#endif

#if defined(__linux__)
#include <unistd.h>
#include <cstdio>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace admiral {
namespace detail {

inline constexpr std::size_t kCacheLine = 64;

inline constexpr std::size_t kIpTileBytes = 32u * 1024u;

inline constexpr std::size_t kFusedTileBytes = 16u * 1024u;

inline constexpr std::size_t kFallbackL2Bytes = std::size_t{2} << 20;
inline constexpr std::size_t kFallbackL3Bytes = std::size_t{45} << 20;
inline constexpr std::size_t kFallbackL1DBytes = std::size_t{32} << 10;

#if defined(__linux__)
// Visit every cpu id in a sysfs cpu list ("0-15,32-47").
template<typename F>
inline void for_each_sysfs_cpu(const char* path, F&& visit) {
    std::FILE* f = std::fopen(path, "re");
    if (!f) return;
    unsigned a = 0, b = 0;
    char sep = 0;
    while (std::fscanf(f, "%u%c", &a, &sep) >= 1) {
        if (sep == '-' && std::fscanf(f, "%u%c", &b, &sep) >= 1)
            for (unsigned c = a; c <= b; ++c) visit(c);
        else
            visit(a);
        if (sep != ',') break;
    }
    std::fclose(f);
}

// True when `cpu` is the first hardware thread of its physical core. Counting only first siblings
// turns a list of logical cpus into a count of physical cores, on SMT and hybrid parts alike.
[[nodiscard]] inline bool is_first_thread_of_core(unsigned cpu) {
    char path[96];
    std::snprintf(path, sizeof path,
                  "/sys/devices/system/cpu/cpu%u/topology/thread_siblings_list", cpu);
    std::FILE* f = std::fopen(path, "re");
    if (!f) return true;
    unsigned first = cpu;
    const bool ok = std::fscanf(f, "%u", &first) == 1;
    std::fclose(f);
    return !ok || first == cpu;
}
#endif

// `l3_cores` counts logical cpus sharing the L3 and answers contention questions. `l3_phys_cores`
// counts physical cores over the same domain and answers capacity questions. The two differ only
// under SMT, and a capacity constant fitted on SMT-off hosts needs the second one.
struct cache_bytes { std::size_t l2, l3, l3_cores, l3_phys_cores, l1d; };
[[nodiscard]] inline const cache_bytes& cpu_cache() {
    static const cache_bytes c = [] {
        cache_bytes d{kFallbackL2Bytes, kFallbackL3Bytes, 0, 0, kFallbackL1DBytes};
#if defined(__linux__)
        const auto pos = [](long v, std::size_t fb) { return v > 0 ? std::size_t(v) : fb; };
        d.l2  = pos(::sysconf(_SC_LEVEL2_CACHE_SIZE),  d.l2);
        d.l3  = pos(::sysconf(_SC_LEVEL3_CACHE_SIZE),  d.l3);
        d.l1d = pos(::sysconf(_SC_LEVEL1_DCACHE_SIZE), d.l1d);
        for_each_sysfs_cpu("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list",
                           [&d](unsigned cpu) {
                               ++d.l3_cores;
                               d.l3_phys_cores += is_first_thread_of_core(cpu) ? 1u : 0u;
                           });
#elif defined(__APPLE__)
        const auto rd = [](const char* key, std::size_t fb) {
            std::uint64_t v = 0; std::size_t n = sizeof(v);
            return ::sysctlbyname(key, &v, &n, nullptr, 0) == 0 && v ? std::size_t(v) : fb;
        };
        d.l2  = rd("hw.l2cachesize",  d.l2);
        d.l3  = rd("hw.l3cachesize",  d.l3);
        d.l1d = rd("hw.l1dcachesize", d.l1d);
#endif
        return d;
    }();
    return c;
}

}
}
