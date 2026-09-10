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

struct cache_bytes { std::size_t l2, l3, l3_cores, l1d; };
[[nodiscard]] inline const cache_bytes& cpu_cache() {
    static const cache_bytes c = [] {
        cache_bytes d{kFallbackL2Bytes, kFallbackL3Bytes, 0, kFallbackL1DBytes};
#if defined(__linux__)
        const auto pos = [](long v, std::size_t fb) { return v > 0 ? std::size_t(v) : fb; };
        d.l2  = pos(::sysconf(_SC_LEVEL2_CACHE_SIZE),  d.l2);
        d.l3  = pos(::sysconf(_SC_LEVEL3_CACHE_SIZE),  d.l3);
        d.l1d = pos(::sysconf(_SC_LEVEL1_DCACHE_SIZE), d.l1d);
        d.l3_cores = [] {
            if (std::FILE* f = std::fopen("/sys/devices/system/cpu/cpu0/cache/index3/"
                                          "shared_cpu_list", "re")) {
                std::size_t n = 0; unsigned a, b; char sep = 0;
                while (std::fscanf(f, "%u%c", &a, &sep) >= 1) {
                    n += (sep == '-' && std::fscanf(f, "%u%c", &b, &sep) >= 1) ? b - a + 1 : 1;
                    if (sep != ',') break;
                }
                std::fclose(f);
                return n;
            }
            return std::size_t{0};
        }();
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
