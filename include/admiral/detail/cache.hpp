#pragma once

// ============================================================================
// Cache geometry, probed once from the OS. Sizes cache tiles and residency
// decisions only; SIMD width is compile-time via -march.
// ============================================================================

#include <cstddef>
#if defined(__APPLE__)
#include <cstdint>
#endif

#if defined(__linux__)
#include <unistd.h>            // sysconf(_SC_LEVEL*_CACHE_SIZE)
#include <cstdio>              // fscanf (L3 shared_cpu_list)
#elif defined(__APPLE__)
#include <sys/sysctl.h>        // sysctlbyname("hw.l*cachesize")
#endif

namespace admiral {
namespace detail {

// Coherence-line bytes on every ISA targeted; the library's one documented non-arch
// constant. Not std::hardware_destructive_interference_size (absent under clang++) and
// not the arch alignment() (a register width would put four atomics on one line).
inline constexpr std::size_t kCacheLine = 64;

// L1d tile bytes: the in-place pass's phase-A store / phase-B reload block limit,
// and the same bound the routing model prices that pass against. A fixed 32 KiB
// rather than the probed L1 because the pass is written around this tile width.
inline constexpr std::size_t kIpTileBytes = 32u * 1024u;

// L1d tile bytes for the fused multi-pass bodies. Each divides it by the number of
// planes it keeps live at once, so the sum stays inside one tile whatever the fusion
// depth. Independent of kIpTileBytes above: a fused body tiles planes, not blocks.
inline constexpr std::size_t kFusedTileBytes = 16u * 1024u;

// Cache sizes (bytes), probed once (function-local static). sysconf on Linux,
// sysctl on macOS/BSD; otherwise uses the fallbacks below. l3_cores counts the
// CPUs sharing the L3 that `l3` measures. A caller divides a threaded per-worker
// footprint by that count, not by a machine-wide thread count.
// 0 = "unknown" = divide by nthreads verbatim.
// Used when the OS reports nothing: a mid-range workstation's L2 and LLC.
inline constexpr std::size_t kFallbackL2Bytes = std::size_t{2} << 20;
inline constexpr std::size_t kFallbackL3Bytes = std::size_t{45} << 20;

struct cache_bytes { std::size_t l2, l3, l3_cores; };
[[nodiscard]] inline const cache_bytes& cpu_cache() {
    static const cache_bytes c = [] {
        cache_bytes d{kFallbackL2Bytes, kFallbackL3Bytes, 0};
#if defined(__linux__)
        const auto pos = [](long v, std::size_t fb) { return v > 0 ? std::size_t(v) : fb; };
        d.l2  = pos(::sysconf(_SC_LEVEL2_CACHE_SIZE),  d.l2);
        d.l3  = pos(::sysconf(_SC_LEVEL3_CACHE_SIZE),  d.l3);
        d.l3_cores = [] {  // e.g. "0-3" or "0-3,8-11"; 0 means "unread -> unknown"
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
#endif
        return d;
    }();
    return c;
}

}  // namespace detail
}  // namespace admiral
