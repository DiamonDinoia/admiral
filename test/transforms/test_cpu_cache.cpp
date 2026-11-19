#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/cache.hpp>

#include <cstddef>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <cstdint>

namespace {
std::size_t sysctl_u64_or_zero(const char* key) {
    std::uint64_t v = 0; std::size_t n = sizeof(v);
    return (::sysctlbyname(key, &v, &n, nullptr, 0) == 0) ? std::size_t(v) : std::size_t{0};
}
std::size_t cachesize_slot(std::size_t idx) {
    std::uint64_t cs[8] = {}; std::size_t n = sizeof(cs);
    return (::sysctlbyname("hw.cachesize", cs, &n, nullptr, 0) == 0 && idx < 8)
               ? std::size_t(cs[idx]) : std::size_t{0};
}
// The probe's invariant: a dedicated key's nonzero value wins; otherwise the hw.cachesize
// aggregate slot answers; otherwise the fallback (or, for l3, zero).
std::size_t probe_expected(const char* key, std::size_t idx) {
    const std::size_t v = sysctl_u64_or_zero(key);
    return v ? v : cachesize_slot(idx);
}
}  // namespace
#endif

TEST_CASE("cpu_cache: every populated field is nonzero") {
    const admiral::detail::cache_bytes& cc = admiral::detail::cpu_cache();
    CHECK(cc.l1d > 0);
    CHECK(cc.l2 > 0);
#if defined(__linux__)
    CHECK(cc.l3 > 0);
    CHECK(cc.l3_cores > 0);
    CHECK(cc.l3_phys_cores > 0);
    CHECK(cc.l3_phys_cores <= cc.l3_cores);
#endif
}

#if defined(__APPLE__) && defined(__aarch64__)
TEST_CASE("cpu_cache: arm64 Darwin counts physical cores and never invents an L3 size") {
    const admiral::detail::cache_bytes& cc = admiral::detail::cpu_cache();
    CHECK(cc.l3_phys_cores > 0);
    // Darwin reports no logical/physical topology split; the aggregate count stays 0.
    CHECK(cc.l3_cores == 0);
    CHECK(cc.l3_phys_cores == sysctl_u64_or_zero("hw.physicalcpu"));
    CHECK(cc.l1d == probe_expected("hw.l1dcachesize", 1));
    CHECK(cc.l2  == probe_expected("hw.l2cachesize",  2));
    // Apple silicon routes the SLC through no sysctl: l3 must read 0 unless a key actually
    // reports one — never the 45 MiB x86 fallback.
    CHECK(cc.l3 == probe_expected("hw.l3cachesize", 3));
}
#endif
