// Test support, linked into every test executable. ADM_TEST_TIER=<tier tag> pins the engine
// tier that the public API reaches before main; unset leaves the library's own dispatch. The
// tags of this architecture are in src/tiers/<arch>.hpp (x86-64: v1, v3, v4; AArch64: neon,
// sve256; riscv64: rv64gc, rvv128, rvv256, rvv512; LoongArch: lsx, lasx). ADM_TEST_TIER_BIND
// exists only as the positive control of the tier-coverage zero rule: scripts/tier_coverage.py
// runs one binary with ADM_TEST_TIER=v1 ADM_TEST_TIER_BIND=v3, and the run must retire
// 256-bit ops, which proves a v1-labeled run detects wide instructions.

#ifdef ADM_DISPATCH

#include "api_tier.hpp"

#include "tiers/tier_map.hpp"

#include <xsimd/xsimd.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

namespace {

// available_architectures() lists only the archs the TU's flags compile, so at a floor build
// a tier arch can sit outside it (rvv<N> compiled at rv64gc): the has() probe does not even
// instantiate there, and it disables itself. The runner (SDE, qemu), not the host, picks the
// CPU in those legs; natively the probe still rejects a tier the host does not have.
template<class Arch, class = void>
struct probe_ok : std::false_type {};
template<class Arch>
struct probe_ok<
    Arch, std::void_t<decltype(xsimd::available_architectures().has(std::declval<Arch>()))>>
    : std::true_type {};

template<class Arch>
bool pin(const char* tier) {
    if constexpr (probe_ok<Arch>::value) {
        if (!xsimd::available_architectures().has(Arch{})) {
            std::fprintf(stderr, "ADM_TEST_TIER=%s: %s is not available on this host\n", tier,
                         Arch::name());
            return false;
        }
    }
    using namespace admiral::detail;
    tier_slot<float>() = tier_entry{}(Arch{}, static_cast<float*>(nullptr));
    tier_slot<double>() = tier_entry{}(Arch{}, static_cast<double*>(nullptr));
    tier_slot<long double>() = tier_entry{}(Arch{}, static_cast<long double*>(nullptr));
    std::fprintf(stderr, "ADM_TEST_TIER=%s: engine arch %s\n", tier, Arch::name());
    return true;
}

struct select_tier {
    select_tier() {
        const char* v = std::getenv("ADM_TEST_TIER");
        if (!v || !*v)
            return;
        // The coverage control: bind the named slot to this table instead.
        const char* bind = std::getenv("ADM_TEST_TIER_BIND");
        if (bind && *bind)
            v = bind;
        bool known = false;
        bool ok = false;
#define ADM_PIN(tag, Arch)                                                      \
    if (!known && !std::strcmp(v, tag)) {                                              \
        known = true;                                                                  \
        ok = pin<Arch>(v);                                                             \
    }
        ADM_TIER_TABLE(ADM_PIN)
#undef ADM_PIN
        if (!known) {
#define ADM_TAG(tag, Arch) " " tag
            std::fprintf(stderr, "ADM_TEST_TIER=%s: unknown, use one of:" ADM_TIER_TABLE(ADM_TAG)
                                 "\n",
                         v);
#undef ADM_TAG
        }
        if (!ok)
            std::exit(2);
    }
} const select_tier_at_startup;

}

#endif
