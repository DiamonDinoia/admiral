// The run-time tier pick, without the test slot: compares the tier ops table that the
// library's own xsimd::dispatch resolved against the table of the tier ADM_EXPECT_TIER names,
// for float, double and long double. Name a tier by its tag from src/tiers/<arch>.hpp
// (x86-64: v1, v3, v4; AArch64: neon, sve256; riscv64: rv64gc, rvv128, rvv256, rvv512;
// LoongArch: lsx, lasx); an xsimd name() also matches where it is unambiguous (the SDE legs
// of CI: sse2, fma3+avx2, avx512bw). Runs natively, under the SDE models (-p4p names sse2,
// -hsw fma3+avx2, -spr avx512bw) and under qemu. Exit 1 on a mismatch, 2 when ADM_EXPECT_TIER
// is unset or unknown.

#include "api_tier.hpp"

#include "tiers/tier_map.hpp"

#include <xsimd/xsimd.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

template<class Arch>
int expect(const char* want) {
    using namespace admiral::detail;
    const int g =
        (tier_slot<float>() == tier_entry{}(Arch{}, static_cast<float*>(nullptr)) ? 0 : 1) +
        (tier_slot<double>() == tier_entry{}(Arch{}, static_cast<double*>(nullptr)) ? 0 : 1) +
        (tier_slot<long double>() == tier_entry{}(Arch{}, static_cast<long double*>(nullptr)) ? 0
                                                                                              : 1);
    std::printf("ADM_EXPECT_TIER=%s: engine arch %s, mismatched tables %d of 3\n", want,
                Arch::name(), g);
    return g == 0 ? 0 : 1;
}

// Names the tier of ADM_TIER_TABLE that the library picked for double.
void print_pick() {
    using namespace admiral::detail;
#define ADM_PICK(tag, Arch)                                                     \
    if (tier_slot<double>() == tier_entry{}(Arch{}, static_cast<double*>(nullptr)))    \
        std::printf("picked tier: %s (%s)\n", tag, Arch::name());
    ADM_TIER_TABLE(ADM_PICK)
#undef ADM_PICK
}

}

int main() {
    print_pick();
    const char* v = std::getenv("ADM_EXPECT_TIER");
    if (!v || !*v) {
        std::fprintf(stderr, "ADM_EXPECT_TIER unset: name a tier tag of src/tiers/<arch>.hpp\n");
        return 2;
    }
    // A tag is the canonical spelling (the SVE and RVV tiers of one width set share one xsimd
    // name()); an unambiguous name() matches too, for CI's SDE legs.
#define ADM_EXPECT(tag, Arch)                                                   \
    if (!std::strcmp(v, tag) || !std::strcmp(v, Arch::name()))                         \
        return expect<Arch>(v);
    ADM_TIER_TABLE(ADM_EXPECT)
#undef ADM_EXPECT
    std::fprintf(stderr, "ADM_EXPECT_TIER=%s: unknown arch\n", v);
    return 2;
}
