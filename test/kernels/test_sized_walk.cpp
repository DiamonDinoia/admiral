// The sized-rung remainder walk (include/admiral/detail/codelet.hpp) must cover every lane of
// a codelet remainder whatever the sized-batch ladder's live pattern, and the sites keep the
// (rem & Wt) bit decomposition wherever the ladder is sound (no dead rung above a live one).
// No shipping target has such a hole (x86-64 keeps sse2..avx512 live above its scalar mask,
// NEON keeps its full ladder), so the hole cases run on synthetic rung policies below; the
// gate that proves the fix on a real holed target is the sve512 ctest leg under qemu.
// Mutation note: reverting the walk to the bit decomposition fails these checks at compile
// time (see rungs<true, false, true>, the sve512 f64 shape).

#include <catch2/catch_test_macros.hpp>

#include <admiral/detail/codelet.hpp>

#include <cstddef>

namespace {

// Rung policy over the 8/4/2 ladder with each rung independently switchable.
template<bool L8, bool L4, bool L2>
struct rungs {
    template<std::size_t Wt>
    [[nodiscard]] static constexpr bool live() {
        if constexpr (Wt == 8) return L8;
        else if constexpr (Wt == 4) return L4;
        else if constexpr (Wt == 2) return L2;
        else return Wt > 8;
    }
};

// Independent reference: an explicit per-rung walk that shares no machinery with the walk
// under test.
template<bool L8, bool L4, bool L2>
constexpr std::size_t ref_cover(std::size_t rem) {
    std::size_t j = 0;
    if constexpr (L8) while (j + 8 <= rem) j += 8;
    if constexpr (L4) while (j + 4 <= rem) j += 4;
    if constexpr (L2) while (j + 2 <= rem) j += 2;
    return j;
}

template<bool L8, bool L4, bool L2>
ADM_CONSTEVAL bool walk_matches_reference() {
    for (std::size_t rem = 0; rem <= 31; ++rem) {
        const std::size_t cov =
            admiral::detail::sized_walk_cover<rungs<L8, L4, L2>, 8>(rem);
        if (cov != ref_cover<L8, L4, L2>(rem)) return false;
        if (cov > rem) return false;
    }
    return true;
}

// The walk over the real xsimd ladder of this TU: never overruns a remainder. Ladder-sound
// targets keep the bit decomposition, so only the bound is assertable here for real types; the
// cover equalities all ride on the synthetic policies above.
template<typename T, std::size_t Top>
ADM_CONSTEVAL bool real_walk_ok() {
    for (std::size_t rem = 0; rem <= 2 * Top + 1; ++rem) {
        const std::size_t cov =
            admiral::detail::sized_walk_cover<admiral::detail::sized_rungs<T>, Top>(rem);
        if (cov > rem) return false;
    }
    return true;
}

}  // namespace

using admiral::detail::sized_ladder_sound;
using admiral::detail::sized_walk_cover;

// Every live/dead combination of the three rungs, every remainder 0..31. The mixed cases are
// the hole shapes the bit decomposition could not serve.
static_assert(walk_matches_reference<true,  true,  true>());
static_assert(walk_matches_reference<true,  true,  false>());
static_assert(walk_matches_reference<true,  false, true>());
static_assert(walk_matches_reference<true,  false, false>());
static_assert(walk_matches_reference<false, true,  true>());
static_assert(walk_matches_reference<false, true,  false>());
static_assert(walk_matches_reference<false, false, true>());
static_assert(walk_matches_reference<false, false, false>());

// The exact sve512 f64 failure shape: native batch 8, dead 4-rung (sve<256> is not in the
// TU's arch list), rem = 4 (N = 16: M = 4 below one full batch). The walk covers all 4 lanes
// through the live 2-rung; the bit decomposition it replaces covered 0 of 4.
static_assert(sized_walk_cover<rungs<true, false, true>, 4>(4) == 4);
static_assert(sized_walk_cover<rungs<true, false, true>, 4>(6) == 6);  // 3x on the 2-rung
static_assert(sized_walk_cover<rungs<true, false, true>, 4>(7) == 6);  // scalar mops the last
// Dead 2-rung: what the 4-rung leaves is scalar work, never dropped.
static_assert(sized_walk_cover<rungs<true, true, false>, 4>(6) == 4);
// The sve512 f32 shape: native batch 16, dead 8-rung below it.
static_assert(sized_walk_cover<rungs<false, true, true>, 8>(8) == 8);
static_assert(sized_walk_cover<rungs<false, true, true>, 8>(12) == 12);

// The soundness predicate that selects the arm at the two remainder sites: a dead rung may
// only sit below every live one (its bits fall into the scalar mask), never above one.
static_assert(sized_ladder_sound<rungs<true, true, true>, 8>());
static_assert(sized_ladder_sound<rungs<true, true, false>, 8>());   // x86-64 f32 shape
static_assert(!sized_ladder_sound<rungs<true, false, true>, 8>());  // sve512 f64 shape
static_assert(!sized_ladder_sound<rungs<false, true, true>, 8>());  // dead 8 above live 4
static_assert(sized_ladder_sound<rungs<true, false, false>, 8>());  // all-scalar tail
static_assert(sized_ladder_sound<rungs<false, false, false>, 1>());

// The real ladders of this TU, float and double, at their widest native tops.
static_assert(real_walk_ok<double, 8>());
static_assert(real_walk_ok<float, 16>());

TEST_CASE("sized-rung remainder walk covers holed ladders") {
    STATIC_REQUIRE(walk_matches_reference<false, true, true>());
    CHECK(sized_walk_cover<rungs<false, true, true>, 8>(12) == 12);
}
