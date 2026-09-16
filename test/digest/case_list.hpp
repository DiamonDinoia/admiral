#pragma once

// Digest case list: every case the digest driver runs, in emission order (f32 block, then
// the identical f64 block), plus the constants the driver's self-checks enforce.
//
// STABILITY RULE. Case ids are POSITIONAL: a running counter assigns them in emission
// order, and the committed goldens in test/digest/golden plus the --ulp 700 positive
// control key on them. Numbering is append-stable BY CONSTRUCTION: new cases append
// after the f64 block ONLY (for the f32 family that means a new block after f64, since
// f64 follows f32 in emission order). A mid-list insert silently renumbers every later
// case: the control's target moves and every golden line after the insert desyncs.
// Three guards make a renumber loud instead of silent:
//   1. kCaseCount below is asserted against the computed case count at build time and
//      re-checked against the emitted count at run time;
//   2. the driver fails unless case kUlpControlCase is still named kUlpControlName;
//   3. check_digest.sh treats any id whose tag/name drifted as a structural error.
// Growth budget: at most 5% of the list length per landing (<= 67 cases at 1344).
// A catalog-structural landing may exceed it only with a `receipt:` trailer.
// See test/digest/README.md.

#include <cstddef>

namespace digest_cases {

// 1-D plan sweep: every size in [kPlan1dFirst, kPlan1dLast], forward then inverse per
// size. kPlan1dLast only ever INCREASES (that appends at the 1-D tail); kPlan1dFirst
// never moves.
inline constexpr std::size_t kPlan1dFirst = 2;
inline constexpr std::size_t kPlan1dLast = 320;

struct NdShape {
    std::size_t dim[3];
    std::size_t ndim;
};

// N-D plan shapes, in emission order. APPEND AT THE END ONLY.
inline constexpr NdShape kNdShapes[] = {
    {{16, 16, 0}, 2}, {{32, 32, 0}, 2}, {{20, 20, 0}, 2}, {{60, 60, 0}, 2}, {{81, 81, 0}, 2},
    {{96, 96, 0}, 2}, {{64, 64, 0}, 2}, {{12, 20, 0}, 2}, {{192, 64, 0}, 2},
    {{8, 8, 8}, 3},   {{16, 16, 16}, 3}, {{12, 10, 9}, 3},  {{64, 4, 4}, 3},
};

struct StridesCase {
    std::size_t len, batch, istride, ostride, ibatch, obatch;
};

// Strides cases, in emission order. APPEND AT THE END ONLY.
inline constexpr StridesCase kStridesCases[] = {
    {64, 2, 1, 1, 64, 64},
    {64, 2, 1, 8192, 64, 1},
    {60, 3, 4, 4, 1, 1},
    {81, 2, 1, 2, 81, 162},
};

// Cube cases, one f32 block then one f64 block, emitted after BOTH main precision blocks
// by the driver's trailing sweep (an append inside kNdShapes/kStridesCases would shift
// every later id, and numbering is global across precisions). APPEND AT THE END ONLY.
inline constexpr NdShape kCubeShapes[] = {
    {{4, 4, 4}, 3},
};

inline constexpr std::size_t kCasesPerPrecision =
    2 * (kPlan1dLast - kPlan1dFirst + 1) + 2 * (sizeof(kNdShapes) / sizeof(kNdShapes[0])) +
    2 * (sizeof(kStridesCases) / sizeof(kStridesCases[0])) +
    2 * (sizeof(kCubeShapes) / sizeof(kCubeShapes[0]));

// Bump on every append; the static_assert pins it to the computed total, so a case-list
// edit that forgets it fails this target's build. The goldens must move in the same
// commit (test/digest/README.md).
inline constexpr std::size_t kCaseCount = 1348;
static_assert(kCaseCount == 2 * kCasesPerPrecision,
              "case-list edit without a kCaseCount bump: append cases after the f64 block "
              "only, update kCaseCount and regenerate test/digest/golden in the same commit");

// --ulp positive control: perturbing this case's first output element by one ulp must
// move exactly this case's digest line and no other.
inline constexpr std::size_t kUlpControlCase = 700;
inline constexpr char kUlpControlName[] = "f64 16xF";
static_assert(kUlpControlCase < kCaseCount, "control case id outside the case list");

} // namespace digest_cases
