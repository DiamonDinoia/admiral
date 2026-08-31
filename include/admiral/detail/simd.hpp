#pragma once

// The one xsimd entry point: every admiral header includes `simd.hpp`, not
// `<xsimd/xsimd.hpp>`.

// Under `-ffast-math` on gcc, `xsimd_constants.hpp` wraps its table in
// `#pragma GCC optimize("signed-zeros")`. Every later declaration then carries
// its own optimization node, and gcc 16's inliner rejects an edge with mixed
// nodes. `reset_options` restores the command-line state every admiral TU
// compiles with.

// The reset is file-scope, wider than the defect needs: a caller who sets its
// own `#pragma GCC optimize` before including an admiral header loses that
// pragma.

#include <xsimd/xsimd.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC reset_options
#endif
