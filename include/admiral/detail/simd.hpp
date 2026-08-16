#pragma once

// The one xsimd entry point: every admiral header includes this, not <xsimd/xsimd.hpp>.

// Under -ffast-math on gcc, xsimd_constants.hpp wraps its table in
// `#pragma GCC optimize("signed-zeros")`. Every later declaration then carries its own
// optimization node, and gcc 16's inliner rejects an edge whose ends do not share one.

// `reset_options` restores the command-line state, which every admiral TU compiles with.

// ponytail: file-scope reset, wider than the defect needs. A caller who sets its own
// `#pragma GCC optimize` before including an admiral header loses it. Narrow this once
// xsimd carries the attribute on its constant functions instead of a file-scope pragma.

#include <xsimd/xsimd.hpp>

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC reset_options
#endif
