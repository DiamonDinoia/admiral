#pragma once

// Debug tracing, driven by `exec_options::debug`. Level 0 costs the hot path
// one compare; the printers are out of line, cold and never inlined.
// Levels are cumulative: 1 is one line per execute, 2 adds the shape, 3 adds
// the cost-model ranking. The field is an integer, so a site can ask for >= 4.
// `stdio`, not `<iostream>`: `<iostream>` would add a static `ios_base::Init`
// to every TU whose diagnostics stay off.

#include <cstdio>

#include "macros.hpp"

namespace admiral {
namespace detail {

enum : unsigned { dbg_off = 0, dbg_route = 1, dbg_shape = 2, dbg_cost = 3 };

// Value overloads, so a call site passes mixed values without a format string.
inline void dbg_put(const char* s) { std::fputs(s, stderr); }
inline void dbg_put(std::size_t v) { std::fprintf(stderr, "%zu", v); }
inline void dbg_put(double v) { std::fprintf(stderr, "%g", v); }

// One line to `stderr`. Cold and noinline, so the hot path keeps only the guard.
template<typename... Args>
ADM_NOINLINE ADM_COLD void dbg_print(const Args&... args) {
    std::fputs("[admiral] ", stderr);
    (dbg_put(args), ...);
    std::fputc('\n', stderr);
}

// Prints `label=` then a space-separated list, as in "radices= 4 4 8".
template<typename R>
ADM_NOINLINE ADM_COLD void dbg_print_seq(const char* label, const R& seq) {
    std::fprintf(stderr, "[admiral] %s=", label);
    for (const auto& v : seq) std::fprintf(stderr, " %zu", static_cast<std::size_t>(v));
    std::fputc('\n', stderr);
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
