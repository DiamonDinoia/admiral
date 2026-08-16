#pragma once

// Debug tracing, driven by exec_options::debug. Level 0 is every shipping call, and
// all it costs the hot path is one compare: the printers are out of line, cold, and
// never inlined, so no call site can pull formatting into the fast path.
//
// Levels are cumulative: 1 is one line per execute (what ran), 2 adds the shape, 3 adds
// the cost-model ranking. The field is an integer so a new site can ask for `>= 4`
// without inventing another knob.
//
// stdio, not <iostream>: reachable from the engine, and <iostream> would add a static
// ios_base::Init to every TU that only wants a diagnostic that is off.

#include <cstdio>

#include "macros.hpp"

namespace admiral {
namespace detail {

enum : unsigned { dbg_off = 0, dbg_route = 1, dbg_shape = 2, dbg_cost = 3 };

// The value overloads exist so a call site can list mixed values without a format
// string; there is nothing to gain from more types than the engine actually prints.
inline void dbg_put(const char* s) { std::fputs(s, stderr); }
inline void dbg_put(std::size_t v) { std::fprintf(stderr, "%zu", v); }
inline void dbg_put(double v) { std::fprintf(stderr, "%g", v); }

// One line to stderr. Cold and noinline so the hot path keeps only the guard.
template<typename... Args>
ADM_NOINLINE ADM_COLD void dbg_print(const Args&... args) {
    std::fputs("[admiral] ", stderr);
    (dbg_put(args), ...);
    std::fputc('\n', stderr);
}

// `label=` then a space-separated list, e.g. "radices= 4 4 8".
template<typename R>
ADM_NOINLINE ADM_COLD void dbg_print_seq(const char* label, const R& seq) {
    std::fprintf(stderr, "[admiral] %s=", label);
    for (const auto& v : seq) std::fprintf(stderr, " %zu", static_cast<std::size_t>(v));
    std::fputc('\n', stderr);
}

} // namespace detail
} // namespace admiral

#include "undef_macros.hpp"
