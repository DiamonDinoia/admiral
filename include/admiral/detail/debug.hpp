#pragma once

#include <cstdio>

#include "macros.hpp"

namespace admiral {
namespace detail {

enum : unsigned { dbg_off = 0, dbg_route = 1, dbg_shape = 2, dbg_cost = 3 };

inline void dbg_put(const char* s) { std::fputs(s, stderr); }
inline void dbg_put(std::size_t v) { std::fprintf(stderr, "%zu", v); }
inline void dbg_put(double v) { std::fprintf(stderr, "%g", v); }

template<typename... Args>
ADM_NOINLINE ADM_COLD void dbg_print(const Args&... args) {
    std::fputs("[admiral] ", stderr);
    (dbg_put(args), ...);
    std::fputc('\n', stderr);
}

template<typename R>
ADM_NOINLINE ADM_COLD void dbg_print_seq(const char* label, const R& seq) {
    std::fprintf(stderr, "[admiral] %s=", label);
    for (const auto& v : seq) std::fprintf(stderr, " %zu", static_cast<std::size_t>(v));
    std::fputc('\n', stderr);
}

}
}

#include "undef_macros.hpp"
