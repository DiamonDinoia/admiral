// The constexpr dd root tables must equal the run-time recurrence build.
// The check is absolute on each full dd value (hi + lo, real and imaginary).

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <vector>

#include "admiral/detail/dd_fft.hpp"

namespace {

namespace d = admiral::detail;
using cv = d::ct_sincos_v<d::ct_dd_plain>;

cv cmul(const cv& a, const cv& b) { return {a.s * b.c + a.c * b.s, a.c * b.c - a.s * b.s}; }

// Run-time build: fine and coarse powers by recurrence, as in dd_roots.
struct runtime_roots {
    std::size_t n, b;
    std::vector<cv> coarse, fine;
    explicit runtime_roots(std::size_t n_) : n(n_), b(d::dd_root_block(n_)) {
        coarse.assign((n + b - 1) / b, cv{d::ct_dd_plain(0), d::ct_dd_plain(1)});
        fine.assign(b, coarse[0]);
        const cv f1 = d::sincos_turns_eval<d::ct_dd_plain>(true, 1, n);
        const cv c1 = d::sincos_turns_eval<d::ct_dd_plain>(true, b, n);
        if (b > 1) fine[1] = f1;
        for (std::size_t e = 2; e < b; ++e) fine[e] = cmul(fine[e - 1], f1);
        if (coarse.size() > 1) coarse[1] = c1;
        for (std::size_t e = 2; e < coarse.size(); ++e) coarse[e] = cmul(coarse[e - 1], c1);
    }
    d::dd_const operator()(std::size_t e) const {
        e %= n;
        return d::dd_split(cmul(coarse[e / b], fine[e % b]));
    }
};

double dd_err(double ah, double al, double bh, double bl) { return std::fabs((ah - bh) + (al - bl)); }

// Largest absolute error over all entries; bump is added to one reference lo word.
double max_err(std::size_t n, double bump) {
    const d::dd_roots ct(n);
    const runtime_roots rt(n);
    double worst = 0.0;
    for (std::size_t e = 0; e < n; ++e) {
        const d::dd_const c = ct(e);
        d::dd_const r = rt(e);
        if (e == 1) r.rl += bump;
        worst = std::fmax(worst, std::fmax(dd_err(c.rh, c.rl, r.rh, r.rl), dd_err(c.ih, c.il, r.ih, r.il)));
    }
    return worst;
}

constexpr std::size_t catalog[] = {9, 18, 30, 32, 50, 60, 64, 100, 128, 256};

}  // namespace

TEST_CASE("dd_roots: compile-time tables equal run-time build") {
    for (const std::size_t n : catalog) {
        CAPTURE(n);
        REQUIRE(max_err(n, 0.0) <= 1e-30);
    }
}

TEST_CASE("dd_roots: a perturbed entry fails the same comparison") {
    for (const std::size_t n : catalog) {
        CAPTURE(n);
        REQUIRE(max_err(n, 1e-29) > 1e-30);
    }
}
