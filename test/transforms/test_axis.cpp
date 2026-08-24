#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>   // axis_plan, plan

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <functional>
#include <limits>
#include <span>
#include <vector>

// axis_plan<T> transforms one full axis of a row-major tensor, restricted to a box.
// Oracle: the verified 1D admiral::plan<T> per line; elements outside the box must
// stay untouched.

namespace {

// Suffix-product strides (stride[d] = product of extents inner to d).
std::vector<std::size_t> strides(const std::vector<std::size_t>& shape) {
    std::vector<std::size_t> s(shape.size());
    std::size_t acc = 1;
    for (std::size_t di = 0; di < shape.size(); ++di) {
        const std::size_t d = shape.size() - 1 - di;
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}

// Recurse over every non-axis dim in [lo,hi); invoke f(base_offset) per line.
template<typename F>
void for_each_line(const std::vector<std::size_t>& shape, const std::vector<std::size_t>& st,
                   const std::vector<std::size_t>& lo, const std::vector<std::size_t>& hi,
                   std::size_t axis, std::size_t d, std::size_t base, F&& f) {
    if (d == shape.size()) { f(base); return; }
    if (d == axis) { for_each_line(shape, st, lo, hi, axis, d + 1, base, f); return; }
    for (std::size_t i = lo[d]; i < hi[d]; ++i)
        for_each_line(shape, st, lo, hi, axis, d + 1, base + i * st[d], std::forward<F>(f));
}

template<typename T>
void check(const std::vector<std::size_t>& shape, std::size_t axis,
           std::vector<std::size_t> lo, std::vector<std::size_t> hi, bool forward) {
    const std::size_t ndim = shape.size();
    const std::size_t len = shape[axis];
    const std::size_t total = shape_product(shape);
    const auto st = strides(shape);
    if (lo.empty()) lo.assign(ndim, 0);
    if (hi.empty()) hi = shape;

    const auto orig = make_input<T>(total, 0xA11CE);
    auto data = orig;

    const admiral::axis_plan<T> ap(std::span<const std::size_t>(shape), axis, forward);
    ap.execute(data.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));

    // Reference: 1D plan per line (same direction, default scale).
    const admiral::plan<T> ref(std::array<std::size_t, 1>{len});
    std::vector<char> in_box(total, 0);
    for_each_line(shape, st, lo, hi, axis, 0, 0, [&](std::size_t base) {
        std::vector<std::complex<T>> line(len), got(len);
        for (std::size_t p = 0; p < len; ++p) line[p] = orig[base + p * st[axis]];
        if (forward) ref.forward(std::span<std::complex<T>>(line));
        else         ref.inverse(std::span<std::complex<T>>(line));
        for (std::size_t p = 0; p < len; ++p) {
            const std::size_t idx = base + p * st[axis];
            in_box[idx] = 1;
            got[p] = data[idx];
        }
        require_close(got, line, fft_tol<T>());
    });
    // Everything outside the box is byte-untouched.
    for (std::size_t i = 0; i < total; ++i)
        if (!in_box[i]) REQUIRE(data[i] == orig[i]);
}

// execute_bands must be indistinguishable from the two execute() calls it replaces,
// packed path or not.
template<typename T>
void check_bands(const std::vector<std::size_t>& shape, std::size_t axis,
                 std::vector<std::size_t> lo, std::vector<std::size_t> hi, std::size_t lo2_last,
                 std::size_t hi2_last, bool forward) {
    const std::size_t ndim = shape.size();
    const std::size_t total = shape_product(shape);
    if (lo.empty()) lo.assign(ndim, 0);
    if (hi.empty()) hi = shape;

    const auto orig = make_input<T>(total, 0xBA11D);
    auto packed = orig, split = orig;

    const admiral::axis_plan<T> ap(std::span<const std::size_t>(shape), axis, forward);
    ap.execute_bands(packed.data(), std::span<const std::size_t>(lo),
                     std::span<const std::size_t>(hi), lo2_last, hi2_last);

    auto lo2 = lo, hi2 = hi;
    lo2[ndim - 1] = lo2_last;
    hi2[ndim - 1] = hi2_last;
    ap.execute(split.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));
    ap.execute(split.data(), std::span<const std::size_t>(lo2), std::span<const std::size_t>(hi2));

    require_close(packed, split, fft_tol<T>());
}

}  // namespace

TEMPLATE_TEST_CASE("axis_plan matches per-line 1D plan", "[axis]", float, double) {
    using T = TestType;
    // innermost axis (contiguous row path)
    check<T>({6, 8}, 1, {}, {}, true);
    check<T>({6, 8}, 1, {2, 0}, {5, 8}, true);          // band the outer dim
    // outer smooth axis (SIMD-DIF column path)
    check<T>({8, 6}, 0, {}, {}, true);
    check<T>({8, 6}, 0, {0, 1}, {8, 5}, false);         // band the inner dim
    // 3-D: middle axis, both other dims banded
    check<T>({4, 5, 12}, 1, {1, 0, 3}, {4, 5, 10}, true);
    check<T>({4, 5, 12}, 1, {1, 0, 3}, {4, 5, 10}, false);
    // outer non-smooth axis (scalar gather path)
    check<T>({101, 4}, 0, {0, 1}, {101, 3}, true);
    check<T>({101, 4}, 0, {}, {}, false);
    // 3-D innermost axis with every batch dim whole: the dense-run shortcut.
    check<T>({3, 4, 8}, 2, {}, {}, true);
    check<T>({3, 4, 8}, 2, {1, 0, 0}, {3, 4, 8}, true);   // slowest dim banded, still dense
    check<T>({3, 4, 8}, 2, {0, 1, 0}, {3, 3, 8}, true);   // inner dim banded, not dense
}

TEMPLATE_TEST_CASE("axis_plan execute_bands matches two execute calls", "[axis]", float, double) {
    using T = TestType;
    // Packed path: len 512 is 5 radix-4/2 passes for float, so the pack gate opens;
    // for double it is 2 passes and the two-run fallback runs instead. The two
    // instantiations exercise both arms of the gate.
    check_bands<T>({512, 12}, 0, {0, 0}, {512, 3}, 9, 12, true);
    check_bands<T>({512, 12}, 0, {0, 0}, {512, 3}, 9, 12, false);
    // 6+6: within one batch for float, wider than one for double.
    check_bands<T>({512, 14}, 0, {0, 0}, {512, 6}, 8, 14, true);
    // 3-D with a banded outer dim, so the packed path walks several lines.
    check_bands<T>({4, 512, 10}, 1, {1, 0, 0}, {4, 512, 3}, 7, 10, true);
    // Short axis: too few passes to pay for the gather, always two runs.
    check_bands<T>({16, 12}, 0, {0, 0}, {16, 3}, 9, 12, true);
    // Non-smooth length: scalar gather path, packing declines.
    check_bands<T>({101, 10}, 0, {0, 0}, {101, 3}, 7, 10, true);
    // lo2 == hi2 is plain execute().
    check_bands<T>({512, 12}, 0, {0, 0}, {512, 3}, 0, 0, true);
    // Unequal widths on the scalar-gather path decline both packing and merging, so
    // the two bands run as two separate strided passes.
    check_bands<T>({101, 12}, 0, {0, 0}, {101, 2}, 8, 12, true);
    check_bands<T>({101, 12}, 0, {0, 0}, {101, 2}, 8, 12, false);
}

TEMPLATE_TEST_CASE("axis_plan forward/inverse round-trips on the box", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{4, 5, 12};
    const std::vector<std::size_t> lo{1, 0, 3}, hi{4, 5, 10};
    const std::size_t axis = 1;
    const std::size_t total = 4 * 5 * 12;
    const auto st = strides(shape);
    const auto orig = make_input<T>(total, 0xBEEF);
    auto data = orig;

    admiral::axis_plan<T> fwd(std::span<const std::size_t>(shape), axis, true);
    admiral::axis_plan<T> inv(std::span<const std::size_t>(shape), axis, false);
    fwd.execute(data.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));
    inv.execute(data.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));

    require_close(data, orig, fft_tol<T>());
}

TEMPLATE_TEST_CASE("axis_plan multithreaded matches serial", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{32, 64, 64};   // big enough to trip the MT gate
    const std::vector<std::size_t> lo{4, 0, 8}, hi{30, 64, 60};
    const std::size_t axis = 1;
    const std::size_t total = 32 * 64 * 64;
    const auto orig = make_input<T>(total, 0xF00D);

    auto ser = orig, par = orig;
    admiral::axis_plan<T> s(std::span<const std::size_t>(shape), axis, true);
    admiral::axis_plan<T> p(std::span<const std::size_t>(shape), axis, true, {.nthreads = 4});
    s.execute(ser.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));
    p.execute(par.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));

    require_close(par, ser, fft_tol<T>());
}

TEMPLATE_TEST_CASE("axis_plan rejects malformed shapes and boxes", "[axis]", float, double) {
    using T = TestType;
    using admiral::axis_plan;
    const auto sp = [](const std::vector<std::size_t>& v) {
        return std::span<const std::size_t>(v);
    };
    const std::vector<std::size_t> none{};

    // --- construction
    const std::vector<std::size_t> empty_shape{};
    REQUIRE_THROWS_AS(axis_plan<T>(sp(empty_shape), 0, true), admiral::size_error);
    REQUIRE_THROWS_AS(axis_plan<T>({6, 8}, 2, true), admiral::size_error);   // axis == rank
    REQUIRE_THROWS_AS(axis_plan<T>({6, 0}, 0, true), admiral::size_error);   // zero extent
    // Product overflows size_t, so it cannot be a valid tensor. Throws before allocating.
    REQUIRE_THROWS_AS(axis_plan<T>({std::numeric_limits<std::size_t>::max(), 2}, 0, true),
                      admiral::size_error);

    // --- box validation, on a plan over {6, 8} transforming the innermost axis
    const std::vector<std::size_t> shape{6, 8};
    std::vector<std::complex<T>> data(48);
    axis_plan<T> ap(sp(shape), 1, true);

    const std::vector<std::size_t> short_lo{0}, ok_lo{0, 0}, ok_hi{6, 8};
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(short_lo), sp(ok_hi)), admiral::size_error);
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(ok_lo), sp(short_lo)), admiral::size_error);

    const std::vector<std::size_t> past_end{6, 9}, inverted_lo{4, 0}, inverted_hi{2, 8};
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(ok_lo), sp(past_end)), admiral::size_error);
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(inverted_lo), sp(inverted_hi)),
                      admiral::size_error);

    const std::vector<std::size_t> part_lo{0, 1}, part_hi{6, 7};
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(ok_lo), sp(part_hi)), admiral::size_error);
    REQUIRE_THROWS_AS(ap.execute(data.data(), sp(part_lo), sp(ok_hi)), admiral::size_error);

    // --- second band. ap transforms the last dim, so it can never carry one.
    REQUIRE_THROWS_AS(ap.execute_bands(data.data(), none, none, 0, 1), admiral::size_error);

    // A plan on the outer axis can: the last dim is free to be banded.
    axis_plan<T> col(sp(shape), 0, true);
    const std::vector<std::size_t> b_lo{0, 0}, b_hi{6, 3};
    REQUIRE_THROWS_AS(col.execute_bands(data.data(), sp(b_lo), sp(b_hi), 5, 9),
                      admiral::size_error);   // hi2 past the extent
    REQUIRE_THROWS_AS(col.execute_bands(data.data(), sp(b_lo), sp(b_hi), 7, 5),
                      admiral::size_error);   // inverted
    REQUIRE_THROWS_AS(col.execute_bands(data.data(), sp(b_lo), sp(b_hi), 2, 6),
                      admiral::size_error);   // overlaps [0,3)
    const std::vector<std::size_t> mt_lo{3, 0}, mt_hi{3, 3};
    REQUIRE_THROWS_AS(col.execute_bands(data.data(), sp(mt_lo), sp(mt_hi), 5, 8),
                      admiral::size_error);   // empty box cannot carry a band
}

TEMPLATE_TEST_CASE("axis_plan survives a move", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{6, 8};
    const auto orig = make_input<T>(48, 7);
    auto want = orig, got = orig, got2 = orig;
    const auto full = std::span<const std::size_t>{};

    admiral::axis_plan<T> ref(std::span<const std::size_t>(shape), 1, true);
    ref.execute(want.data(), full, full);

    admiral::axis_plan<T> src(std::span<const std::size_t>(shape), 1, true);
    admiral::axis_plan<T> moved(std::move(src));
    moved.execute(got.data(), full, full);
    REQUIRE(got == want);

    admiral::axis_plan<T> dst(std::span<const std::size_t>(shape), 0, true);
    dst = std::move(moved);
    dst.execute(got2.data(), full, full);
    REQUIRE(got2 == want);
}

TEMPLATE_TEST_CASE("axis_plan empty box is a no-op", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{6, 8};
    const auto orig = make_input<T>(48, 1);
    auto data = orig;
    admiral::axis_plan<T> ap(std::span<const std::size_t>(shape), 1, true);
    const std::vector<std::size_t> lo{3, 0}, hi{3, 8};   // empty on dim 0
    ap.execute(data.data(), std::span<const std::size_t>(lo), std::span<const std::size_t>(hi));
    REQUIRE(data == orig);
}

TEMPLATE_TEST_CASE("axis_plan on an extent-1 axis is a no-op", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{4, 1};
    auto data = make_input<T>(4, 0xA11CE);
    const auto orig = data;
    admiral::axis_plan<T> p(std::span<const std::size_t>(shape), 1, true);
    const auto full = std::span<const std::size_t>{};
    p.execute(data.data(), full, full);
    REQUIRE(data == orig);  // extent 1: nothing to transform, elements untouched
}

TEMPLATE_TEST_CASE("axis_plan multithreaded on a single batch unit", "[axis]", float,
                   double) {
    using T = TestType;
    // units == prod(shape[d!=axis]) == 1 < 2, so the batch loop cannot thread and
    // the real thread count reaches the line plan (cpp_api.hpp's axis_threads
    // false arm). Threaded-vs-serial must still agree.
    const std::vector<std::size_t> shape{65536, 1};
    const auto orig = make_input<T>(65536, 0xB00C);
    auto ser = orig, par = orig;
    admiral::axis_plan<T> s(std::span<const std::size_t>(shape), 0, true);
    admiral::axis_plan<T> p(std::span<const std::size_t>(shape), 0, true, {.nthreads = 4});
    const auto full = std::span<const std::size_t>{};
    s.execute(ser.data(), full, full);
    p.execute(par.data(), full, full);
    require_close(par, ser, fft_tol<T>());
}

TEMPLATE_TEST_CASE("axis_plan execute with an explicit fct", "[axis]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape{16, 32};
    const auto orig = make_input<T>(shape[0] * shape[1], 0xC7);
    auto plain = orig, scaled = orig;
    admiral::axis_plan<T> p(std::span<const std::size_t>(shape), 0, true);
    const auto full = std::span<const std::size_t>{};
    p.execute(plain.data(), full, full);
    p.execute(scaled.data(), full, full, T(2));
    // fct multiplies the output: compare against twice the unscaled result.
    std::vector<std::complex<T>> want(plain.size());
    for (std::size_t i = 0; i < plain.size(); ++i) want[i] = T(2) * plain[i];
    require_close(scaled, want, fft_tol<T>());
}
