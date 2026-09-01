#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>   // `strides_plan`, `plan`
#include <admiral/detail/nd_plan.hpp>   // `make_nd_axis_state` (the col-codelet route pin)

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <vector>

// `strides_plan<T>` transforms a batch of strided lines out of place:
//   transform l, element p:
//       src[p * in_stride + l * in_dist] -> dst[p * out_stride + l * out_dist]
// Oracle: gather each line, transform with `admiral::plan<T>`, scatter into `want`.

using admiral::span;

namespace {

template<typename T>
void check(std::size_t len, std::size_t nbatch, std::size_t in_stride, std::size_t in_dist,
           std::size_t out_stride, std::size_t out_dist, bool forward) {
    const std::size_t in_n = (len - 1) * in_stride + (nbatch - 1) * in_dist + 1;
    const std::size_t out_n = (len - 1) * out_stride + (nbatch - 1) * out_dist + 1;
    const auto src = make_input<T>(in_n, 0x11E5);
    const auto orig_dst = make_input<T>(out_n, 0xDE57);
    auto dst = orig_dst;

    admiral::strides_plan<T> lp(len, nbatch, in_stride, in_dist, out_stride, out_dist);
    REQUIRE(lp.size() == len * nbatch);
    if (forward) lp.forward(src.data(), dst.data());
    else         lp.inverse(src.data(), dst.data());

    const admiral::plan<T> ref(len);
    std::vector<std::complex<T>> want(out_n);
    std::vector<char> touched(out_n, 0);
    for (std::size_t l = 0; l < nbatch; ++l) {
        std::vector<std::complex<T>> line(len);
        for (std::size_t p = 0; p < len; ++p) line[p] = src[p * in_stride + l * in_dist];
        if (forward) ref.forward(span<std::complex<T>>(line));
        else         ref.inverse(span<std::complex<T>>(line));
        for (std::size_t p = 0; p < len; ++p) {
            const std::size_t i = p * out_stride + l * out_dist;
            want[i] = line[p];
            touched[i] = 1;
        }
    }
    // Disjointness: the test strides must address every line's span without
    // aliasing, or the untouched check below judges its own overwrites.
    for (std::size_t i = 0; i < out_n; ++i)
        if (!touched[i]) want[i] = orig_dst[i];

    require_close(dst, want, fft_tol<T>());
}

}  // namespace

// Both directions, over geometries that hit every route:
//   rows       in_stride == out_stride == 1 (contiguous lines, any dist)
//   col        unit dist both sides, smooth len (SIMD column engine: the one-call
//              column codelet at <= 64, the dif chain above)
//   transposed dist > 1, or non-codelet len (gather -> plan -> scatter)
TEMPLATE_TEST_CASE("strides_plan matches per-line plan", "[transforms][strides]", float, double) {
    using T = TestType;
    for (const bool forward : {true, false}) {
        // Contiguous rows, compact and padded dists.
        check<T>(64, 8, 1, 64, 1, 64, forward);
        check<T>(64, 8, 1, 96, 1, 80, forward);
        // Contiguous rows in, column layout out: the per-line engine writes through
        // one line of scratch and scatters. `in_stride` == 1 is the only contiguous
        // side, so no other case reaches this scatter.
        check<T>(64, 16, 1, 64, 16, 1, forward);
        check<T>(37, 11, 1, 37, 11, 1, forward);
        // Column layout of a (len x nbatch) contiguous tensor, axis 0.
        check<T>(64, 37, 37, 1, 37, 1, forward);
        check<T>(1024, 33, 33, 1, 33, 1, forward);
        // Strided source into contiguous destination (sliced-view input), and back.
        check<T>(64, 32, 64, 1, 32, 1, forward);
        check<T>(64, 32, 32, 1, 64, 1, forward);
        // Non-unit dists force the transposed route.
        check<T>(64, 16, 32, 3, 32, 5, forward);
        check<T>(16, 7, 16, 2, 16, 2, forward);
        // Unit-dist source into a dist-spread destination: col chain to slab + scatter.
        check<T>(64, 37, 64, 1, 64, 3, forward);
        check<T>(100, 9, 100, 1, 100, 2, forward);
        check<T>(16, 32, 16, 1, 16, 2, forward);
        // Non-codelet length (prime): only the transposed route exists.
        check<T>(37, 16, 16, 1, 16, 1, forward);
        check<T>(37, 9, 18, 2, 18, 4, forward);
        // Prime length, unit `in_dist`, spread destination: the slab route needs a
        // col chain, so a prime length falls through to the transposed route.
        check<T>(37, 12, 37, 1, 37, 3, forward);
        // Narrow batch into a wide destination stride: the col route must hold
        // (the route reads the input geometry only), not flip on the output.
        check<T>(64, 2, 2, 1, 8192, 1, forward);
        // Radix-32-capable single-pass length.
        check<T>(32, 24, 24, 1, 24, 1, forward);
        // Narrow batch: below one SIMD register on some widths.
        check<T>(256, 3, 3, 1, 3, 1, forward);
        check<T>(256, 1, 1, 1, 1, 1, forward);
    }
}

// One-call column codelet (col axis len <= 64 catalog): the per-line oracle must match
// across the sweep, exercising the full tiles, the scalar tail tile (nbatch not a
// multiple of W, including below one register), and the dif-chain controls above 64.
TEMPLATE_TEST_CASE("column codelet lens <= 64 match per-line plan, tails included",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    // Route pin: the col codelet arm must actually carry these lengths, or the sweep
    // below exercises nothing. 16 is in the catalog of every build config (sanitizers
    // cap it at 16); 96 is smooth but past 64, so it stays on the dif chain.
    REQUIRE(admiral::detail::make_nd_axis_state<T>(16, 17, true, false).col_codelet);
    REQUIRE(admiral::detail::make_nd_axis_state<T>(8, 17, true, false).col_codelet);
    REQUIRE(!admiral::detail::make_nd_axis_state<T>(96, 17, true, false).col_codelet);

    // The len-64 arm is gated by per-core probed L3 (knob A/B 2026-08-31: ice
    // 1.5 MiB fails, rome 4 MiB and genoa 5.3 MiB pass). Synthetic geometries —
    // the build's L3 can't be faked in one process, so the predicate is driven
    // directly — and the live gate must elect consistently with it.
    using admiral::detail::e2_len_cap;
    using admiral::detail::e2_len_cap_by_l3;
    CHECK(e2_len_cap_by_l3((std::size_t{3} << 20) / 2) == 32);  // 1.5 MiB, ice-class
    CHECK(e2_len_cap_by_l3(std::size_t{4} << 20) == 64);        // rome-class
    CHECK(e2_len_cap_by_l3(std::size_t{5} << 20) == 64);        // genoa-class
    CHECK(e2_len_cap_by_l3(0) == 32);
    CHECK(admiral::detail::make_nd_axis_state<T>(64, 17, true, false).col_codelet ==
          (e2_len_cap() == 64));
    for (const bool forward : {true, false})
        for (const std::size_t nbatch : {std::size_t{3}, std::size_t{7}, std::size_t{17}}) {
            for (const std::size_t len :
                 {std::size_t{2}, std::size_t{4}, std::size_t{8}, std::size_t{16},
                  std::size_t{20}, std::size_t{32}, std::size_t{33}, std::size_t{60},
                  std::size_t{64}})
                check<T>(len, nbatch, nbatch, 1, nbatch, 1, forward);
            // dif-chain controls: smooth, past the codelet arm.
            check<T>(96, nbatch, nbatch, 1, nbatch, 1, forward);
            check<T>(192, nbatch, nbatch, 1, nbatch, 1, forward);
        }
}

TEMPLATE_TEST_CASE("strides_plan in place and scaled", "[transforms][strides]", float, double) {
    using T = TestType;
    constexpr std::size_t len = 60, nbatch = 20;
    // In place over a column layout of a contiguous (len x nbatch) tensor.
    const auto orig = make_input<T>(len * nbatch, 0x1ACE);
    auto data = orig;
    admiral::strides_plan<T> lp(len, nbatch, nbatch, 1, nbatch, 1);
    lp.inverse(data.data(), data.data());
    const admiral::plan<T> ref(len);
    for (std::size_t l = 0; l < nbatch; ++l) {
        std::vector<std::complex<T>> want(len), got(len);
        for (std::size_t p = 0; p < len; ++p) want[p] = orig[p * nbatch + l];
        ref.inverse(span<std::complex<T>>(want));
        for (std::size_t p = 0; p < len; ++p) got[p] = data[p * nbatch + l];
        require_close(got, want, fft_tol<T>());
    }

    // In place over contiguous rows: the other in-place route, one whole transform
    // per line rather than a column chain.
    {
        const auto rows_orig = make_input<T>(len * nbatch, 0x0FF1);
        auto rows = rows_orig;
        const admiral::strides_plan<T> rp(len, nbatch, 1, len, 1, len);
        rp.forward(rows.data(), rows.data());
        const admiral::plan<T> rref(len);
        for (std::size_t l = 0; l < nbatch; ++l) {
            std::vector<std::complex<T>> want(rows_orig.begin() + std::ptrdiff_t(l * len),
                                              rows_orig.begin() + std::ptrdiff_t((l + 1) * len));
            rref.forward(span<std::complex<T>>(want));
            std::vector<std::complex<T>> got(rows.begin() + std::ptrdiff_t(l * len),
                                             rows.begin() + std::ptrdiff_t((l + 1) * len));
            require_close(got, want, fft_tol<T>());
        }
    }

    // Threads change how the batch loop and the column tiles are cut. Run the
    // same geometry against the same oracle with the pool forced on.
    {
        admiral::options opts;
        opts.nthreads = 4;
        const auto tsrc = make_input<T>(len * nbatch, 0x7413);
        std::vector<std::complex<T>> tout(len * nbatch);
        const admiral::strides_plan<T> tp(len, nbatch, nbatch, 1, nbatch, 1, opts);
        tp.forward(tsrc.data(), tout.data());
        const admiral::plan<T> tref(len);
        for (std::size_t l = 0; l < nbatch; ++l) {
            std::vector<std::complex<T>> want(len), got(len);
            for (std::size_t p = 0; p < len; ++p) want[p] = tsrc[p * nbatch + l];
            tref.forward(span<std::complex<T>>(want));
            for (std::size_t p = 0; p < len; ++p) got[p] = tout[p * nbatch + l];
            require_close(got, want, fft_tol<T>());
        }
    }

    // The custom scale applies through the out-of-place call.
    const auto src = make_input<T>(len * nbatch, 0x5CA1E);
    std::vector<std::complex<T>> out(len * nbatch);
    std::vector<std::complex<T>> want(len * nbatch);
    lp.forward(src.data(), out.data(), T(2.5));
    const admiral::plan<T> ref2(len);
    for (std::size_t l = 0; l < nbatch; ++l) {
        std::vector<std::complex<T>> line(len);
        for (std::size_t p = 0; p < len; ++p) line[p] = src[p * nbatch + l];
        ref2.forward(span<std::complex<T>>(line), T(2.5));
        for (std::size_t p = 0; p < len; ++p) want[p * nbatch + l] = line[p];
    }
    require_close(out, want, fft_tol<T>());
}

TEMPLATE_TEST_CASE("strides_plan degenerate and rejected geometry", "[transforms][strides]",
                   float, double) {
    using T = TestType;
    // `len` == 1: identity axis, a scaled strided copy.
    const auto src = make_input<T>(4, 0x1DE71);
    std::vector<std::complex<T>> out(4);
    admiral::strides_plan<T> lp(1, 4, 1, 1, 1, 1);
    lp.forward(src.data(), out.data(), T(3));
    for (std::size_t i = 0; i < 4; ++i) REQUIRE(out[i] == src[i] * T(3));

    // Zero stride/extent rejections, on both sides.
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(0, 4, 1, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 0, 1, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 0, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 1, 1, 0, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 4, 0, 4, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 4, 1, 4, 0)), admiral::size_error);

    // A dist separates transforms, so with one transform there is nothing for it to
    // separate and zero is legal on either side.
    REQUIRE_NOTHROW((admiral::strides_plan<T>(4, 1, 1, 0, 1, 0)));

    // `len` * `nbatch` must fit in `size_t`: the plan sizes its scratch from that product.
    constexpr std::size_t big = std::numeric_limits<std::size_t>::max() / 2;
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(big, 4, 1, 1, 1, 1)), admiral::size_error);

    // In place with mismatched strides is an error at execute time.
    auto data = make_input<T>(16, 0xBAD);
    admiral::strides_plan<T> bad(4, 4, 4, 1, 1, 4);
    REQUIRE_THROWS_AS(bad.forward(data.data(), data.data()), admiral::size_error);
}

// Bit-identical across alignment classes.
//
// The column engine clones its store-policy lambda per inline context; under
// fast-math the compiler contracts and reassociates each clone differently.
// Which clone covers a column depends on where the data starts. Without the
// numerics pin in `src/CMakeLists.txt`, one transform returns 1-ULP-different
// bits per alignment class. Every check above compares against a tolerance and
// passes right through a 1-ULP difference; only bitwise equality catches one.
//
// The check is a real one. Strip the pin and rebuild: this case fails at
// Release/x86-64-v4 with gcc 14.2.
namespace {

// Runs one transform at every element offset 0..15 into the start of a buffer and
// requires the results to agree bit for bit. Two plan types, one engine.
template<typename T>
void require_align_stable(std::size_t len, std::size_t nbatch, bool forward, bool axis) {
    constexpr std::size_t kOff = 16;   // covers `W` for every ISA level admiral targets
    const std::size_t n = len * nbatch;
    const auto in = make_input<T>(n, 0xA11C);
    std::vector<std::complex<T>> ref(n);

    for (std::size_t off = 0; off < kOff; ++off) {
        std::vector<std::complex<T>> buf(n + kOff);
        std::copy(in.begin(), in.end(), buf.begin() + static_cast<std::ptrdiff_t>(off));
        std::complex<T>* const p = buf.data() + off;
        if (axis) {
            const std::size_t shape[2] = {len, nbatch};
            const admiral::axis_plan<T> ap(span<const std::size_t>(shape, 2), 0, forward);
            ap.execute(p, {}, {});
        } else {
            const admiral::strides_plan<T> sp(len, nbatch, nbatch, 1, nbatch, 1);
            if (forward) sp.forward(p, p);
            else         sp.inverse(p, p);
        }
        if (off == 0) {
            std::copy(p, p + n, ref.begin());
            continue;
        }
        std::size_t diffs = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (std::memcmp(&p[i], &ref[i], sizeof(std::complex<T>)) != 0) ++diffs;
        INFO("offset " << off << " len " << len << " nbatch " << nbatch
                       << (axis ? " axis_plan" : " strides_plan")
                       << (forward ? " forward" : " inverse"));
        REQUIRE(diffs == 0);
    }
}

// One input layout, several output layouts, results gathered to logical (p, l)
// order: bitwise equal. The route and the factoring proxy pick the numbers, so
// bitwise equality holds only while both read the input geometry alone. Price the
// output stride into the route or the proxy and bitwise equality breaks: the
// wide-stride layout flips the chooser (f64 needs 2 * nbatch <= W, so v4 hits
// the flip), and pow2 f32 flips the radix-4 proxy.
template<typename T>
void require_output_layout_stable(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                                  std::size_t in_dist, bool forward) {
    const std::size_t n = len * nbatch;
    const std::size_t in_n = (len - 1) * in_stride + (nbatch - 1) * in_dist + 1;
    const auto src = make_input<T>(in_n, 0x0517);
    std::vector<std::complex<T>> ref(n);

    // Packed columns (direct col), wide stride (the chooser-flip arm), rows and a
    // strided spread (both dist != 1, the slab arm when the col route is chosen).
    const std::size_t layouts[][2] = {{nbatch, 1}, {8192, 1}, {1, len}, {2, 2 * len}};
    bool first = true;
    for (const auto& lo : layouts) {
        const std::size_t os = lo[0], od = lo[1];
        std::vector<std::complex<T>> dst((len - 1) * os + (nbatch - 1) * od + 1);
        const admiral::strides_plan<T> sp(len, nbatch, in_stride, in_dist, os, od);
        if (forward) sp.forward(src.data(), dst.data());
        else         sp.inverse(src.data(), dst.data());
        std::size_t diffs = 0;
        for (std::size_t p = 0; p < len; ++p)
            for (std::size_t l = 0; l < nbatch; ++l) {
                const std::complex<T>& got = dst[p * os + l * od];
                std::complex<T>& want = ref[p * nbatch + l];
                if (first) want = got;
                else if (std::memcmp(&got, &want, sizeof got) != 0) ++diffs;
            }
        first = false;
        INFO("len " << len << " nbatch " << nbatch << " in_stride " << in_stride
                    << " out (" << os << ", " << od << ")"
                    << (forward ? " forward" : " inverse"));
        REQUIRE(diffs == 0);
    }
}

}  // namespace

TEMPLATE_TEST_CASE("strides_plan bits do not depend on the output layout",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    for (const bool forward : {true, false}) {
        // Col-route inputs (pow2 and non-pow2), a transposed-route input, and a
        // rows input (per-line engine, direct store versus line scratch and
        // scatter). The wide element stride puts every output layout on the gather
        // path.
        require_output_layout_stable<T>(64, 2, 2, 1, forward);
        require_output_layout_stable<T>(60, 2, 2, 1, forward);
        require_output_layout_stable<T>(64, 2, 8192, 1, forward);
        require_output_layout_stable<T>(64, 8, 1, 64, forward);
    }
}

TEMPLATE_TEST_CASE("column engine is bit-identical across alignment classes",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    // 20 and 60 broke without the pin; they now run the col codelet (per-lane
    // bit-stable across tiles by construction). 96, 192 and 256 keep the dif col
    // chain covered, which is where the -ffp-contract pin lives.
    for (const std::size_t len : {std::size_t{20}, std::size_t{60}, std::size_t{96},
                                  std::size_t{192}, std::size_t{256}})
        for (const bool forward : {true, false})
            for (const bool axis : {true, false})
                require_align_stable<T>(len, 16, forward, axis);
}
