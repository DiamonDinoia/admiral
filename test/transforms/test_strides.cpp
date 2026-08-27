#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>   // strides_plan, plan

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <vector>

// strides_plan<T> transforms a batch of strided lines out of place:
//   transform l, element p:
//       src[p * in_stride + l * in_dist] -> dst[p * out_stride + l * out_dist]
// Oracle: one admiral::plan<T> per line, gathered and scattered by hand.

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
//   col_dif    unit dist both sides, smooth len (SIMD column chain)
//   transposed dist > 1, or non-codelet len (gather -> plan -> scatter)
TEMPLATE_TEST_CASE("strides_plan matches per-line plan", "[transforms][strides]", float, double) {
    using T = TestType;
    for (const bool forward : {true, false}) {
        // Contiguous rows, compact and padded dists.
        check<T>(64, 8, 1, 64, 1, 64, forward);
        check<T>(64, 8, 1, 96, 1, 80, forward);
        // Contiguous rows in, column layout out: the per-line engine writes through
        // one line of scratch and scatters. Unit in-stride is the only side that is
        // contiguous, so no other case reaches this scatter.
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
        // Prime length with unit in-dist and a spread destination: the slab route
        // needs a col chain, so this length falls through to the transposed one.
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

    // Threads change how the batch loop and the column tiles are cut, so run the
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

    // Custom scale rides the out-of-place call.
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
    // len == 1: identity axis, a scaled strided copy.
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

    // len * nbatch has to fit size_t: the plan sizes its scratch from that product.
    constexpr std::size_t big = std::numeric_limits<std::size_t>::max() / 2;
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(big, 4, 1, 1, 1, 1)), admiral::size_error);

    // In place with mismatched strides is an error at execute time.
    auto data = make_input<T>(16, 0xBAD);
    admiral::strides_plan<T> bad(4, 4, 4, 1, 1, 4);
    REQUIRE_THROWS_AS(bad.forward(data.data(), data.data()), admiral::size_error);
}

namespace {

// One input layout, several output layouts, results gathered to logical (p, l)
// order: bitwise equal. The route and the factoring proxy pick the numbers, so
// this holds only while both read the input geometry alone. Pricing the output
// stride into either breaks it: the wide-stride layout then flips the chooser
// (f64 needs 2 * nbatch <= W, so v4 there) and, at pow2 f32, the radix-4 proxy.
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
        // Col-route input (pow2 and non-pow2), a transposed-route input whose wide
        // element stride puts every output layout on the gather path, and a rows
        // input (per-line engine, direct store vs line scratch and scatter).
        require_output_layout_stable<T>(64, 2, 2, 1, forward);
        require_output_layout_stable<T>(60, 2, 2, 1, forward);
        require_output_layout_stable<T>(64, 2, 8192, 1, forward);
        require_output_layout_stable<T>(64, 8, 1, 64, forward);
    }
}
