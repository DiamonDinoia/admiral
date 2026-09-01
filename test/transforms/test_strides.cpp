#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/nd_plan.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <vector>

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

    admiral::plan<T> ref(len);
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
    for (std::size_t i = 0; i < out_n; ++i)
        if (!touched[i]) want[i] = orig_dst[i];

    require_close(dst, want, fft_tol<T>());
}

}

TEMPLATE_TEST_CASE("strides_plan matches per-line plan", "[transforms][strides]", float, double) {
    using T = TestType;
    for (const bool forward : {true, false}) {
        check<T>(64, 8, 1, 64, 1, 64, forward);
        check<T>(64, 8, 1, 96, 1, 80, forward);
        check<T>(64, 16, 1, 64, 16, 1, forward);
        check<T>(37, 11, 1, 37, 11, 1, forward);
        check<T>(64, 37, 37, 1, 37, 1, forward);
        check<T>(1024, 33, 33, 1, 33, 1, forward);
        check<T>(64, 32, 64, 1, 32, 1, forward);
        check<T>(64, 32, 32, 1, 64, 1, forward);
        check<T>(64, 16, 32, 3, 32, 5, forward);
        check<T>(16, 7, 16, 2, 16, 2, forward);
        check<T>(64, 37, 64, 1, 64, 3, forward);
        check<T>(100, 9, 100, 1, 100, 2, forward);
        check<T>(16, 32, 16, 1, 16, 2, forward);
        check<T>(37, 16, 16, 1, 16, 1, forward);
        check<T>(37, 9, 18, 2, 18, 4, forward);
        check<T>(37, 12, 37, 1, 37, 3, forward);
        check<T>(64, 2, 2, 1, 8192, 1, forward);
        check<T>(32, 24, 24, 1, 24, 1, forward);
        check<T>(256, 3, 3, 1, 3, 1, forward);
        check<T>(256, 1, 1, 1, 1, 1, forward);
    }
}

TEMPLATE_TEST_CASE("column codelet lens <= 64 match per-line plan, tails included",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    REQUIRE(admiral::detail::make_nd_axis_state<T>(16, 17, true, false).col_codelet);
    REQUIRE(admiral::detail::make_nd_axis_state<T>(8, 17, true, false).col_codelet);
    REQUIRE(!admiral::detail::make_nd_axis_state<T>(96, 17, true, false).col_codelet);

    using admiral::detail::e2_len_cap;
    using admiral::detail::e2_len_cap_by_l3;
    CHECK(e2_len_cap_by_l3((std::size_t{3} << 20) / 2) == 32);
    CHECK(e2_len_cap_by_l3(std::size_t{4} << 20) == 64);
    CHECK(e2_len_cap_by_l3(std::size_t{5} << 20) == 64);
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
            check<T>(96, nbatch, nbatch, 1, nbatch, 1, forward);
            check<T>(192, nbatch, nbatch, 1, nbatch, 1, forward);
        }
}

TEMPLATE_TEST_CASE("strides_plan in place and scaled", "[transforms][strides]", float, double) {
    using T = TestType;
    constexpr std::size_t len = 60, nbatch = 20;
    const auto orig = make_input<T>(len * nbatch, 0x1ACE);
    auto data = orig;
    admiral::strides_plan<T> lp(len, nbatch, nbatch, 1, nbatch, 1);
    lp.inverse(data.data(), data.data());
    admiral::plan<T> ref(len);
    for (std::size_t l = 0; l < nbatch; ++l) {
        std::vector<std::complex<T>> want(len), got(len);
        for (std::size_t p = 0; p < len; ++p) want[p] = orig[p * nbatch + l];
        ref.inverse(span<std::complex<T>>(want));
        for (std::size_t p = 0; p < len; ++p) got[p] = data[p * nbatch + l];
        require_close(got, want, fft_tol<T>());
    }

    {
        const auto rows_orig = make_input<T>(len * nbatch, 0x0FF1);
        auto rows = rows_orig;
        admiral::strides_plan<T> rp(len, nbatch, 1, len, 1, len);
        rp.forward(rows.data(), rows.data());
        admiral::plan<T> rref(len);
        for (std::size_t l = 0; l < nbatch; ++l) {
            std::vector<std::complex<T>> want(rows_orig.begin() + std::ptrdiff_t(l * len),
                                              rows_orig.begin() + std::ptrdiff_t((l + 1) * len));
            rref.forward(span<std::complex<T>>(want));
            std::vector<std::complex<T>> got(rows.begin() + std::ptrdiff_t(l * len),
                                             rows.begin() + std::ptrdiff_t((l + 1) * len));
            require_close(got, want, fft_tol<T>());
        }
    }

    {
        admiral::options opts;
        opts.nthreads = 4;
        const auto tsrc = make_input<T>(len * nbatch, 0x7413);
        std::vector<std::complex<T>> tout(len * nbatch);
        admiral::strides_plan<T> tp(len, nbatch, nbatch, 1, nbatch, 1, opts);
        tp.forward(tsrc.data(), tout.data());
        admiral::plan<T> tref(len);
        for (std::size_t l = 0; l < nbatch; ++l) {
            std::vector<std::complex<T>> want(len), got(len);
            for (std::size_t p = 0; p < len; ++p) want[p] = tsrc[p * nbatch + l];
            tref.forward(span<std::complex<T>>(want));
            for (std::size_t p = 0; p < len; ++p) got[p] = tout[p * nbatch + l];
            require_close(got, want, fft_tol<T>());
        }
    }

    const auto src = make_input<T>(len * nbatch, 0x5CA1E);
    std::vector<std::complex<T>> out(len * nbatch);
    std::vector<std::complex<T>> want(len * nbatch);
    lp.forward(src.data(), out.data(), T(2.5));
    admiral::plan<T> ref2(len);
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
    const auto src = make_input<T>(4, 0x1DE71);
    std::vector<std::complex<T>> out(4);
    admiral::strides_plan<T> lp(1, 4, 1, 1, 1, 1);
    lp.forward(src.data(), out.data(), T(3));
    for (std::size_t i = 0; i < 4; ++i) REQUIRE(out[i] == src[i] * T(3));

    REQUIRE_THROWS_AS((admiral::strides_plan<T>(0, 4, 1, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 0, 1, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 0, 1, 1, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 1, 1, 0, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 4, 0, 4, 1)), admiral::size_error);
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(4, 4, 4, 1, 4, 0)), admiral::size_error);

    REQUIRE_NOTHROW((admiral::strides_plan<T>(4, 1, 1, 0, 1, 0)));

    constexpr std::size_t big = std::numeric_limits<std::size_t>::max() / 2;
    REQUIRE_THROWS_AS((admiral::strides_plan<T>(big, 4, 1, 1, 1, 1)), admiral::size_error);

    auto data = make_input<T>(16, 0xBAD);
    admiral::strides_plan<T> bad(4, 4, 4, 1, 1, 4);
    REQUIRE_THROWS_AS(bad.forward(data.data(), data.data()), admiral::size_error);
}

namespace {

template<typename T>
void require_align_stable(std::size_t len, std::size_t nbatch, bool forward, bool axis) {
    constexpr std::size_t kOff = 16;
    const std::size_t n = len * nbatch;
    const auto in = make_input<T>(n, 0xA11C);
    std::vector<std::complex<T>> ref(n);

    for (std::size_t off = 0; off < kOff; ++off) {
        std::vector<std::complex<T>> buf(n + kOff);
        std::copy(in.begin(), in.end(), buf.begin() + static_cast<std::ptrdiff_t>(off));
        std::complex<T>* const p = buf.data() + off;
        if (axis) {
            const std::size_t shape[2] = {len, nbatch};
            admiral::axis_plan<T> ap(span<const std::size_t>(shape, 2), 0, forward);
            ap.execute(p, {}, {});
        } else {
            admiral::strides_plan<T> sp(len, nbatch, nbatch, 1, nbatch, 1);
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

template<typename T>
void require_output_layout_stable(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                                  std::size_t in_dist, bool forward) {
    const std::size_t n = len * nbatch;
    const std::size_t in_n = (len - 1) * in_stride + (nbatch - 1) * in_dist + 1;
    const auto src = make_input<T>(in_n, 0x0517);
    std::vector<std::complex<T>> ref(n);

    const std::size_t layouts[][2] = {{nbatch, 1}, {8192, 1}, {1, len}, {2, 2 * len}};
    bool first = true;
    for (const auto& lo : layouts) {
        const std::size_t os = lo[0], od = lo[1];
        std::vector<std::complex<T>> dst((len - 1) * os + (nbatch - 1) * od + 1);
        admiral::strides_plan<T> sp(len, nbatch, in_stride, in_dist, os, od);
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

}

TEMPLATE_TEST_CASE("strides_plan bits do not depend on the output layout",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    for (const bool forward : {true, false}) {
        require_output_layout_stable<T>(64, 2, 2, 1, forward);
        require_output_layout_stable<T>(60, 2, 2, 1, forward);
        require_output_layout_stable<T>(64, 2, 8192, 1, forward);
        require_output_layout_stable<T>(64, 8, 1, 64, forward);
    }
}

TEMPLATE_TEST_CASE("column engine is bit-identical across alignment classes",
                   "[transforms][strides][numerics]", float, double) {
    using T = TestType;
    for (const std::size_t len : {std::size_t{20}, std::size_t{60}, std::size_t{96},
                                  std::size_t{192}, std::size_t{256}})
        for (const bool forward : {true, false})
            for (const bool axis : {true, false})
                require_align_stable<T>(len, 16, forward, axis);
}
