#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <xsimd/xsimd.hpp>
#include <admiral/detail/four_step_large.hpp>
#include <admiral/detail/plan.hpp>
#include <admiral/detail/scratch.hpp>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

template<typename T>
std::vector<std::complex<T>> tone_input(std::size_t N, std::size_t K) {
    std::vector<std::complex<T>> v(N);
    for (std::size_t n = 0; n < N; ++n) v[n] = unit_phasor<T>(turn_fraction(K, n, N));
    return v;
}

template<typename T>
std::vector<std::complex<T>> tone_spectrum(std::size_t N, std::size_t K) {
    std::vector<std::complex<T>> ref(N, std::complex<T>(0, 0));
    ref[K] = std::complex<T>(static_cast<T>(N), 0);
    return ref;
}

template<typename T>
struct offset_buffer {
    static constexpr std::size_t kAlign = admiral::detail::span_align<T>;
    std::vector<std::complex<T>> storage;
    std::complex<T>* ptr;
    explicit offset_buffer(std::size_t n, std::size_t off_elems)
        : storage(n + kAlign / sizeof(std::complex<T>) + off_elems) {
        auto base = reinterpret_cast<std::uintptr_t>(storage.data());
        std::size_t pad = (kAlign - (base % kAlign)) % kAlign / sizeof(std::complex<T>);
        ptr = storage.data() + pad + off_elems;
    }
};

constexpr std::size_t kBelowFuse = 1048576;
constexpr std::size_t kRect2M    = 2097152;
constexpr std::size_t kNonPow2   = 2073600;
constexpr std::size_t kAboveFuse = 8294400;

}

TEST_CASE("four_step_large forward vs analytical (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kRect2M, kNonPow2}) {
        CAPTURE(N);
        CHECK(std::string(admiral::detail::plan_impl<double>(N, true).route_name())
              == "four_step_large");
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<double>(N, K);
        const auto ref = tone_spectrum<double>(N, K);

        auto plan = admiral::plan<double>(N);
        auto out = in;
        plan.forward(admiral::span(out));

        require_close(out, ref, fft_tol<double>());
    }
}

TEST_CASE("four_step_large round-trip identity (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kRect2M, kNonPow2, kAboveFuse}) {
        CAPTURE(N);
        CHECK(std::string(admiral::detail::plan_impl<double>(N, true).route_name())
              == "four_step_large");
        const auto in = make_input<double>(N, 77u + unsigned(N));
        auto plan = admiral::plan<double>(N);

        auto data = in;
        plan.forward(admiral::span(data));
        plan.inverse(admiral::span(data));

        require_close(data, in, fft_tol<double>());
    }
}

TEST_CASE("four_step_large impulse flatness (double)", "[large][fourstep]") {
    constexpr std::size_t N = 1048576;
    const std::size_t n0 = 97;
    std::vector<std::complex<double>> in(N, {0.0, 0.0}), out(N);
    in[n0] = {1.0, 0.0};
    auto plan = admiral::plan<double>(N);
    plan.forward(in.data(), out.data());

    std::vector<std::complex<double>> ref(N);
    for (std::size_t k = 0; k < N; ++k)
        ref[k] = std::conj(unit_phasor<double>(turn_fraction(n0, k, N)));
    require_close(out, ref, fft_tol<double>());
}

TEST_CASE("four_step_large fused-band impulse (double)", "[large][fourstep]") {
    for (const std::size_t N : {kRect2M, std::size_t{4194304}}) {
        CAPTURE(N);
        const std::size_t n0 = N / 8 + 97;
        std::vector<std::complex<double>> in(N, {0.0, 0.0});
        in[n0] = {1.0, 0.0};
        auto plan = admiral::plan<double>(N);

        std::vector<std::complex<double>> fwd_ref(N), inv_ref(N);
        for (std::size_t k = 0; k < N; ++k) {
            fwd_ref[k] = std::conj(unit_phasor<double>(turn_fraction(n0, k, N)));
            inv_ref[k] = unit_phasor<double>(turn_fraction(n0, k, N)) / double(N);
        }

        std::vector<std::complex<double>> out(N);
        plan.forward(in.data(), out.data());
        require_close(out, fwd_ref, fft_tol<double>());
        plan.inverse(in.data(), out.data());
        require_close(out, inv_ref, fft_tol<double>());

        auto ip = in;
        plan.forward(admiral::span(ip));
        require_close(ip, fwd_ref, fft_tol<double>());
        ip = in;
        plan.inverse(admiral::span(ip));
        require_close(ip, inv_ref, fft_tol<double>());
    }
}

TEST_CASE("four_step_large tone and impulse (float)", "[large][fourstep]") {
    for (const auto& [N, nt] : {std::pair{std::size_t{4194304}, std::size_t{1}},
                                {std::size_t{2097152}, std::size_t{1}},
                                {std::size_t{1048576}, std::size_t{16}}}) {
        CAPTURE(N, nt);
        CHECK(std::string(admiral::detail::plan_impl<float>(N, true, nt).route_name())
              == "four_step_large");
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<float>(N, K);
        const auto ref = tone_spectrum<float>(N, K);
        auto plan = admiral::plan<float>({N}, {nt});
        std::vector<std::complex<float>> out(N);
        plan.forward(in.data(), out.data());
        require_close(out, ref, fft_tol<float>(64));

        std::vector<std::complex<float>> imp(N, {0.0f, 0.0f}), got(N), iref(N);
        imp[N / 8 + 97] = {1.0f, 0.0f};
        plan.inverse(imp.data(), got.data());
        for (std::size_t k = 0; k < N; ++k)
            iref[k] = unit_phasor<float>(turn_fraction(N / 8 + 97, k, N)) / float(N);
        require_close(got, iref, fft_tol<float>(64));
    }
}

TEMPLATE_TEST_CASE("execute() input/output alignment variants vs analytical",
                   "[large][align]", float, double) {
    using T = TestType;
    for (const std::size_t N : {std::size_t{4096}, std::size_t{1048576}}) {
        CAPTURE(N);
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<T>(N, K);
        const auto ref = tone_spectrum<T>(N, K);
        const double tol = fft_tol<T>();
        auto plan = admiral::plan<T>(N);

        for (const std::size_t in_off : {std::size_t{0}, std::size_t{1}}) {
            for (const std::size_t out_off : {std::size_t{0}, std::size_t{1}}) {
                CAPTURE(in_off, out_off);
                offset_buffer<T> src(N, in_off);
                offset_buffer<T> dst(N, out_off);
                std::copy(in.begin(), in.end(), src.ptr);

                plan.forward(src.ptr, dst.ptr);

                std::vector<std::complex<T>> got(dst.ptr, dst.ptr + N);
                require_close(got, ref, tol);

                for (std::size_t i = 0; i < N; ++i) REQUIRE(src.ptr[i] == in[i]);
            }
        }
    }
}

TEMPLATE_TEST_CASE("out-aliased SoA pair equals the 4-plane path", "[dif][align]", float, double) {
    using T = TestType;
    std::size_t fired = 0;
    for (const std::size_t N : {std::size_t{4096}, std::size_t{12288}, std::size_t{24576},
                                std::size_t{49152}, std::size_t{65536}}) {
        const auto tw = admiral::detail::build_dif_twiddle_set<T>(N);
        const auto in = tone_input<T>(N, N / 4 + 7);
        for (const bool forward : {true, false}) {
            offset_buffer<T> aligned(N, 0), off(N, 1);
            if (!admiral::detail::dif_out_aliasable<T>(forward, aligned.ptr, N, tw)) continue;
            CAPTURE(N, forward);
            REQUIRE_FALSE(admiral::detail::dif_out_aliasable<T>(forward, off.ptr, N, tw));
            ++fired;
            admiral::detail::dif_execute_in_place<T>(forward, in.data(), aligned.ptr, N, tw, T(1));
            admiral::detail::dif_execute_in_place<T>(forward, in.data(), off.ptr, N, tw, T(1));
            require_close(std::vector<std::complex<T>>(off.ptr, off.ptr + N),
                          std::vector<std::complex<T>>(aligned.ptr, aligned.ptr + N),
                          fft_tol<T>());

            offset_buffer<T> ip(N, 0);
            std::copy(in.begin(), in.end(), ip.ptr);
            admiral::detail::dif_execute_in_place<T>(forward, ip.ptr, ip.ptr, N, tw, T(1));
            for (std::size_t i = 0; i < N; ++i) REQUIRE(ip.ptr[i] == aligned.ptr[i]);
        }
    }
    CAPTURE(fired);
    REQUIRE(fired > 0);
}

TEMPLATE_TEST_CASE("four_step_transpose_inplace vs naive", "[large][fourstep]", float, double) {
    using T = TestType;
    static constexpr std::size_t shapes[][2] = {{64, 64}, {32, 96}, {96, 32},
                                                {48, 80}, {49, 128}, {33, 100}};
    for (const auto& s : shapes) {
        const std::size_t R = s[0], C = s[1];
        CAPTURE(R, C);
        std::vector<std::complex<T>> m(R * C), want(R * C);
        for (std::size_t i = 0; i < R * C; ++i)
            m[i] = {static_cast<T>(i), static_cast<T>(2 * i + 1)};
        for (std::size_t i = 0; i < R; ++i)
            for (std::size_t j = 0; j < C; ++j) want[j * R + i] = m[i * C + j];
        admiral::detail::four_step_transpose_inplace<T>(m.data(), R, C, nullptr);
        REQUIRE(m == want);
    }
}

TEST_CASE("WS-3 sweep-bits invariant pins the pool gate", "[large][fourstep]") {
    using admiral::detail::fsl_ws_engaged;
    admiral::detail::thread_pool pool(4);
    // ADM_THREADS == 0 leaves a serial no-op pool whose size() is 1, so the gate reads false and
    // the sweep below runs serially; the bit-identity it checks holds either way.
    REQUIRE(fsl_ws_engaged(&pool) == (ADM_THREADS != 0));
    REQUIRE(!fsl_ws_engaged(nullptr));
    constexpr std::size_t N = std::size_t{1} << 21;
    const auto in = make_input<double>(N, 0xA11);
    admiral::detail::four_step_large_plan<double> fsp(N, true);
    std::vector<std::complex<double>> buf(N + 16), off(N + 16);
    fsp.execute(in.data(), buf.data(), 1.0, &pool);
    fsp.execute(in.data(), off.data() + 1, 1.0, &pool);
    REQUIRE(std::memcmp(buf.data(), off.data() + 1, N * sizeof(std::complex<double>)) == 0);
    std::vector<std::complex<double>> st0(N + 16);
    fsp.execute(in.data(), st0.data(), 1.0, nullptr);
    for (std::size_t i = 0; i < N; ++i)
        REQUIRE(std::abs(st0[i] - buf[i]) <= 1e-9 * std::max(1.0, std::abs(buf[i])));
}

TEST_CASE("streaming transpose band matches the cached one bit for bit", "[large][fourstep]") {
    using admiral::detail::four_step_transpose_band;
    using admiral::detail::four_step_stream_ok;
    constexpr std::size_t A = xsimd::batch<double>::arch_type::alignment();
    constexpr std::size_t n1 = 64, n2 = 128;

    // The gate decides whether the streaming arm ever runs, so it is checked in both directions
    // before the kernel is. span_align is max(arch alignment, 64), so offset_buffer supplies an
    // arch-aligned base on every target and the positive control is not host-specific.
    const std::size_t huge = 64 * admiral::detail::cpu_cache().l3;
    offset_buffer<double> aligned(n1 * n2, 0);
    REQUIRE(reinterpret_cast<std::uintptr_t>(aligned.ptr) % A == 0);
    CHECK(four_step_stream_ok<double>(aligned.ptr, n2, huge));
    CHECK(!four_step_stream_ok<double>(aligned.ptr, n2, 0));

    // One T past the base, never dereferenced. Shifting by a whole complex would still be
    // arch-aligned wherever A is 16, so that control could not fail on SSE.
    const auto* shifted =
        reinterpret_cast<const std::complex<double>*>(reinterpret_cast<const double*>(aligned.ptr) + 1);
    CHECK(!four_step_stream_ok<double>(shifted, n2, huge));

    // A row pitch that is not a whole number of arch vectors. It can only fail where an arch
    // vector holds more than one complex, so it is asserted only there.
    if constexpr (xsimd::batch<double>::size > 2) {
        CHECK(!four_step_stream_ok<double>(aligned.ptr, n2 + 1, huge));
    }

    const auto in = make_input<double>(n1 * n2, 0x5EED);
    offset_buffer<double> cached(n1 * n2, 0), streamed(n1 * n2, 0);
    std::fill_n(cached.ptr, n1 * n2, std::complex<double>(-1.0, -1.0));
    std::fill_n(streamed.ptr, n1 * n2, std::complex<double>(-1.0, -1.0));
    four_step_transpose_band<double, false>(in.data(), cached.ptr, n1, n1, n2, 0, n2);
    four_step_transpose_band<double, true>(in.data(), streamed.ptr, n1, n1, n2, 0, n2);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    REQUIRE(std::memcmp(cached.ptr, streamed.ptr, n1 * n2 * sizeof(std::complex<double>)) == 0);

    // A transpose that wrote nothing would also compare equal, so one element is checked
    // against the source it came from.
    REQUIRE(streamed.ptr[1 * n2 + 0] == in[0 * n1 + 1]);
}

TEST_CASE("2^24 impulse against the analytic spectrum, aligned out-of-place",
          "[large][fourstep]") {
    // Coverage stopped at 2^23. Two 256 MiB buffers is the whole budget, so the reference is
    // recomputed per element instead of materialized.
    constexpr std::size_t N = std::size_t{1} << 24;
    const std::size_t n0 = N / 3 + 11;
    offset_buffer<double> in(N, 0), out(N, 0);
    std::fill_n(in.ptr, N, std::complex<double>(0.0, 0.0));
    in.ptr[n0] = {1.0, 0.0};

    admiral::plan<double> p(N, {std::size_t{1}, admiral::effort::estimate});
    REQUIRE(std::string(admiral::detail::plan_impl<double>(N, true).route_name())
            == "four_step_large");
    p.forward(in.ptr, out.ptr);

    // Whether the streaming arm ran depends on this host's L3, so the test says which path it
    // measured instead of assuming one. The check is not the byte line restated: it fails if
    // this buffer stops satisfying the arm's alignment condition while the byte line still asks
    // for it, which is how a silently non-streaming run would look.
    const auto split = admiral::detail::choose_fused_large_split<double>(N);
    REQUIRE(split.valid());
    const std::size_t bytes = N * sizeof(std::complex<double>);
    const std::size_t l3 = admiral::detail::cpu_cache().l3;
    const bool streamed = admiral::detail::four_step_stream_ok<double>(out.ptr, split.n2, bytes);
    std::cout << "[2^24 aligned out-of-place] " << bytes / (1024 * 1024) << " MiB, L3 "
              << l3 / (1024 * 1024) << " MiB, streaming transpose = " << streamed << '\n';
    CHECK(streamed == (bytes >= admiral::detail::kFourStepStreamL3Mult * l3));

    const double tol = fft_tol<double>();
    double worst = 0.0;
    std::size_t worst_k = 0;
    for (std::size_t k = 0; k < N; ++k) {
        const auto ref = std::conj(unit_phasor<double>(turn_fraction(n0, k, N)));
        const double e = std::abs(out.ptr[k] - ref);
        if (e > worst) { worst = e; worst_k = k; }
    }
    CAPTURE(worst_k, worst, tol, streamed);
    CHECK(worst <= tol);
}

TEST_CASE("WS-3 impulse flatness at 2^23, serial and auto", "[large][fourstep]") {
    constexpr std::size_t N = std::size_t{1} << 23;
    const std::size_t n0 = N / 5 + 13;
    std::vector<std::complex<double>> in(N, {0.0, 0.0}), out(N);
    in[n0] = {1.0, 0.0};
    std::vector<std::complex<double>> ref(N);
    for (std::size_t k = 0; k < N; ++k)
        ref[k] = std::conj(unit_phasor<double>(turn_fraction(n0, k, N)));
    for (const std::size_t nt : {std::size_t{1}, std::size_t{0}}) {
        CAPTURE(nt);
        admiral::plan<double> p(N, {nt, admiral::effort::estimate});
        p.forward(in.data(), out.data());
        require_close(out, ref, fft_tol<double>());
    }
}
