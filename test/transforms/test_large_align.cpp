// Coverage for the large-N four_step_large route (f64 above ~786k points) and for
// pointer-alignment variants of the public execute() path.
//
// Two gaps this closes:
//   1. four_step_large: forward against an analytical single-tone reference (exact
//      at machine precision for any N), plus round-trip identity, across the
//      L3-transpose-fuse boundary and a non-power-of-two unbalanced split.
//   2. Unaligned INPUT and OUTPUT buffers through the store-align peel in
//      dif_col_pass_last / four_step_large: user data is not cache-line aligned,
//      so every (in,out) alignment pairing runs.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/four_step_large.hpp>   // four_step_transpose_inplace, tested directly
#include <admiral/detail/plan.hpp>   // plan_impl::route_name: keeps the fsl cells honest
#include <admiral/detail/scratch.hpp>  // span_align: the alignment the kernels assume
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <vector>

namespace {

// Analytical forward reference: input is a single complex exponential at bin K.
// admiral::forward is X[k] = sum_n x[n]*exp(-2*pi*i*k*n/N), unscaled.
// For x[n] = exp(+2*pi*i*K*n/N), the DFT is exactly N at k==K and 0 elsewhere.
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

// Owns an over-aligned buffer and hands out a pointer offset `off_elems` complex
// elements past a span_align boundary. span_align rather than a literal 64
// because that is the alignment the kernels actually assume, so off_elems=0 is
// the aligned case on every ISA. off_elems in {1,2,3} => 16/32/48-byte misaligned
// (complex<double> is 16 B), exercising the store-align peel's masked windows.
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

// four_step_large routes for f64 when N*sizeof(complex<double>) > 12 MiB
// (N > 786432). The six-step engine's in-place transpose decomposes by split
// shape: square (1M, 4M), block-grid at C = 2*R (2M), element cycles otherwise.
// Every size below must pass the public serial gate (four_step_large_fused_shape)
// or the case is vacuously iterative_dif; the route is pinned in each case.
constexpr std::size_t kBelowFuse = 1048576;   // 16 MB, square split 1024x1024
constexpr std::size_t kRect2M    = 2097152;   // 32 MB, block split 1024x2048
constexpr std::size_t kNonPow2   = 2073600;   // 33 MB, non-pow2 1440x1440
constexpr std::size_t kAboveFuse = 8294400;   // 132 MB, non-pow2 2880x2880, past L3

} // namespace

TEST_CASE("four_step_large forward vs analytical (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kRect2M, kNonPow2}) {
        CAPTURE(N);
        // Pin the route here too, else this "fsl" case could silently become a dif test.
        CHECK(std::string(admiral::detail::plan_impl<double>(N, true).route_name())
              == "four_step_large");
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<double>(N, K);
        const auto ref = tone_spectrum<double>(N, K);

        auto plan = admiral::plan<double>(N);
        auto out = in;
        plan.forward(std::span(out));

        require_close(out, ref, fft_tol<double>());
    }
}

TEST_CASE("four_step_large round-trip identity (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kRect2M, kNonPow2, kAboveFuse}) {
        CAPTURE(N);
        // Pin the route, else this "fsl" case could silently become a dif test.
        CHECK(std::string(admiral::detail::plan_impl<double>(N, true).route_name())
              == "four_step_large");
        const auto in = make_input<double>(N, 77u + unsigned(N));
        auto plan = admiral::plan<double>(N);

        auto data = in;
        plan.forward(std::span(data));
        plan.inverse(std::span(data));

        require_close(data, in, fft_tol<double>());
    }
}

// Impulse at n0: X[k] = exp(-2*pi*i*k*n0/N), flat |X| == 1. The OOP (two-pointer)
// path runs P1's band transpose, and a nonzero n0 pins the split-twist's phases:
// a twist applied on the wrong side of the n2 DFT still round-trips but fails here.
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

// Band-fused P4+P5 shapes: impulse known-answer in every (direction, placement)
// pairing, at the square 4M split (2048x2048) and the defer 2M split
// (1024x2048). A nonzero n0 pins phase across the fused sweep's band order;
// in-place runs the whole fused engine on one buffer.
TEST_CASE("four_step_large fused-band impulse (double)", "[large][fourstep]") {
    for (const std::size_t N : {kRect2M, std::size_t{4194304}}) {
        CAPTURE(N);
        const std::size_t n0 = N / 8 + 97;   // crosses tile-band boundaries
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
        plan.forward(std::span(ip));
        require_close(ip, fwd_ref, fft_tol<double>());
        ip = in;
        plan.inverse(std::span(ip));
        require_close(ip, inv_ref, fft_tol<double>());
    }
}

// f32 direction-sensitive absolute oracle (round-trip alone is blind to an f32-width
// sweep-shape bug). Serial 4M/2M sit inside the f32 serial window (16 MiB line, 32 MiB
// bound); the threaded 1M cell crosses the line every nthreads > 1 shares.
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
        auto plan = admiral::plan<float>({N}, {.nthreads = nt});
        std::vector<std::complex<float>> out(N);
        plan.forward(in.data(), out.data());
        require_close(out, ref, fft_tol<float>(64));

        // Impulse: exact analytic phasor answer at every bin, inverse direction.
        std::vector<std::complex<float>> imp(N, {0.0f, 0.0f}), got(N), iref(N);
        imp[N / 8 + 97] = {1.0f, 0.0f};
        plan.inverse(imp.data(), got.data());
        for (std::size_t k = 0; k < N; ++k)
            iref[k] = unit_phasor<float>(turn_fraction(N / 8 + 97, k, N)) / float(N);
        require_close(got, iref, fft_tol<float>(64));
    }
}

// Every (input, output) cache-line-alignment pairing through the OOP execute
// path. Covers both a four_step_large size (store-align peel in the col pass)
// and an iterative_dif size (peel in dif_col_pass_last). Forward validated
// against the analytical tone reference so a mis-peeled store is caught, not
// just a self-consistent one.
TEMPLATE_TEST_CASE("execute() input/output alignment variants vs analytical",
                   "[large][align]", float, double) {
    using T = TestType;
    // four_step_large is f64-only; for f32 this size routes iterative_dif, still
    // a valid alignment probe of the shared peel.
    for (const std::size_t N : {std::size_t{4096}, std::size_t{1048576}}) {
        CAPTURE(N);
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<T>(N, K);
        const auto ref = tone_spectrum<T>(N, K);
        const double tol = fft_tol<T>();
        auto plan = admiral::plan<T>(N);

        // off=0 aligned, off=1 misaligned by one complex (16 B for f64 / 8 B for f32).
        for (const std::size_t in_off : {std::size_t{0}, std::size_t{1}}) {
            for (const std::size_t out_off : {std::size_t{0}, std::size_t{1}}) {
                CAPTURE(in_off, out_off);
                offset_buffer<T> src(N, in_off);
                offset_buffer<T> dst(N, out_off);
                std::copy(in.begin(), in.end(), src.ptr);

                plan.forward(src.ptr, dst.ptr);

                std::vector<std::complex<T>> got(dst.ptr, dst.ptr + N);
                require_close(got, ref, tol);

                // src must be untouched by the OOP path regardless of alignment.
                for (std::size_t i = 0; i < N; ++i) REQUIRE(src.ptr[i] == in[i]);
            }
        }
    }
}

// The out-aliased SoA pair: dif_execute_in_place lends `out` to the driver as the second
// ping-pong pair when the tape's parity leaves it dead, halving the scratch. The control arm
// needs no build flag: a span_align-misaligned `out` fails dif_out_aliasable's alignment
// clause and takes the 4-plane fallback, which also pins that clause. That arm is compared
// to tolerance, not bitwise: the same misalignment also makes the last pass take its AoS
// store peel, whose partial-row block groups the same values differently. Bitwise still
// holds where alignment matches, the in == out arm below. `fired` guards the whole case
// against passing vacuously if the parity predicate ever stops admitting these sizes.
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

            // in == out is the shape the alias actually ships into: bluestein's and rader's
            // inner DIF both call dif_execute_in_place(buf, buf, ...) on soa_scratch memory,
            // which is aligned, so they take the aliased path on every qualifying chain.
            // It is also the only case where lending `out` out is not clearly safe: pass 1
            // writes the lent pair, and that pair IS the input. Legal because pass 0 fully
            // drains `in` before pass 1 runs, but "legal by argument" is what a test is for.
            offset_buffer<T> ip(N, 0);
            std::copy(in.begin(), in.end(), ip.ptr);
            admiral::detail::dif_execute_in_place<T>(forward, ip.ptr, ip.ptr, N, tw, T(1));
            for (std::size_t i = 0; i < N; ++i) REQUIRE(ip.ptr[i] == aligned.ptr[i]);
        }
    }
    CAPTURE(fired);
    REQUIRE(fired > 0);
}

// four_step_transpose_inplace over all four shape branches, including the
// non-divisible one (C % R != 0 and R % C != 0) that the public gate currently
// rejects and no route therefore reaches. That branch is a whole separate
// kernel; without this it is untested code.
TEMPLATE_TEST_CASE("four_step_transpose_inplace vs naive", "[large][fourstep]", float, double) {
    using T = TestType;
    // square, C = m*R, R = m*C, and three non-divisible shapes (both < and >
    // the vector width, so the band remainder is exercised on each axis).
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
