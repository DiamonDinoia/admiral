#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>   // admiral::plan, admiral::plan_r2c (nthreads ctor param)
#include <admiral/detail/plan.hpp>         // plan_impl::route_name route pins
#include <admiral/detail/thread_pool.hpp>   // parallel_for's exception contract

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// With ADM_THREADS=0 the pool is a serial inline stub, so every nthreads>1
// expectation below is meaningless there. Guard the whole TU off; Catch2's
// discovery on an empty file exits 4, which the presets register as skip.
#if ADM_THREADS

// Multithreading correctness gate: nthreads=1 (the tuned serial path) and
// nthreads=4 must agree to the FFT's own rounding floor. Threading changes the
// ORDER in which a SIMD column/tile pass groups its FMAs (chunk vs full sweep),
// which under -ffast-math perturbs the last bit: serial and threaded are two valid
// FFTs of the same data, differing by rounding, not zero.
//
// Higham (Accuracy & Stability of Num. Algorithms, Thm 24.2): a Cooley-Tukey FFT
// satisfies ||fl(y)-y||_2 / ||y||_2 <= c*u*log2(N), u = eps/2. Both paths obey it,
// so ||threaded-serial||_2/||serial||_2 <= 2*c*u*log2(N); forecast_tol() below is
// C*u*log2(N) with C covering 2c plus twiddle/butterfly constants and still ~1e5x
// tighter than any real race/chunk bug (those are O(1e-3)+).
//
// Shapes are sized so each threaded path crosses the dispatch gate (outer >= 2 &&
// total >= 1<<15). A 1D {N} has one line, so the batch loop cannot thread and the
// axis sub-plan owns the pool: small N runs serial, a DRAM-bound N routes to
// four_step_large and threads its own passes ({1<<20}); {64,512}/{16,8192} thread
// the row + column-DIF passes; {67,512}'s outer prime axis exercises the
// scalar-fallback column pass; the 3D shape threads a middle axis. r2c adds the
// batched real tile loop.

namespace {

std::string shape_str(const std::vector<std::size_t>& s) {
    std::string r;
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "x" : "") + std::to_string(s[i]);
    return r;
}

constexpr std::size_t kNthreads = 4;

// Analytical serial-vs-threaded agreement bound: C*u*log2(N) (see file header).
template<typename T>
double forecast_tol(std::size_t N) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    return 16.0 * u * std::log2(static_cast<double>(N));  // C=16 covers 2c + twiddles + margin
}

} // namespace

TEMPLATE_TEST_CASE("c2c N-D nthreads=1 vs 4 agrees within the FFT rounding floor", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {4096},          // 1D: single line, below the DRAM route -> serial, must still match
        {1 << 20},       // 1D f64 16 MB: four_step_large, threads its own col/row passes
        {1 << 21},       // 1D f64 32 MB: four_step_large defer split (n2 = 2*n1, m=2 panels)
        {64, 512},       // 2D: threads the innermost row pass
        {16, 8192},      // 2D: threads the row pass + the batched column-DIF pass
        {67, 512},       // outer prime axis (>catalog) -> scalar-fallback column pass
        {8, 8, 512},     // 3D: threads a middle-axis column pass
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        std::size_t Ntot = 1;
        for (auto e : shape) Ntot *= e;
        const auto in = make_input<T>(Ntot, 0xC2C0u);
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());

        admiral::plan<T> serial(sp);
        admiral::plan<T> threaded(sp, {kNthreads});

        const double tol = forecast_tol<T>(Ntot);
        auto a = in, b = in;
        serial.forward(a.data());
        threaded.forward(b.data());
        REQUIRE(relerrtwonorm(a, b) < tol);

        serial.inverse(a.data());
        threaded.inverse(b.data());
        REQUIRE(relerrtwonorm(a, b) < tol);
    }
}

TEMPLATE_TEST_CASE("r2c/c2r N-D nthreads=1 vs 4 agrees within the FFT rounding floor", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {64, 512},
        {16, 8192},
        {8, 8, 512},
        {512, 513},      // odd innermost real axis -> threads the r2c_odd/c2r_odd row loop
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());
        admiral::plan_r2c<T> serial(sp);
        admiral::plan_r2c<T> threaded(sp, {kNthreads});

        const auto rin = make_real_input<T>(serial.real_size(), 0x2C20u);
        const double tol = forecast_tol<T>(serial.real_size());

        std::vector<std::complex<T>> ca(serial.cplx_size()), cb(serial.cplx_size());
        serial.forward(rin.data(), ca.data());
        threaded.forward(rin.data(), cb.data());
        REQUIRE(relerrtwonorm(ca, cb) < tol);

        // c2r consumes its complex input -> feed each a private copy.
        auto ca_in = ca, cb_in = cb;
        std::vector<T> ra(serial.real_size()), rb(serial.real_size());
        serial.inverse(ca_in.data(), ra.data());
        threaded.inverse(cb_in.data(), rb.data());
        REQUIRE(relerrtwonorm(ra, rb) < tol);
    }
}

// nthreads=0 resolves to hardware_concurrency (capped) at the ctor boundary and
// must not crash or change the result vs the serial path (to the FFT rounding
// floor; see file header). On a single-core host it resolves to 1 (serial).
TEMPLATE_TEST_CASE("nthreads=0 auto-select matches serial", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape = {16, 8192};
    const admiral::span<const std::size_t> sp(shape.data(), shape.size());
    const auto in = make_input<T>(16 * 8192, 0xA705u);

    admiral::plan<T> serial(sp);
    admiral::plan<T> autop(sp, {0});   // 0 -> hardware_concurrency, capped

    auto a = in, b = in;
    serial.forward(a.data());
    autop.forward(b.data());
    REQUIRE(relerrtwonorm(a, b) < forecast_tol<T>(16 * 8192));
}

// parallel_for's documented contract: "First exception wins; rethrown after join."
// No FFT body throws, so this is unreachable through the public plan API and needs
// the pool directly. It is live error handling and not dead code. Every
// four_step_large and real_fft body constructs a soa_scratch, which past SBO_MAX
// calls ::operator new[] and can therefore throw bad_alloc on a WORKER thread.
// Uncaught, that escapes the thread's callable and terminates the process.
// n=64 over 4 threads gives chunk 16, so the throwing chunk (begin==0) belongs to a
// worker while the caller runs the last one, i.e. the capture-and-rethrow path,
// not a plain local throw. Holds for ADM_THREADS=0 too, where parallel_for runs the
// body inline and the exception propagates directly.
TEST_CASE("parallel_for propagates a body exception, then resets", "[threads]") {
    admiral::detail::thread_pool pool(4);

    std::atomic<int> ran{0};
    REQUIRE_THROWS_AS(pool.parallel_for(64, [&](std::size_t b, std::size_t, std::size_t) {
        ++ran;
        if (b == 0) throw std::runtime_error("chunk failed");
    }), std::runtime_error);
    REQUIRE(ran.load() > 0);

    // The captured exception must not leak into the next call, or one failed
    // transform would poison every later one on the same plan.
    std::atomic<std::size_t> sum{0};
    REQUIRE_NOTHROW(pool.parallel_for(64, [&](std::size_t b, std::size_t e, std::size_t) {
        for (std::size_t i = b; i < e; ++i) sum += i;
    }));
    REQUIRE(sum.load() == std::size_t{64 * 63 / 2});
}

TEST_CASE("parallel_for with fewer units than threads leaves workers idle", "[threads]") {
    admiral::detail::thread_pool pool(4);
    // n=2 over 4 threads: two of the four chunks are empty and their bodies must
    // not run (the empty-chunk arm keeps a 2-unit sweep correct).
    std::atomic<int> ran{0};
    std::atomic<std::size_t> sum{0};
    pool.parallel_for(2, [&](std::size_t b, std::size_t e, std::size_t) {
        ++ran;
        for (std::size_t i = b; i < e; ++i) sum += i;
    });
    REQUIRE(ran.load() == 2);
    REQUIRE(sum.load() == 1);
}

// 810000 = 900^2 has n2 % n1 == 0 but 900 % W != 0. The serial gate refuses it and
// the threaded gate admits it, so the pool runs the unfused sweeps, executable
// only in this configuration.
TEST_CASE("threaded unfused four_step_large agrees across nthreads (double)",
          "[threads][fourstep]") {
    constexpr std::size_t N = 810000;
    REQUIRE(std::string(admiral::detail::plan_impl<double>(N, true, 4).route_name())
            == "four_step_large");
    const auto in = make_input<double>(N, 1234u);
    admiral::plan<double> p2({N}, {2}), p4({N}, {4});
    auto a = in;
    p2.forward(admiral::span(a));
    auto b = in;
    p4.forward(admiral::span(b));
    require_close(b, a, forecast_tol<double>(N));
    p4.inverse(admiral::span(b));
    require_close(b, in, forecast_tol<double>(N));
}

// Out-of-place executes share the pool with the in-place path; 2M is a fused
// defer split whose band transpose is the only OOP-specific piece.
TEST_CASE("threaded out-of-place four_step_large matches serial (double)",
          "[threads][fourstep]") {
    constexpr std::size_t N = 2097152;
    const auto in = make_input<double>(N, 77u);
    admiral::plan<double> p1({N}), p4({N}, {4});
    std::vector<std::complex<double>> o1(N), o4(N);
    p1.forward(in.data(), o1.data());
    p4.forward(in.data(), o4.data());
    require_close(o4, o1, forecast_tol<double>(N));
}
#endif  // ADM_THREADS
