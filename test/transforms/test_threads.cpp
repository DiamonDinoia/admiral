#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.hpp>   // `admiral::plan`, `admiral::plan_r2c` (`nthreads` ctor param)
#include <admiral/detail/plan.hpp>         // `plan_impl::route_name` route pins
#include <admiral/detail/thread_pool.hpp>   // `parallel_for`'s exception contract

#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// If `ADM_THREADS`=0, the pool is a serial inline stub, so every `nthreads>1`
// expectation below is meaningless there. Guard the whole TU off. Catch2 discovery
// on an empty file exits 4, and the presets register exit 4 as a skip.
#if ADM_THREADS

// Multithreading correctness gate: `nthreads=1` (the tuned serial path) and
// `nthreads=4` must agree to the FFT's own rounding floor. Threading changes the
// ORDER in which a SIMD column/tile pass groups FMAs (chunk vs full sweep). Under
// `-ffast-math` the regrouping perturbs the last bit. Serial and threaded are two
// valid FFTs of the same data; the outputs differ by rounding, not by zero.
//
// Higham (Accuracy & Stability of Num. Algorithms, Thm 24.2): a Cooley-Tukey FFT
// satisfies ||fl(y)-y||_2 / ||y||_2 <= c*u*log2(N), u = eps/2. Both paths obey the
// bound, so ||threaded-serial||_2/||serial||_2 <= 2*c*u*log2(N). `forecast_tol()`
// below is C*u*log2(N). C covers 2c plus the twiddle/butterfly constants and stays
// ~1e5x tighter than any real race/chunk bug (O(1e-3)+).
//
// Every shape crosses the dispatch gate on the threaded path (`outer >= 2 &&
// total >= 1<<15`). A 1D {N} has one line, so the batch loop cannot thread and the
// axis sub-plan owns the pool. A small N then runs serial; a DRAM-bound N routes to
// `four_step_large` and threads its own passes ({1<<20}). {64,512} and {16,8192}
// thread the row and column-DIF passes. The outer prime axis of {67,512} exercises
// the scalar-fallback column pass. The 3D shape threads a middle axis. r2c adds the
// batched real tile loop.

namespace {

std::string shape_str(const std::vector<std::size_t>& s) {
    std::string r;
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "x" : "") + std::to_string(s[i]);
    return r;
}

constexpr std::size_t kNthreads = 4;

// Analytical serial-vs-threaded agreement bound: C*u*log2(N) (see the file header).
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
        {1 << 20},       // 1D f64 16 MB: `four_step_large` threads its own col/row passes
        {1 << 21},       // 1D f64 32 MB: `four_step_large` defer split (n2 = 2*n1, m=2 panels)
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
        {512, 513},      // odd innermost real axis -> threads the `r2c_odd`/`c2r_odd` row loop
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

        // `c2r` consumes the complex input, so feed each plan a private copy.
        auto ca_in = ca, cb_in = cb;
        std::vector<T> ra(serial.real_size()), rb(serial.real_size());
        serial.inverse(ca_in.data(), ra.data());
        threaded.inverse(cb_in.data(), rb.data());
        REQUIRE(relerrtwonorm(ra, rb) < tol);
    }
}

// `nthreads` = 0 resolves to the allowed physical cores at the ctor boundary. The
// auto path must not crash, and it must not change the result vs the serial path
// (to the FFT rounding floor; see the file header). On a single-core host the
// resolution is 1 (serial).
TEMPLATE_TEST_CASE("nthreads=0 auto-select matches serial", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape = {16, 8192};
    const admiral::span<const std::size_t> sp(shape.data(), shape.size());
    const auto in = make_input<T>(16 * 8192, 0xA705u);

    admiral::plan<T> serial(sp);
    admiral::plan<T> autop(sp, {0});   // 0 -> allowed physical cores

    auto a = in, b = in;
    serial.forward(a.data());
    autop.forward(b.data());
    REQUIRE(relerrtwonorm(a, b) < forecast_tol<T>(16 * 8192));
}

// `parallel_for`'s documented contract: "First exception wins; rethrown after join."
// No FFT body throws, so the contract is unreachable through the public plan API and
// needs the pool directly. The path is live error handling, not dead code. Every
// `four_step_large` and `real_fft` body constructs a `soa_scratch`, and past
// `SBO_MAX` the scratch calls `::operator new[]` and can throw `bad_alloc` on a
// WORKER thread. Uncaught there, the exception escapes the thread's callable and
// terminates the process. n=64 over 4 threads gives chunk 16, so the throwing chunk
// (`begin==0`) belongs to a worker and the caller runs the last chunk. The
// chunk-to-thread assignment runs the capture-and-rethrow path, not a plain local
// throw. The test also holds for
// `ADM_THREADS=0`, where `parallel_for` runs the body inline and the exception
// propagates directly.
TEST_CASE("parallel_for propagates a body exception, then resets", "[threads]") {
    admiral::detail::thread_pool pool(4);

    std::atomic<int> ran{0};
    REQUIRE_THROWS_AS(pool.parallel_for(64, [&](std::size_t b, std::size_t, std::size_t) {
        ++ran;
        if (b == 0) throw std::runtime_error("chunk failed");
    }), std::runtime_error);
    REQUIRE(ran.load() > 0);

    // The captured exception must not leak into the next call; one failed transform
    // would poison every later transform on the same plan.
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

// 810000 = 900^2 has `n2 % n1 == 0` but `900 % W != 0`. The serial gate refuses the
// size and the threaded gate admits the size, so the pool runs the unfused sweeps.
// The unfused sweeps are executable only in this threaded configuration.
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

// The auto count is the wake-law argmin (fi/mt t0-modeler-r2.md, 2026-08-31):
//   nt* = argmin over pow2 nt <= P of  W / min(nt, knee[cls]) + K * Dhat(nt, gap^),
// W the serial work estimate, K the dispatches per execute, Dhat the probed
// wake-cost grid. This case pins the law's named regimes against hand-computed
// values; the machine-dependent rows come from the probed tables in
// `wake_family_row`, so the same checks hold on any host.
TEST_CASE("resolve_nthreads wake law: serial floor, knee, pocket, pow2, cap", "[threads]") {
    using admiral::detail::dhat_ns;
    using admiral::detail::has_single_bit;
    using admiral::detail::kAutoSerialElems;
    using admiral::detail::kPocketOnsetNs;
    using admiral::detail::kPocketTierNs;
    using admiral::detail::resolve_nthreads;
    using admiral::detail::wake_family_for;
    using admiral::detail::wake_family_row;

    const std::size_t P = resolve_nthreads(0);
    const std::size_t C0 = admiral::detail::cores_per_socket();
    REQUIRE(P >= 1);
    REQUIRE(C0 >= 1);
    const wake_family_row& fam = wake_family_for(P, C0);

    // Serial floor: below 2^15 elements, or K == 0, there is never a pool.
    REQUIRE(resolve_nthreads(0, kAutoSerialElems - 1, 3, 1e9, 1) == 1);
    REQUIRE(resolve_nthreads(0, std::size_t{1} << 30, 0, 1e9, 1) == 1);

    // An explicit count is still returned verbatim.
    REQUIRE(resolve_nthreads(3) == 3);
    REQUIRE(resolve_nthreads(3, 1) == 3);
    REQUIRE(resolve_nthreads(3, kAutoSerialElems - 1, 0, 0, 0) == 3);

    // Quantization and cap over the whole (size, K, class, work) box: pow2-or-1
    // and never past the machine count.
    for (std::size_t lg = 15; lg < 34; ++lg)
        for (const std::size_t K : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{5}})
            for (const unsigned cls : {0u, 1u, 2u})
                for (const double w : {1e4, 1e7, 1e12}) {
                    const std::size_t nt =
                        resolve_nthreads(0, std::size_t{1} << lg, K, w, cls);
                    REQUIRE(nt <= P);
                    REQUIRE((nt == 1 || has_single_bit(nt)));
                }

    // Knee clamp: work that dwarfs every dhat entry buys the saturating width and
    // no more. Below the knee each doubling halves the work; at and past it the
    // work is flat and the dhat term only grows. 2^30 f64 elements (16 GB) is past
    // every row's gateMB, so the deep tier is read.
    for (const unsigned cls : {0u, 1u, 2u}) {
        const std::size_t knee = fam.knee[cls][1];
        const std::size_t want = knee <= P ? knee : std::size_t{1}
                                   << (admiral::detail::bit_width(P) - 1);
        REQUIRE(resolve_nthreads(0, std::size_t{1} << 30, 1, 1e12, cls) == want);
    }

    // Hand-computed eval: re-derive T(nt) from the probed table with the pocket
    // rider and the two fixed-point gap iterations, then require the shipped
    // argmin to match. Catches a mis-fed K/W/class, the loop bounds and the
    // quantization; a wrong pocket or knee constant lands a different argmin.
    const auto law_eval = [&](std::size_t total, std::size_t K, double w_ns, unsigned cls) {
        const bool deep = double(total) * 16.0 > double(fam.gateMB[cls]) * 1e6;
        const std::size_t knee = fam.knee[cls][deep];   // hand-mirrors the shipped gate
        std::size_t best = 1;
        double best_t = w_ns;
        for (std::size_t nt = 2; nt <= P; nt <<= 1) {
            const double work = w_ns / double(std::min(nt, knee));
            const double gh = work / double(K);
            const bool pocket = gh < kPocketOnsetNs && nt >= C0 / 2;
            double gap = pocket ? std::max(gh, kPocketTierNs) : gh;
            double dh = dhat_ns(fam, nt, gap);
            for (int i = 0; i < 2; ++i) {
                const double t = work + double(K) * dh;
                gap = t / double(K);
                if (pocket) gap = std::max(gap, kPocketTierNs);
                dh = dhat_ns(fam, nt, gap);
            }
            if (const double t = work + double(K) * dh; t < best_t) { best_t = t; best = nt; }
        }
        return best;
    };
    // A pocket-positive cell (hot gap < 30 us at >= half a socket), a parked cell
    // (work per dispatch in the ms range) and a near-serial cell.
    for (const auto& c : {std::size_t{5}, std::size_t{1}, std::size_t{2}})
        for (const unsigned cls : {0u, 1u, 2u})
            for (const double w : {1e5, 9e6, 3e9}) {
                INFO("K=" << c << " cls=" << cls << " w=" << w);
                REQUIRE(resolve_nthreads(0, std::size_t{1} << 24, c, w, cls) ==
                        law_eval(std::size_t{1} << 24, c, w, cls));
            }

    // Exact picks on the probed host classes (fi/mt t0-modeler-r3.md receipt table,
    // t0/scoring-final-r3.txt), one per knee regime. W is priced the way the plans
    // price it: fsl pays kFourStepOverhead over the large split, ND sums the
    // per-axis line work, both at the shelves' core frequency. Hand-verified
    // arithmetics (ns, two fixed-point gap iterations):
    // * shallow regime, rome 2-D 2048^2 (64 MB <= 256 MB gate -> knee 16):
    //   T(16) = 1.054e6 + 2*17.1e3 = 1.088e6 < T(32) = 1.054e6 + 2*31.0e3 = 1.116e6 -> 16.
    // * deep regime, rome 1-D 2^22 (67.1 MB > 64 MB gate -> knee 32):
    //   T(32) = 5.96e5 + 5*23.0e3 = 7.11e5 < T(16) = 1.19e6 + 5*13.0e3 = 1.26e6 -> 32.
    // * gate-inert row, ice 1-D 2^21 (ice fsl layer is (32,32)):
    //   T(16) = 5.28e5 + 5*211.5e3 = 1.585e6 < T(32) = 2.64e5 + 5*298.1e3 = 1.754e6 -> 16.
    const auto fsl_pick = [&](std::size_t lg) {
        const std::size_t n = std::size_t{1} << lg;
        const auto sp = admiral::detail::choose_large_split(n);
        const double cyc =
            sp.valid() ? admiral::detail::kFourStepOverhead *
                             (double(sp.n1) * admiral::detail::line_work_cyc<double>(sp.n2) +
                              double(sp.n2) * admiral::detail::line_work_cyc<double>(sp.n1))
                       : admiral::detail::line_work_cyc<double>(n);
        return resolve_nthreads(0, n, 5, cyc / admiral::detail::core_cyc_per_ns(), 0);
    };
    const auto sq2_pick = [&](std::size_t e) {   // 2-D: K = 2, cls = 1, summed line work
        const std::size_t n = std::size_t{1} << e;
        const double cyc = 2.0 * double(n) * admiral::detail::line_work_cyc<double>(n);
        return resolve_nthreads(0, n * n, 2, cyc / admiral::detail::core_cyc_per_ns(), 1);
    };
    if (P == 128 && C0 == 64) {          // rome: 2^21 stays 16 (shallow), 2^22 moves to 32 (deep)
        REQUIRE(fsl_pick(21) == 16);
        REQUIRE(fsl_pick(22) == 32);
        REQUIRE(sq2_pick(11) == 16);     // 2048^2, the shallow regime above
        REQUIRE(sq2_pick(13) == 32);     // 8192^2: accepted +6.5% loss (r3 sect. 4 item 3)
    }
    if (P == 64 && C0 == 32) {           // icelake: 2^20 -> 8, 2^21 -> 16
        REQUIRE(fsl_pick(20) == 8);
        REQUIRE(fsl_pick(21) == 16);
        REQUIRE(sq2_pick(11) == 32);     // 2048^2 (deep tier, dhat-dominated)
        REQUIRE(sq2_pick(12) == 64);     // 4096^2
    }
    if (P == 96 && C0 == 48) {           // genoa: 2^20 -> 16, 2^21 -> 32
        REQUIRE(fsl_pick(20) == 16);
        REQUIRE(fsl_pick(21) == 32);
        REQUIRE(sq2_pick(10) == 32);     // 1024^2
        REQUIRE(sq2_pick(13) == 64);     // 8192^2 (deep gate)
    }
}
#endif  // ADM_THREADS
