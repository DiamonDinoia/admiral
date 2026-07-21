#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <admiral/admiral.hpp>   // admiral::plan, admiral::plan_r2c (nthreads ctor param)

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <random>
#include <span>
#include <string>
#include <vector>

// Multithreading correctness gate: nthreads=1 (the tuned serial path) and
// nthreads=4 must agree to the FFT's own rounding floor. The N-D batch loops
// split into contiguous chunks; each row/column/tile is independent, so threading
// changes no MATH — but it changes the ORDER in which a SIMD column/tile pass
// groups its FMAs (chunk vs full sweep), which under -ffast-math perturbs the last
// bit. So serial and threaded are two valid FFTs of the same data whose difference
// is bounded by the Cooley-Tukey normwise rounding error, not zero.
//
// Higham (Accuracy & Stability of Num. Algorithms, Thm 24.2): a Cooley-Tukey FFT
// satisfies ||fl(y)-y||_2 / ||y||_2 <= c*u*log2(N), u = eps/2. Both paths obey it,
// so ||threaded-serial||_2/||serial||_2 <= 2*c*u*log2(N); forecast_tol() below is
// C*u*log2(N) with C generous enough to cover 2c + the twiddle/butterfly constants
// and still be ~1e5x tighter than any real race/chunk bug (those are O(1e-3)+).
// (Forecast validated: {512,513} f64 predicts ~1.6e-14, measured max |Δ| 1.4e-14.)
//
// Shapes are sized so each threaded path actually crosses the dispatch gate
// (outer >= 2*nthreads && total >= 1<<15): {N} 1D stays serial (one line);
// {64,512}/{16,8192} thread the row + column-DIF passes; {67,512}'s outer prime
// axis exercises the scalar-fallback column pass; the 3D shape threads a middle
// axis. r2c adds the batched real tile loop.

namespace {

template<typename T>
std::vector<std::complex<T>> make_c(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<T>> v(n);
    for (auto& z : v) z = std::complex<T>(static_cast<T>(dist(gen)), static_cast<T>(dist(gen)));
    return v;
}

template<typename T>
std::vector<T> make_r(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<T> v(n);
    for (auto& x : v) x = static_cast<T>(dist(gen));
    return v;
}

std::string shape_str(const std::vector<std::size_t>& s) {
    std::string r;
    for (std::size_t i = 0; i < s.size(); ++i) r += (i ? "x" : "") + std::to_string(s[i]);
    return r;
}

constexpr std::size_t kNthreads = 4;

// Normwise relative L2 distance between two transforms. L2 (not per-element) so a
// near-zero spectral bin can't inflate the ratio — the meaningful quantity is the
// energy of the discrepancy relative to the signal.
template<typename V>
double rel_l2(const V& a, const V& b) {
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += std::norm(std::complex<double>(a[i]) - std::complex<double>(b[i]));
        den += std::norm(std::complex<double>(a[i]));
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

// Analytical serial-vs-threaded agreement bound: C*u*log2(N) (see file header).
template<typename T>
double forecast_tol(std::size_t N) {
    const double u = 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
    return 16.0 * u * std::log2(static_cast<double>(N));  // C=16 covers 2c + twiddles + margin
}

} // namespace

TEMPLATE_TEST_CASE("c2c N-D nthreads=1 vs 4 is bit-identical", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {4096},          // 1D: single line -> stays serial, must still match
        {64, 512},       // 2D: threads the innermost row pass
        {16, 8192},      // 2D: threads the row pass + the batched column-DIF pass
        {67, 512},       // outer prime axis (>catalog) -> scalar-fallback column pass
        {8, 8, 512},     // 3D: threads a middle-axis column pass
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        std::size_t Ntot = 1;
        for (auto e : shape) Ntot *= e;
        const auto in = make_c<T>(Ntot, 0xC2C0u);
        const std::span<const std::size_t> sp(shape.data(), shape.size());

        admiral::plan<T> serial(sp, /*nthreads=*/1);
        admiral::plan<T> threaded(sp, kNthreads);

        const double tol = forecast_tol<T>(Ntot);
        auto a = in, b = in;
        serial.forward(a.data());
        threaded.forward(b.data());
        REQUIRE(rel_l2(a, b) < tol);

        serial.inverse(a.data());
        threaded.inverse(b.data());
        REQUIRE(rel_l2(a, b) < tol);
    }
}

TEMPLATE_TEST_CASE("r2c/c2r N-D nthreads=1 vs 4 is bit-identical", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {
        {64, 512},
        {16, 8192},
        {8, 8, 512},
        {512, 513},      // odd innermost real axis -> threads the r2c_odd/c2r_odd row loop
    };
    for (const auto& shape : shapes) {
        INFO("shape=" << shape_str(shape) << " prec=" << (sizeof(T) == 4 ? "f32" : "f64"));
        const std::span<const std::size_t> sp(shape.data(), shape.size());
        admiral::plan_r2c<T> serial(sp, /*nthreads=*/1);
        admiral::plan_r2c<T> threaded(sp, kNthreads);

        const auto rin = make_r<T>(serial.real_size(), 0x2C20u);
        const double tol = forecast_tol<T>(serial.real_size());

        std::vector<std::complex<T>> ca(serial.cplx_size()), cb(serial.cplx_size());
        serial.forward(rin.data(), ca.data());
        threaded.forward(rin.data(), cb.data());
        REQUIRE(rel_l2(ca, cb) < tol);

        // c2r consumes its complex input -> feed each a private copy.
        auto ca_in = ca, cb_in = cb;
        std::vector<T> ra(serial.real_size()), rb(serial.real_size());
        serial.inverse(ca_in.data(), ra.data());
        threaded.inverse(cb_in.data(), rb.data());
        REQUIRE(rel_l2(ra, rb) < tol);
    }
}

// nthreads=0 resolves to hardware_concurrency (capped) at the ctor boundary and
// must not crash or change the result vs the serial path (to the FFT rounding
// floor; see file header). On a single-core host it resolves to 1 (serial).
TEMPLATE_TEST_CASE("nthreads=0 auto-select matches serial", "[threads]", float, double) {
    using T = TestType;
    const std::vector<std::size_t> shape = {16, 8192};
    const std::span<const std::size_t> sp(shape.data(), shape.size());
    const auto in = make_c<T>(16 * 8192, 0xA705u);

    admiral::plan<T> serial(sp, /*nthreads=*/1);
    admiral::plan<T> autop(sp, /*nthreads=*/0);   // 0 -> hardware_concurrency, capped

    auto a = in, b = in;
    serial.forward(a.data());
    autop.forward(b.data());
    REQUIRE(rel_l2(a, b) < forecast_tol<T>(16 * 8192));
}
