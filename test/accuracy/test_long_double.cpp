// The long double backend (detail/scalar_fft.hpp): plan and plan_r2c against
// reference_dft and against round-trips, across smooth, prime and Bluestein
// sizes. Bounds stay within fft_tol<long double>: the backend compensates its
// accumulations, and the reference itself rounds at ld precision.
#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/scalar_fft.hpp>  // scalar_twiddle

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <vector>

namespace {

using lt_t = long double;
using lc_t = std::complex<lt_t>;

// relerrtwonorm measures in double, whose floor (~2e-16) is far above the
// long double engine's error (~few ld-ulps); accumulate in long double here.
template<typename V>
long double relerr_ld(const std::vector<V>& ref, const std::vector<V>& got) {
    REQUIRE(ref.size() == got.size());
    const auto norm = [](const V& v) {
        if constexpr (std::is_same_v<V, lc_t>) return std::norm(v);
        else return v * v;
    };
    long double err = 0, nrm = 0;
    for (std::size_t m = 0; m < ref.size(); ++m) {
        err += norm(ref[m] - got[m]);
        nrm += norm(ref[m]);
    }
    return nrm == 0 ? (err == 0 ? 0.0L : std::numeric_limits<long double>::infinity())
                    : std::sqrt(err / nrm);
}

template<typename V>
void require_close_lt(const std::vector<V>& ref, const std::vector<V>& got) {
    const long double err = relerr_ld(ref, got);
    INFO("relative L2 error " << static_cast<double>(err));
    REQUIRE(err <= static_cast<long double>(fft_tol<lt_t>()));
}

std::mt19937_64 rng(7);

std::vector<lc_t> random_c(std::size_t n) {
    std::vector<lc_t> x(n);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    for (auto& v : x) v = {unif(rng), unif(rng)};
    return x;
}

// Sizes: 7-smooth, primes inside the direct-DFT band, primes and prime-power
// residues beyond it (Bluestein), and one large composite.
constexpr std::size_t sizes[] = {1, 2, 4, 8, 16, 27, 32, 64, 255, 1024,
                                 3, 11, 13, 17, 31, 37, 101, 821, 4093, 3939};

TEST_CASE("long double 1-D c2c matches reference_dft", "[longdouble]") {
    for (const std::size_t n : sizes) {
        const auto x = random_c(n);
        admiral::plan<lt_t> p(n);
        auto fwd = x;
        p.forward(fwd.data());
        // reference_dft is O(N^2); the big sizes get round-trip-only checks.
        if (n <= 1024) {
            const auto ref = reference_dft<lt_t, lt_t>(x, /*forward=*/true);
            require_close_lt(ref, fwd);
        }
        auto rt = x;
        p.forward(rt.data());
        p.inverse(rt.data());
        require_close_lt(x, rt);
    }
}

TEST_CASE("long double N-D c2c round-trips", "[longdouble]") {
    for (const auto& shape : {std::vector<std::size_t>{6, 20}, {3, 11, 9}}) {
        std::size_t total = 1;
        for (const std::size_t e : shape) total *= e;
        const auto x = random_c(total);
        admiral::plan<lt_t> p(shape);
        auto w = x;
        p.forward(w.data());
        p.inverse(w.data());
        require_close_lt(x, w);
    }
}

// A round trip alone cannot separate r2c from c2r: a sign flip or a permuted
// half-spectrum that both directions share cancels. Check the spectrum itself
// against the reference DFT of the real input, then round-trip.
TEST_CASE("long double r2c/c2r round-trip", "[longdouble]") {
    for (const std::size_t n : {1ul, 2ul, 5ul, 8ul, 31ul, 64ul, 1023ul, 4093ul}) {
        std::vector<lt_t> x(n);
        std::uniform_real_distribution<lt_t> unif(-1, 1);
        for (auto& v : x) v = unif(rng);
        admiral::plan_r2c<lt_t> rp(n);
        std::vector<lc_t> spec(rp.cplx_size());
        rp.forward(x.data(), spec.data());

        if (n <= 1024) {   // reference_dft is O(N^2)
            std::vector<lc_t> xc(n);
            for (std::size_t i = 0; i < n; ++i) xc[i] = {x[i], lt_t(0)};
            const auto full = reference_dft<lt_t, lt_t>(xc, /*forward=*/true);
            const auto nc = static_cast<std::ptrdiff_t>(rp.cplx_size());
            const std::vector<lc_t> half(full.begin(), full.begin() + nc);
            require_close_lt(half, spec);
        }

        std::vector<lc_t> spec_copy = spec;
        std::vector<lt_t> back(n);
        rp.inverse(spec_copy.data(), back.data());
        require_close_lt(x, back);
    }
}

TEST_CASE("long double r2c N-D round-trip", "[longdouble]") {
    const std::vector<std::size_t> shape{4, 12};
    std::size_t total = 4 * 12;
    std::vector<lt_t> x(total);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    for (auto& v : x) v = unif(rng);
    admiral::plan_r2c<lt_t> rp(shape);
    std::vector<lc_t> spec(rp.cplx_size());
    rp.forward(x.data(), spec.data());
    std::vector<lc_t> spec_copy = spec;
    std::vector<lt_t> back(total);
    rp.inverse(spec_copy.data(), back.data());
    require_close_lt(x, back);
}

// Threading must not move the answer. Every engine in the backend slabs its
// scratch per thread id; a slab that fails to reach one call site is a silent
// data race, so this compares nthreads=4 against nthreads=1 on shapes wide
// enough to fan out. 4093 is prime and above the direct-DFT cap, so its lines
// take the Bluestein path, which owns a second engine and a second slab array.
// {65536} has one line at or above kThreadMinElems, so the engine splits its
// own first level over the pool instead of fanning out lines.
TEST_CASE("long double plans thread without sharing scratch", "[longdouble][threads]") {
    const std::vector<std::vector<std::size_t>> shapes = {
        {64, 4093}, {32, 3939}, {8, 8, 821}, {65536}};
    for (const auto& shape : shapes) {
        INFO("shape " << shape.size() << "-D, inner " << shape.back());
        std::size_t total = 1;
        for (const std::size_t e : shape) total *= e;
        const admiral::span<const std::size_t> sp(shape.data(), shape.size());

        const auto x = random_c(total);
        admiral::plan<lt_t> serial(sp);
        admiral::plan<lt_t> threaded(sp, {/*nthreads=*/4});
        auto a = x, b = x;
        serial.forward(a.data());
        threaded.forward(b.data());
        require_close_lt(a, b);
        serial.inverse(a.data());
        threaded.inverse(b.data());
        require_close_lt(a, b);

        admiral::plan_r2c<lt_t> rserial(sp);
        admiral::plan_r2c<lt_t> rthreaded(sp, {/*nthreads=*/4});
        std::vector<lt_t> r(rserial.real_size());
        std::uniform_real_distribution<lt_t> unif(-1, 1);
        for (auto& v : r) v = unif(rng);
        std::vector<lc_t> ca(rserial.cplx_size()), cb(rserial.cplx_size());
        rserial.forward(r.data(), ca.data());
        rthreaded.forward(r.data(), cb.data());
        require_close_lt(ca, cb);

        // c2r consumes its spectrum, so hand each plan its own copy.
        std::vector<lt_t> ra(rserial.real_size()), rb(rserial.real_size());
        auto ca_copy = ca, cb_copy = cb;
        rserial.inverse(ca_copy.data(), ra.data());
        rthreaded.inverse(cb_copy.data(), rb.data());
        require_close_lt(ra, rb);
        require_close_lt(r, ra);
    }
}

// The quadrant fold in scalar_twiddle keeps the trig argument below a quarter
// turn and makes the four quarter turns come out exact. Both hold without a
// reference of higher precision than long double.
TEST_CASE("long double: scalar_twiddle folds the quadrants exactly", "[longdouble]") {
    using admiral::detail::scalar_twiddle;
    const lt_t eps = std::numeric_limits<lt_t>::epsilon();
    for (const std::size_t n : {4u, 8u, 12u, 100u, 4096u, 65536u}) {
        CAPTURE(n);
        REQUIRE(scalar_twiddle<lt_t>(0, n, true) == lc_t(1, 0));
        REQUIRE(scalar_twiddle<lt_t>(n / 4, n, true) == lc_t(0, -1));
        REQUIRE(scalar_twiddle<lt_t>(n / 2, n, true) == lc_t(-1, 0));
        REQUIRE(scalar_twiddle<lt_t>(3 * n / 4, n, true) == lc_t(0, 1));
        for (std::size_t k = 0; k < n; k += n / 37 + 1) {
            CAPTURE(k);
            const lc_t w = scalar_twiddle<lt_t>(k, n, true);
            REQUIRE(std::abs(std::norm(w) - 1.0L) <= 4 * eps);
            const lc_t wb = scalar_twiddle<lt_t>(k, n, false);   // backward conjugates
            REQUIRE(wb.real() == w.real());
            REQUIRE(wb.imag() == -w.imag());
        }
    }
}

}  // namespace
