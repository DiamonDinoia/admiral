#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/scalar_fft.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <vector>

namespace {

using lt_t = long double;
using lc_t = std::complex<lt_t>;

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
    INFO("relative L2 error " << static_cast<double>(err) << " = "
                              << static_cast<double>(err / std::numeric_limits<lt_t>::epsilon())
                              << " eps");
    REQUIRE(err <= static_cast<long double>(fft_tol<lt_t>()));
}

std::mt19937_64 rng(7);

std::vector<lc_t> random_c(std::size_t n) {
    std::vector<lc_t> x(n);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    for (auto& v : x) v = {unif(rng), unif(rng)};
    return x;
}

constexpr std::size_t sizes[] = {1, 2, 4, 8, 16, 27, 32, 64, 255, 1024,
                                 3, 11, 13, 17, 31, 37, 101, 821, 4093, 3939};

TEST_CASE("long double 1-D c2c matches reference_dft", "[longdouble]") {
    for (const std::size_t n : sizes) {
        const auto x = random_c(n);
        admiral::plan<lt_t> p(n);
        auto fwd = x;
        p.forward(fwd.data());
        if (n <= 1024) {
            const auto ref = reference_dft<lt_t, lt_t>(x, true);
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

TEST_CASE("long double r2c/c2r round-trip", "[longdouble]") {
    for (const std::size_t n : {1ul, 2ul, 5ul, 8ul, 31ul, 64ul, 1023ul, 4093ul}) {
        std::vector<lt_t> x(n);
        std::uniform_real_distribution<lt_t> unif(-1, 1);
        for (auto& v : x) v = unif(rng);
        admiral::plan_r2c<lt_t> rp(n);
        std::vector<lc_t> spec(rp.cplx_size());
        rp.forward(x.data(), spec.data());

        if (n <= 1024) {
            std::vector<lc_t> xc(n);
            for (std::size_t i = 0; i < n; ++i) xc[i] = {x[i], lt_t(0)};
            const auto full = reference_dft<lt_t, lt_t>(xc, true);
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
        admiral::plan<lt_t> threaded(sp, {4});
        auto a = x, b = x;
        serial.forward(a.data());
        threaded.forward(b.data());
        require_close_lt(a, b);
        serial.inverse(a.data());
        threaded.inverse(b.data());
        require_close_lt(a, b);

        admiral::plan_r2c<lt_t> rserial(sp);
        admiral::plan_r2c<lt_t> rthreaded(sp, {4});
        std::vector<lt_t> r(rserial.real_size());
        std::uniform_real_distribution<lt_t> unif(-1, 1);
        for (auto& v : r) v = unif(rng);
        std::vector<lc_t> ca(rserial.cplx_size()), cb(rserial.cplx_size());
        rserial.forward(r.data(), ca.data());
        rthreaded.forward(r.data(), cb.data());
        require_close_lt(ca, cb);

        std::vector<lt_t> ra(rserial.real_size()), rb(rserial.real_size());
        auto ca_copy = ca, cb_copy = cb;
        rserial.inverse(ca_copy.data(), ra.data());
        rthreaded.inverse(cb_copy.data(), rb.data());
        require_close_lt(ra, rb);
        require_close_lt(r, ra);
    }
}

void series_cos_sin(lt_t t, lt_t& c, lt_t& s) {
    const lt_t t2 = t * t;
    lt_t ct = 1, cs = 1, st = t, sn = t;
    for (int k = 1; k <= 30; ++k) {
        ct *= -t2 / (lt_t(2 * k - 1) * lt_t(2 * k));
        st *= -t2 / (lt_t(2 * k) * lt_t(2 * k + 1));
        cs += ct;
        sn += st;
    }
    c = cs;
    s = sn;
}

TEST_CASE("long double: scalar_twiddle phase matches an independent series", "[longdouble]") {
    using admiral::detail::scalar_twiddle;
    const lt_t eps = std::numeric_limits<lt_t>::epsilon();
    lt_t worst = 0;
    std::size_t worst_k = 0, worst_n = 0;
    for (const std::size_t n : {31u, 101u, 821u, 1024u, 3939u, 4093u, 8192u}) {
        for (std::size_t k = 0; k < n; ++k) {
            const std::size_t k4 = 4 * k, q = k4 / n;
            const lt_t rem = static_cast<lt_t>(k4 % n) / static_cast<lt_t>(4 * n);
            lt_t c, s;
            series_cos_sin(lt_t(2) * admiral::detail::numbers::pi_v<lt_t> * rem, c, s);
            const lc_t want = q == 0   ? lc_t(c, s)
                              : q == 1 ? lc_t(-s, c)
                              : q == 2 ? lc_t(-c, -s)
                                       : lc_t(s, -c);
            const lc_t got = scalar_twiddle<lt_t>(k, n, false);
            const lt_t e = std::abs(std::conj(want) * got - lc_t(1, 0)) / eps;
            if (e > worst) {
                worst = e;
                worst_k = k;
                worst_n = n;
            }
        }
    }
    INFO("worst phase error " << static_cast<double>(worst) << " eps at k=" << worst_k
                              << " n=" << worst_n);
    REQUIRE(worst <= 8);
}

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
            const lc_t wb = scalar_twiddle<lt_t>(k, n, false);
            REQUIRE(wb.real() == w.real());
            REQUIRE(wb.imag() == -w.imag());
        }
    }
}

}
