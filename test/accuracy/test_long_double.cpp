#include <catch2/catch_test_macros.hpp>

#include "utils/reference.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/cxx_compat.hpp>
#include <admiral/detail/dd_fft.hpp>
#include <admiral/detail/scalar_fft.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <sstream>
#include <string>
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

TEST_CASE("long double r2c/c2r with an explicit fct scale every element", "[longdouble]") {
    constexpr lt_t fct = 0.37L;
    for (const auto& shape : {std::vector<std::size_t>{2}, {5}, {8}, {31}, {64}, {3, 7}, {4, 12},
                              {2, 3, 5}}) {
        INFO("shape " << shape.size() << "-D, inner " << shape.back());
        admiral::plan_r2c<lt_t> rp(shape);
        std::vector<lt_t> x(rp.real_size());
        std::uniform_real_distribution<lt_t> unif(-1, 1);
        for (auto& v : x) v = unif(rng);

        std::vector<lc_t> plain(rp.cplx_size()), scaled(rp.cplx_size());
        rp.forward(x.data(), plain.data(), lt_t(1));
        rp.forward(x.data(), scaled.data(), fct);
        for (auto& v : plain) v *= fct;
        require_close_lt(plain, scaled);
        for (std::size_t k = 0; k < plain.size(); ++k) {
            CAPTURE(k);
            REQUIRE(std::abs(plain[k] - scaled[k]) <= 16 * std::numeric_limits<lt_t>::epsilon());
        }

        auto spec = scaled, spec1 = scaled;
        std::vector<lt_t> back(x.size()), back1(x.size());
        rp.inverse(spec.data(), back.data(), fct);
        rp.inverse(spec1.data(), back1.data(), lt_t(1));
        for (auto& v : back1) v *= fct;
        require_close_lt(back1, back);
    }
}

TEST_CASE("long double plans validate extents like the SIMD path", "[longdouble][error]") {
    // A zero extent or overflowing product throws at plan time instead of SIGFPE at
    // execute; r2c rejects an empty shape, matching nd_runtime_plan and nd_real_plan.
    REQUIRE_THROWS_AS(admiral::plan<lt_t>({3, 0, 5}), admiral::size_error);
    REQUIRE_THROWS_AS(
        admiral::plan<lt_t>(std::initializer_list<std::size_t>{std::size_t(-1) / 2, 4}),
        admiral::size_error);
    REQUIRE_THROWS_AS(admiral::plan_r2c<lt_t>({0, 8}), admiral::size_error);
    REQUIRE_THROWS_AS(admiral::plan_r2c<lt_t>(admiral::span<const std::size_t>{}),
                      admiral::size_error);
    REQUIRE_THROWS_AS(
        admiral::plan_r2c<lt_t>(std::initializer_list<std::size_t>{std::size_t(-1) / 2, 4}),
        admiral::size_error);
}

TEST_CASE("long double rank-0 plan is the identity", "[longdouble][edge]") {
    admiral::plan<lt_t> p(std::array<std::size_t, 0>{});
    REQUIRE(p.size() == 1);

    lc_t a{2.0L, -3.0L};
    p.forward(&a);
    REQUIRE(a == lc_t(2.0L, -3.0L));

    lc_t dst{99.0L, 99.0L};
    p.forward(&a, &dst);
    REQUIRE(dst == lc_t(2.0L, -3.0L));

    p.inverse(&a, &dst, 0.25L);
    REQUIRE(dst == lc_t(0.5L, -0.75L));
    p.forward(&a, 0.25L);
    REQUIRE(a == lc_t(0.5L, -0.75L));
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


#if defined(__SIZEOF_FLOAT128__)
__extension__ typedef __float128 f128;
struct c128 {
    f128 re, im;
};

f128 abs128(f128 v) { return v < 0 ? -v : v; }

// e^(-+2 pi i k / n) to 1.2 * 2^-110 (measured against libquadmath's sincosq): pi in three
// doubles, the quadrant of 4k / n folds the angle into [0, pi / 2), and 40 series terms.
c128 root128(std::size_t k, std::size_t n, bool forward) {
    const std::size_t k4 = 4 * (k % n), q = k4 / n;
    const f128 pi = f128(0x1.921fb54442d18p+1) + f128(0x1.1a62633145c07p-53) +
                    f128(-0x1.f1976b7ed8fbcp-109);
    const f128 t = pi * f128(k4 % n) / f128(2 * n), t2 = t * t;
    f128 c = 1, s = t, ct = 1, st = t;
    for (int j = 1; j < 40; ++j) {
        ct *= -t2 / f128((2 * j - 1) * (2 * j));
        st *= -t2 / f128((2 * j) * (2 * j + 1));
        c += ct;
        s += st;
    }
    const c128 w = q == 0   ? c128{c, s}
                   : q == 1 ? c128{-s, c}
                   : q == 2 ? c128{-c, -s}
                            : c128{s, -c};
    return forward ? c128{w.re, -w.im} : w;
}

std::vector<c128> dft128(const std::vector<c128>& x, bool forward) {
    const std::size_t n = x.size();
    std::vector<c128> w(n), y(n);
    for (std::size_t k = 0; k < n; ++k) w[k] = root128(k, n, forward);
    for (std::size_t k = 0; k < n; ++k) {
        f128 re = 0, im = 0;
        for (std::size_t j = 0, e = 0; j < n; ++j, e = (e + k) % n) {
            re += x[j].re * w[e].re - x[j].im * w[e].im;
            im += x[j].re * w[e].im + x[j].im * w[e].re;
        }
        y[k] = {re, im};
    }
    return y;
}

std::vector<f128> flat(const std::vector<c128>& v, f128 scale = 1) {
    std::vector<f128> o;
    for (const c128& z : v) {
        o.push_back(z.re * scale);
        o.push_back(z.im * scale);
    }
    return o;
}

std::vector<lt_t> flat(const std::vector<lc_t>& v) {
    std::vector<lt_t> o;
    for (const lc_t& z : v) {
        o.push_back(z.real());
        o.push_back(z.imag());
    }
    return o;
}

// |got - ref| in units of the long double ulp at ref, the subnormal spacing 2^-16445 below
// 2^-16382.
double ulp_err(f128 ref, lt_t got) {
    const int e = std::max(std::ilogb(static_cast<lt_t>(ref)),
                           std::numeric_limits<lt_t>::min_exponent - 1);
    const f128 ulp = f128(std::ldexp(lt_t(1), e - (std::numeric_limits<lt_t>::digits - 1)));
    return static_cast<double>(abs128(f128(got) - ref) / ulp);
}

// The worst |got - ref| / ulp(ref) over the outputs within 2^30 of the largest: the dd grid
// carries every output to about 2^-104 of the largest, not of itself. NaN propagates.
double worst_ulp(const std::vector<f128>& ref, const std::vector<lt_t>& got) {
    REQUIRE(ref.size() == got.size());
    f128 mx = 0;
    for (const f128 r : ref) mx = std::max(mx, abs128(r));
    double worst = 0;
    std::size_t counted = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (abs128(ref[i]) < mx * f128(0x1p-30)) continue;
        ++counted;
        const double err = ulp_err(ref[i], got[i]);
        if (!(err <= worst)) worst = err;
    }
    REQUIRE(counted > ref.size() / 2);
    return worst;
}

// Correct rounding is 0.5 ulp. The slack covers the reference: its roots and its n-term sums
// put about 2^-103 of the largest output on an output of 2^-93 of it.
constexpr double kRoundedUlp = 0.5 + 0x1p-7;

struct rounding_tally {
    double dd = 0, control = std::numeric_limits<double>::infinity();
    std::ostringstream routes;

    // The dd output rounds correctly; the x87 engine's output, the positive control, does not.
    void check(const char* what, std::size_t n, const std::vector<f128>& ref,
               const std::vector<lt_t>& got, const std::vector<lt_t>& x87) {
        const double e = worst_ulp(ref, got), c = worst_ulp(ref, x87);
        INFO(what << " n=" << n << ": dd " << e << " ulp, x87 control " << c << " ulp");
        REQUIRE(e <= kRoundedUlp);
        REQUIRE(c > kRoundedUlp);
        dd = std::max(dd, e);
        control = std::min(control, c);
    }
};

const char* c2c_route(std::size_t n) {
    using admiral::detail::dd_c2c;
    using admiral::detail::dd_four_step;
    if (n < admiral::detail::kDdMinLen) return "x87";
    if (dd_c2c<lt_t>::blue_pad(n) != 0) return "bluestein";
    return dd_four_step<lt_t>::split(n, false).n1 == 1 ? "line" : "four-step";
}

void require_dd_engine() {
    if (!admiral::detail::kDdLongDouble)
        SKIP("long double is not x87 double-extended with fma here; the x87 engine runs");
}

TEST_CASE("long double c2c rounds every output correctly", "[longdouble][dd]") {
    require_dd_engine();
    std::mt19937_64 gen(11);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    rounding_tally t;
    for (const std::size_t n : {8u, 9u, 11u, 13u, 16u, 17u, 22u, 27u, 37u, 64u, 101u, 210u, 314u,
                                471u, 1000u, 1024u}) {
        std::vector<lc_t> x(n);
        std::vector<c128> xq(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = {unif(gen), unif(gen)};
            xq[i] = {f128(x[i].real()), f128(x[i].imag())};
        }
        t.routes << ' ' << n << ':' << c2c_route(n);
        admiral::plan<lt_t> p(n);
        admiral::detail::scalar_c2c<lt_t> x87(n);
        const auto fwd = dft128(xq, true), inv = dft128(xq, false);

        auto y = x, z = x;
        p.forward(y.data());
        x87.execute<true>(z.data(), lt_t(1), 0);
        t.check("forward", n, flat(fwd), flat(y), flat(z));

        y = x;
        z = x;
        p.inverse(y.data(), lt_t(0.125));
        x87.execute<false>(z.data(), lt_t(0.125), 0);
        t.check("inverse at 1/8", n, flat(inv, f128(0.125)), flat(y), flat(z));

        // The default scale folds in 1 / n exactly, not 1 / n rounded to long double.
        const lt_t def = admiral::detail::default_transform_fct<lt_t>(false, n);
        y = x;
        z = x;
        p.inverse(y.data());
        x87.execute<false>(z.data(), def, 0);
        t.check("inverse at the default scale", n, flat(inv, f128(1) / f128(n)), flat(y), flat(z));
    }
    // 471 runs Bluestein at kDdW 4 and 8.
    REQUIRE(t.routes.str().find("bluestein") != std::string::npos);
    WARN("c2c forward, inverse at 1/8 and at the default exact 1/n round correctly: worst "
         << t.dd << " ulp of " << kRoundedUlp << " allowed; the x87 control exceeds it at every n "
         << "(least " << t.control << " ulp); n:route" << t.routes.str());
}

TEST_CASE("long double r2c and c2r round every output correctly", "[longdouble][dd]") {
    require_dd_engine();
    std::mt19937_64 gen(13);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    rounding_tally t;
    for (const std::size_t n :
         {9u, 11u, 16u, 17u, 22u, 34u, 64u, 101u, 202u, 255u, 471u, 628u, 1000u, 1024u}) {
        const std::size_t nh = n / 2 + 1, m = n % 2 == 0 ? n / 2 : n;
        std::vector<lt_t> x(n);
        std::vector<c128> xq(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = unif(gen);
            xq[i] = {f128(x[i]), 0};
        }
        t.routes << ' ' << n << ':' << (n % 2 == 0 ? "half-" : "full-") << c2c_route(m);
        admiral::plan_r2c<lt_t> p(n);
        admiral::detail::scalar_r2c_1d<lt_t> x87(n);
        auto ref = dft128(xq, true);
        ref.resize(nh);
        std::vector<lc_t> y(nh), z(nh);
        p.forward(x.data(), y.data());
        x87.forward(x.data(), z.data(), 0);
        t.check("r2c", n, flat(ref), flat(y), flat(z));

        // c2r of the reference spectrum rounded to long double, bins 0 and n / 2 real, at the
        // default scale 1 / n: the transform folds it in unrounded.
        std::vector<lc_t> spec(nh);
        for (std::size_t k = 0; k < nh; ++k)
            spec[k] = {static_cast<lt_t>(ref[k].re), static_cast<lt_t>(ref[k].im)};
        spec[0].imag(0);
        if (n % 2 == 0) spec[n / 2].imag(0);
        std::vector<c128> full(n);
        for (std::size_t k = 0; k < n; ++k) {
            const lc_t v = k < nh ? spec[k] : std::conj(spec[n - k]);
            full[k] = {f128(v.real()), f128(v.imag())};
        }
        const auto back = dft128(full, false);
        std::vector<f128> rb(n);
        for (std::size_t i = 0; i < n; ++i) rb[i] = back[i].re / f128(n);
        std::vector<lt_t> o(n), o87(n);
        auto s1 = spec, s2 = spec;
        p.inverse(s1.data(), o.data());
        x87.inverse(s2.data(), o87.data(), lt_t(1), 0);
        t.check("c2r", n, rb, o, o87);
    }
    WARN("r2c and c2r at the default 1/n round correctly: worst " << t.dd << " ulp of "
         << kRoundedUlp << " allowed; the x87 control exceeds it at every n (least " << t.control
         << " ulp); n:route of the inner c2c" << t.routes.str());
}

// Bluestein at a large prime: pad 2^17 or more grows the grid most. Sampled bins against f128.
TEST_CASE("long double c2c rounds sampled bins correctly at a large prime length",
          "[longdouble][dd]") {
    require_dd_engine();
    constexpr std::size_t n = 65537;
    REQUIRE(std::string(c2c_route(n)) == "bluestein");
    std::mt19937_64 gen(31);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    std::vector<lc_t> x(n);
    for (auto& v : x) v = {unif(gen), unif(gen)};
    std::vector<std::size_t> bins = {0, 1, n / 3, n / 2, n - 1};
    while (bins.size() < 12) bins.push_back(admiral::detail::width_cast<std::size_t>(gen() % n));
    admiral::plan<lt_t> p(n);
    admiral::detail::scalar_c2c<lt_t> x87(n);
    rounding_tally t;
    for (const bool forward : {true, false}) {
        std::vector<c128> w(n);
        for (std::size_t k = 0; k < n; ++k) w[k] = root128(k, n, forward);
        std::vector<f128> ref;
        for (const std::size_t k : bins) {
            f128 re = 0, im = 0;
            for (std::size_t j = 0, e = 0; j < n; ++j, e = (e + k) % n) {
                const f128 xr = f128(x[j].real()), xi = f128(x[j].imag());
                re += xr * w[e].re - xi * w[e].im;
                im += xr * w[e].im + xi * w[e].re;
            }
            const f128 sc = forward ? f128(1) : f128(1) / f128(n);
            ref.push_back(re * sc);
            ref.push_back(im * sc);
        }
        auto y = x, z = x;
        if (forward) {
            p.forward(y.data());
            x87.execute<true>(z.data(), lt_t(1), 0);
        } else {
            p.inverse(y.data());
            x87.execute<false>(z.data(), admiral::detail::default_transform_fct<lt_t>(false, n), 0);
        }
        std::vector<lt_t> gy, gz;
        for (const std::size_t k : bins) {
            gy.insert(gy.end(), {y[k].real(), y[k].imag()});
            gz.insert(gz.end(), {z[k].real(), z[k].imag()});
        }
        t.check(forward ? "forward" : "inverse at the default scale", n, ref, gy, gz);
    }
    WARN("c2c n=65537 (bluestein), 12 bins forward and at the default inverse: worst " << t.dd
         << " ulp of " << kRoundedUlp << " allowed; the x87 control reaches " << t.control
         << " ulp at least");
}

TEST_CASE("long double dd output scales exactly with the input exponent", "[longdouble][dd]") {
    require_dd_engine();
    std::mt19937_64 gen(17);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    // The grid follows the largest input exponent, so 2^k x runs the same doubles as x. k spans
    // the long double range: outputs reach 2^16380 and 2^-16190.
    std::ostringstream routes;
    for (const std::size_t n : {16u, 101u, 1000u}) {
        routes << ' ' << n << ':' << c2c_route(n);
        std::vector<lc_t> x(n);
        for (auto& v : x) v = {unif(gen), unif(gen)};
        admiral::plan<lt_t> p(n);
        auto y = x;
        p.forward(y.data());
        for (const int k : {16370, -16200}) {
            CAPTURE(n, k);
            auto ys = x;
            for (auto& v : ys) v = {std::ldexp(v.real(), k), std::ldexp(v.imag(), k)};
            p.forward(ys.data());
            for (std::size_t i = 0; i < n; ++i) {
                CAPTURE(i);
                REQUIRE(ys[i].real() == std::ldexp(y[i].real(), k));
                REQUIRE(ys[i].imag() == std::ldexp(y[i].imag(), k));
            }
        }
        // Positive control: one ulp on one input moves an output, so the comparison can fail.
        auto xp = x;
        xp[0].real(std::nextafter(xp[0].real(), lt_t(2)));
        p.forward(xp.data());
        REQUIRE(xp != y);
    }
    WARN("c2c forward at 2^16370 x and 2^-16200 x equals 2^k times the output for x, bit for bit; "
         << "a one-ulp input change moves the output; n:route" << routes.str());
}

TEST_CASE("long double dd engine rounds subnormal outputs correctly", "[longdouble][dd]") {
    require_dd_engine();
    std::mt19937_64 gen(19);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    rounding_tally t;
    for (const std::size_t n : {16u, 101u, 1000u}) {
        std::vector<lc_t> x(n);
        std::vector<c128> xq(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = {std::ldexp(unif(gen), -16420), std::ldexp(unif(gen), -16420)};
            xq[i] = {f128(x[i].real()), f128(x[i].imag())};
        }
        REQUIRE(std::fpclassify(x[0].real()) == FP_SUBNORMAL);
        t.routes << ' ' << n << ':' << c2c_route(n);
        admiral::plan<lt_t> p(n);
        admiral::detail::scalar_c2c<lt_t> x87(n);
        auto y = x, z = x;
        p.forward(y.data());
        x87.execute<true>(z.data(), lt_t(1), 0);
        t.check("subnormal forward", n, flat(dft128(xq, true)), flat(y), flat(z));
    }
    WARN("c2c forward of inputs near 2^-16420 rounds to the subnormal grid correctly: worst "
         << t.dd << " ulp; the x87 control exceeds it (least " << t.control << " ulp); n:route"
         << t.routes.str());
}

// A batch with a lane outside the normal x87 range stores lane by lane. Every lane there rounds
// once: a normal lane to 64 bits, a subnormal lane to the 2^-16445 grid.
TEST_CASE("long double dd engine rounds every lane of a batch that leaves the normal range",
          "[longdouble][dd]") {
    require_dd_engine();
    constexpr std::size_t n = 16;
    const lt_t third = lt_t(1) / 3;
    admiral::plan<lt_t> p(n);
    admiral::detail::scalar_c2c<lt_t> x87(n);
    std::mt19937_64 gen(23);
    double dd = 0, control = 0;
    for (int trial = 0; trial < 1000; ++trial) {
        // x_j = a and x_0 = a + b rounded: X_0 = 15 a + x_0 stays normal at fct 1/3, X_k = x_0 - a
        // for k > 0 falls below 2^-16382. Both are exact in f128.
        const lt_t a = std::ldexp(static_cast<lt_t>(gen() | std::uint64_t{1} << 63), -63 - 16330);
        const lt_t b = std::ldexp(static_cast<lt_t>(gen() | std::uint64_t{1} << 63), -63 - 16385);
        std::vector<lc_t> x(n, lc_t(a, 0));
        x[0].real(a + b);
        const f128 x0 = f128(x[0].real()), fq = f128(third);
        std::vector<f128> ref(2 * n, f128(0));
        ref[0] = (f128(15) * f128(a) + x0) * fq;
        for (std::size_t k = 1; k < n; ++k) ref[2 * k] = (x0 - f128(a)) * fq;
        REQUIRE(std::fpclassify(static_cast<lt_t>(ref[2])) == FP_SUBNORMAL);
        auto y = x, z = x;
        p.inverse(y.data(), third);
        x87.execute<false>(z.data(), third, 0);
        const auto fy = flat(y), fz = flat(z);
        for (std::size_t i = 0; i < 2 * n; ++i) {
            CAPTURE(trial, i);
            const double e = ulp_err(ref[i], fy[i]);
            REQUIRE(e <= kRoundedUlp);
            dd = std::max(dd, e);
            control = std::max(control, ulp_err(ref[i], fz[i]));
        }
    }
    // Positive control: the x87 engine rounds the sum and the scale apart.
    REQUIRE(control > kRoundedUlp);
    WARN("a batch of one normal and 15 subnormal outputs at fct 1/3: worst " << dd << " ulp of "
         << kRoundedUlp << " allowed over 1000 trials; the x87 control reaches " << control
         << " ulp");
}

TEST_CASE("long double dd engine rounds once at an extreme scale", "[longdouble][dd]") {
    require_dd_engine();
    std::mt19937_64 gen(29);
    std::uniform_real_distribution<lt_t> unif(-1, 1);
    const lt_t third = lt_t(1) / 3;
    std::ostringstream routes;
    double dd = 0;
    for (const std::size_t n : {16u, 101u}) {
        std::vector<lc_t> x(n);
        std::vector<c128> xq(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = {unif(gen), unif(gen)};
            xq[i] = {f128(x[i].real()), f128(x[i].imag())};
        }
        routes << ' ' << n << ':' << c2c_route(n);
        const auto inv = dft128(xq, false);
        admiral::plan<lt_t> p(n);
        // Every output subnormal, outputs across 2^-16382, and outputs near 2^16000.
        for (const lt_t fct : {std::ldexp(lt_t(1), -16440), std::numeric_limits<lt_t>::denorm_min(),
                               std::ldexp(third, -16380), std::ldexp(third, 16000)}) {
            CAPTURE(n, static_cast<double>(std::log2(fct)));
            auto y = x;
            p.inverse(y.data(), fct);
            const auto ref = flat(inv, f128(fct));
            auto got = flat(y);
            const double e = worst_ulp(ref, got);
            REQUIRE(e <= kRoundedUlp);
            dd = std::max(dd, e);
            // Positive control: the largest output one ulp further from the reference fails.
            std::size_t i = 0;
            for (std::size_t j = 1; j < ref.size(); ++j)
                if (abs128(ref[j]) > abs128(ref[i])) i = j;
            const lt_t away = f128(got[i]) < ref[i] ? -std::numeric_limits<lt_t>::infinity()
                                                    : std::numeric_limits<lt_t>::infinity();
            got[i] = std::nextafter(got[i], away);
            REQUIRE(worst_ulp(ref, got) > kRoundedUlp);
        }
    }
    WARN("c2c inverse at fct 2^-16440, 2^-16445, 2^-16380 / 3 and 2^16000 / 3 rounds once: worst "
         << dd << " ulp; a one-ulp move of the largest output fails the check; n:route"
         << routes.str());
}

TEST_CASE("long double dd engine turns a non-finite input into an all-NaN output",
          "[longdouble][dd]") {
    require_dd_engine();
    const lt_t inf = std::numeric_limits<lt_t>::infinity();
    for (const lt_t bad : {inf, -inf, std::numeric_limits<lt_t>::quiet_NaN()}) {
        CAPTURE(static_cast<double>(bad));
        for (const std::size_t n : {16u, 101u}) {
            CAPTURE(n);
            std::vector<lc_t> x(n, lc_t(1, -1));
            x[3].imag(bad);
            admiral::plan<lt_t> p(n);
            p.forward(x.data());
            for (const lc_t& v : x) REQUIRE((std::isnan(v.real()) && std::isnan(v.imag())));

            admiral::plan_r2c<lt_t> rp(2 * n);
            std::vector<lt_t> r(2 * n, lt_t(1));
            r.at(3) = bad;
            std::vector<lc_t> spec(rp.cplx_size());
            rp.forward(r.data(), spec.data());
            for (const lc_t& v : spec) REQUIRE((std::isnan(v.real()) && std::isnan(v.imag())));
            std::fill(spec.begin(), spec.end(), lc_t(1, 0));
            spec[3] = {bad, 0};
            rp.inverse(spec.data(), r.data());
            for (const lt_t v : r) REQUIRE(std::isnan(v));
        }
    }
}

// The grid comes from the largest exponent field. Lengths 1 .. 24 put the value in the masked
// tail of every batch width.
TEST_CASE("long double dd engine reads the largest exponent at every position",
          "[longdouble][dd]") {
    require_dd_engine();
    using admiral::detail::ld_max_exponent;
    const lt_t big = std::ldexp(lt_t(1), 100), inf = std::numeric_limits<lt_t>::infinity();
    for (std::size_t n = 1; n <= 24; ++n) {
        std::vector<lc_t> x(n, lc_t(1, -1));
        CAPTURE(n);
        REQUIRE(ld_max_exponent(x.data(), n) == 16383);
        for (std::size_t q = 0; q < 2 * n; ++q) {
            CAPTURE(q);
            auto* v = reinterpret_cast<lt_t*>(x.data()) + q;
            *v = -big;
            REQUIRE(ld_max_exponent(x.data(), n) == 16383 + 100);
            *v = inf;
            REQUIRE(ld_max_exponent(x.data(), n) == admiral::detail::kLdExpInfNan);
            *v = 1;
        }
    }
}
#else
TEST_CASE("long double dd engine accuracy", "[longdouble][dd]") {
    SKIP("no __float128 to hold the reference DFT");
}
#endif

}
