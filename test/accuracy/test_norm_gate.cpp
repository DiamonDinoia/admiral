// Norm-gate accuracy test for one-shot plan builds (spec change 4, plan-fast).
// The transform's relative L2 error against a naive reference DFT at double-double (ct_dd)
// precision must sit at machine precision in norm: rel_err <= 4 * eps * log2(n), on random
// normal input, three seeds per size, for r2c and inverse c2c in the numpy one-shot
// configuration (effort::estimate, nthreads 0). The roots of unity fold at ct_dd (about
// 2^-100, ct_math.hpp) because long double is double on MSVC and aarch64 macOS and 53 bits
// under valgrind, and an angle folded there loses the gate budget before the transform
// does. One table of the n roots serves every k; the sums accumulate in ct_dd. The per-call
// pocketfft ratio is reported in logs/plan-fast/REPORT.md; the absolute bound here is the
// suite gate.
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/ct_math.hpp>

namespace {

using dd = admiral::detail::ct_dd;

// cis(-2*pi*t/n) at ct_dd through the repo's own series fold; the transform's tables come
// from the libm turns_memo path, so the reference shares no table with them. Conjugate
// flips the sign of the angle, so one flag serves both test loops.
std::vector<std::complex<double>> root_table(std::size_t n, bool conjugate) {
    std::vector<std::complex<double>> w(n);
    for (std::size_t t = 0; t < n; ++t) {
        const auto r = admiral::detail::sincos_turns_fold<dd>(conjugate, t, n);
        w[t] = {r.c.hi + r.c.lo, r.s.hi + r.s.lo};
    }
    return w;
}

double rel_err(const std::complex<double>* got, const std::vector<dd>& rre,
               const std::vector<dd>& rim) {
    dd num{0.0}, den{0.0};
    for (std::size_t i = 0; i < rre.size(); ++i) {
        const dd gr{got[i].real()}, gi{got[i].imag()};
        const dd er = gr - rre[i], ei = gi - rim[i];
        num += er * er + ei * ei;
        den += rre[i] * rre[i] + rim[i] * rim[i];
    }
    return std::sqrt((num.hi + num.lo) / (den.hi + den.lo));
}

}  // namespace

TEST_CASE("one-shot plan transforms stay at machine precision in norm", "[norm_gate][accuracy]") {
    const admiral::options opts{0, admiral::effort::estimate};
    const double eps = std::numeric_limits<double>::epsilon();
    for (const std::size_t n : {200u, 360u, 500u, 600u, 768u, 1000u, 1200u, 1536u, 2000u, 3000u, 4800u,
                                6000u}) {
        const std::vector<std::complex<double>> w = root_table(n, true);
        for (const unsigned seed : {1234u, 5678u, 9012u}) {
            std::mt19937 rng(seed);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> x(n);
            for (auto& v : x) v = nd(rng);
            const std::size_t m = n / 2 + 1;
            std::vector<dd> rre(m), rim(m);
            for (std::size_t k = 0; k < m; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    const std::complex<double> wj = w[(j * k) % n];
                    const dd xj{x[j]};
                    rre[k] += xj * dd{wj.real()};
                    rim[k] += xj * dd{wj.imag()};
                }
            std::vector<std::complex<double>> out(m);
            admiral::plan_r2c<double> p(n, opts);
            p.forward(x.data(), out.data());
            const double e = rel_err(out.data(), rre, rim);
            INFO("r2c n=" << n << " seed=" << seed << " rel_err " << e);
            REQUIRE(e <= 4.0 * eps * std::log2(static_cast<double>(n)));
        }
    }
    for (const std::size_t n : {500u, 1000u}) {
        const std::vector<std::complex<double>> w = root_table(n, false);
        for (const unsigned seed : {1234u, 5678u, 9012u}) {
            std::mt19937 rng(seed);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<std::complex<double>> x(n);
            for (auto& v : x) v = {nd(rng), nd(rng)};
            std::vector<dd> rre(n), rim(n);
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    const std::complex<double> wj = w[(j * k) % n];
                    const dd xr{x[j].real()}, xi{x[j].imag()};
                    rre[k] += xr * dd{wj.real()} - xi * dd{wj.imag()};
                    rim[k] += xr * dd{wj.imag()} + xi * dd{wj.real()};
                }
            // admiral inverse carries numpy's 1/n scaling.
            const dd dn{static_cast<double>(n)};
            for (std::size_t k = 0; k < n; ++k) {
                rre[k] = rre[k] / dn;
                rim[k] = rim[k] / dn;
            }
            std::vector<std::complex<double>> out(n);
            admiral::plan<double> p(n, opts);
            p.inverse(x.data(), out.data());
            const double e = rel_err(out.data(), rre, rim);
            INFO("c2c inverse n=" << n << " seed=" << seed << " rel_err " << e);
            REQUIRE(e <= 4.0 * eps * std::log2(static_cast<double>(n)));
        }
    }
}

// The real recombine halves its sum and difference terms before the twiddle products. An
// order that scales after them overflows when a bin is above half the type maximum, while
// the exact output is finite. The r2c spectrum peak is 0.55 of the maximum. The reference
// runs on the input scaled down by an exact power of two, so ct_dd does not overflow.
TEMPLATE_TEST_CASE("real transforms stay finite and accurate with a bin near the type maximum",
                   "[norm_gate][accuracy][overflow]", float, double) {
    using T = TestType;
    const admiral::options opts{1, admiral::effort::estimate};
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const T top = std::numeric_limits<T>::max();
    const int shift = std::numeric_limits<T>::max_exponent - 100;
    const double peak = 0.55 * static_cast<double>(top);
    const double pi = std::acos(-1.0);
    for (const std::size_t n : {128u, 256u, 1024u}) {
        const std::size_t m = n / 2 + 1;
        const double dn = static_cast<double>(n);
        const double tol = 4.0 * eps * std::log2(dn);
        std::mt19937 rng(4321u);
        std::normal_distribution<double> nd(0.0, 1.0);
        SECTION("r2c n=" + std::to_string(n)) {
            // r2c: a tone at bin 1 of amplitude peak, plus noise 1e-2 of it.
            const std::vector<std::complex<double>> w = root_table(n, true);
            std::vector<T> x(n);
            for (std::size_t j = 0; j < n; ++j)
                x[j] = static_cast<T>(peak * (2.0 / dn) *
                                     (std::cos(2.0 * pi * static_cast<double>(j) / dn) + 1e-2 * nd(rng)));
            std::vector<dd> rre(m), rim(m);
            for (std::size_t k = 0; k < m; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    const std::complex<double> wj = w[(j * k) % n];
                    const dd xj{std::ldexp(static_cast<double>(x[j]), -shift)};
                    rre[k] += xj * dd{wj.real()};
                    rim[k] += xj * dd{wj.imag()};
                }
            std::vector<std::complex<T>> out(m);
            admiral::plan_r2c<T> p(n, opts);
            p.forward(x.data(), out.data());
            std::vector<std::complex<double>> got(m);
            bool finite = true;
            for (std::size_t k = 0; k < m; ++k) {
                finite = finite && std::isfinite(out[k].real()) && std::isfinite(out[k].imag());
                got[k] = {std::ldexp(static_cast<double>(out[k].real()), -shift),
                          std::ldexp(static_cast<double>(out[k].imag()), -shift)};
            }
            REQUIRE(finite);
            const double e = rel_err(got.data(), rre, rim);
            INFO("rel_err " << e);
            REQUIRE(e <= tol);
        }
        SECTION("c2r n=" + std::to_string(n)) {
            // c2r: bin 1 has modulus 0.85 of the maximum at phase -pi / 4, where the unscaled
            // recombine term re - im reaches 1.2 of the maximum. The noise is 1e-3 of it.
            const double cpeak = 0.85 * static_cast<double>(top);
            const std::vector<std::complex<double>> w = root_table(n, false);
            std::vector<std::complex<T>> spec(m);
            for (std::size_t k = 0; k < m; ++k)
                spec[k] = {static_cast<T>(1e-3 * cpeak * nd(rng)),
                           (k == 0 || k == n / 2) ? T(0) : static_cast<T>(1e-3 * cpeak * nd(rng))};
            spec[1] = {static_cast<T>(cpeak * std::cos(-0.25 * pi)), static_cast<T>(cpeak * std::sin(-0.25 * pi))};
            std::vector<dd> rre(n), rim(n);
            for (std::size_t j = 0; j < n; ++j) {
                dd acc{0.0};
                for (std::size_t k = 0; k < m; ++k) {
                    const std::complex<double> wj = w[(j * k) % n];
                    const dd sr{std::ldexp(static_cast<double>(spec[k].real()), -shift)};
                    const dd si{std::ldexp(static_cast<double>(spec[k].imag()), -shift)};
                    const dd term = sr * dd{wj.real()} - si * dd{wj.imag()};
                    acc += (k == 0 || k == n / 2) ? term : term + term;
                }
                rre[j] = acc / dd{static_cast<double>(n)};
            }
            std::vector<T> out(n);
            admiral::plan_r2c<T> p(n, opts);
            p.inverse(spec.data(), out.data());
            std::vector<std::complex<double>> got(n);
            bool finite = true;
            for (std::size_t j = 0; j < n; ++j) {
                finite = finite && std::isfinite(out[j]);
                got[j] = {std::ldexp(static_cast<double>(out[j]), -shift), 0.0};
            }
            REQUIRE(finite);
            const double e = rel_err(got.data(), rre, rim);
            INFO("rel_err " << e);
            REQUIRE(e <= tol);
        }
    }
}
