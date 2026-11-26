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
#include <tuple>
#include <type_traits>
#include <vector>

#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"

#include <admiral/admiral.hpp>
#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/plan.hpp>

// The row-batched c2r entries of the numpy binding (src/inst_real_f.cpp, src/inst_real_d.cpp).
extern "C" void admiral_c2r_rows_f32(const std::complex<float>* in, float* out, std::size_t rows,
                                     std::size_t n);
extern "C" void admiral_c2r_rows_f64(const std::complex<double>* in, double* out,
                                     std::size_t rows, std::size_t n);

namespace {

using dd = admiral::detail::ct_dd;

// A dispatch build runs each public plan on the route of the run-time tier. This TU is built
// at the floor ISA, so the route of its plan_impl can be different.
#ifdef ADM_DISPATCH
constexpr bool kRouteKnown = false;
#else
constexpr bool kRouteKnown = true;
#endif

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

// A real output times 2^-shift must be finite and within tol of the reference rre.
template<typename T>
void require_real_close(const T* out, const std::vector<dd>& rre, int shift, double tol) {
    const std::size_t n = rre.size();
    std::vector<std::complex<double>> got(n);
    bool finite = true;
    for (std::size_t j = 0; j < n; ++j) {
        finite = finite && std::isfinite(out[j]);
        got[j] = {std::ldexp(static_cast<double>(out[j]), -shift), 0.0};
    }
    REQUIRE(finite);
    const double e = rel_err(got.data(), rre, std::vector<dd>(n));
    INFO("rel_err " << e);
    REQUIRE(e <= tol);
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
// the exact output is finite. The r2c spectrum peak is 0.55 of the maximum. The inverse
// transforms apply 1 / n before the growth for the same reason: the inner c2c loads of the
// odd c2r, the c2r tile of the row-batched entry, and the DCT-II that inverts a DCT-III or
// DST-III. The reference runs on the input scaled down by an exact power of two, so ct_dd
// does not overflow.
TEMPLATE_TEST_CASE("real transforms stay finite and accurate with a bin near the type maximum",
                   "[norm_gate][accuracy][overflow]", float, double) {
    using T = TestType;
    const admiral::options opts{1, admiral::effort::estimate};
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const T top = std::numeric_limits<T>::max();
    const int shift = std::numeric_limits<T>::max_exponent - 100;
    const double peak = 0.55 * static_cast<double>(top);
    const double pi = std::acos(-1.0);
    using P = admiral::detail::plan_impl<T>;
    using R = typename P::route_kind;
    // The odd c2r gets 1 / n from the loads of its inner c2c on the codelet and good-thomas
    // routes. An iterative DIF, Rader or Bluestein c2c applies it after the transform and
    // still overflows, so the c2r case skips an odd n with such a route. The odd r2r runs a
    // forward c2c of n. A Rader or Bluestein c2c applies 1 / L after its inner inverse c2c of
    // length L, also in the forward direction, so the r2r case skips such an n. The route of
    // an odd n changes with the SIMD width and the codelet catalog. 11 elects the codelet in
    // the trimmed catalog of a sanitizer build, also where 15 and 45 elect the DIF.
    std::size_t odd_c2r = 0;
    for (const std::size_t n : {128u, 256u, 1024u, 11u, 15u, 17u, 45u}) {
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
        const R inv = P(n, false).route(), fwd = P(n, true).route();
        const bool c2r =
            n % 2 == 0 || (kRouteKnown && (inv == R::codelet || inv == R::good_thomas));
        const bool r2r = n % 2 == 0 || (kRouteKnown && fwd != R::rader && fwd != R::bluestein);
        if (n % 2 == 1 && c2r) ++odd_c2r;
        if (c2r) SECTION("c2r n=" + std::to_string(n)) {
            // c2r: bin 1 has modulus 0.85 of the maximum at phase -pi / 4, where the unscaled
            // recombine term re - im reaches 1.2 of the maximum. The noise is 1e-3 of it.
            const double cpeak = 0.85 * static_cast<double>(top);
            const std::vector<std::complex<double>> w = root_table(n, false);
            std::vector<std::complex<T>> spec(m);
            for (std::size_t k = 0; k < m; ++k)
                spec[k] = {static_cast<T>(1e-3 * cpeak * nd(rng)),
                           (k == 0 || 2 * k == n) ? T(0) : static_cast<T>(1e-3 * cpeak * nd(rng))};
            spec[1] = {static_cast<T>(cpeak * std::cos(-0.25 * pi)), static_cast<T>(cpeak * std::sin(-0.25 * pi))};
            std::vector<dd> rre(n);
            for (std::size_t j = 0; j < n; ++j) {
                dd acc{0.0};
                for (std::size_t k = 0; k < m; ++k) {
                    const std::complex<double> wj = w[(j * k) % n];
                    const dd sr{std::ldexp(static_cast<double>(spec[k].real()), -shift)};
                    const dd si{std::ldexp(static_cast<double>(spec[k].imag()), -shift)};
                    const dd term = sr * dd{wj.real()} - si * dd{wj.imag()};
                    acc += (k == 0 || 2 * k == n) ? term : term + term;
                }
                rre[j] = acc / dd{static_cast<double>(n)};
            }
            SECTION("plan") {
                std::vector<T> out(n);
                admiral::plan_r2c<T> p(n, opts);
                p.inverse(spec.data(), out.data());
                require_real_close(out.data(), rre, shift, tol);
            }
            SECTION("16 rows") {
                // The row-batched entry of the numpy binding: 16 rows take the tile path
                // when n * sizeof(T) <= 4096 and the SIMD width is 16 or less.
                constexpr std::size_t rows = 16;
                std::vector<std::complex<T>> in(rows * m);
                for (std::size_t r = 0; r < rows; ++r)
                    std::copy(spec.begin(), spec.end(),
                              in.begin() + static_cast<std::ptrdiff_t>(r * m));
                std::vector<T> out(rows * n);
                if constexpr (std::is_same_v<T, float>)
                    admiral_c2r_rows_f32(in.data(), out.data(), rows, n);
                else
                    admiral_c2r_rows_f64(in.data(), out.data(), rows, n);
                for (std::size_t r = 0; r < rows; ++r) {
                    INFO("row " << r);
                    require_real_close(out.data() + r * n, rre, shift, tol);
                }
            }
        }
        if (r2r) SECTION("r2r inverse n=" + std::to_string(n)) {
            // The inverse of a DCT-III or DST-III is the DCT-II or DST-II times 1 / (2 n).
            // Every input is near 0.6 of the maximum, so the unscaled sums overflow.
            const std::vector<std::complex<double>> w = root_table(4 * n, false);
            std::vector<T> x(n);
            for (auto& v : x)
                v = static_cast<T>(0.6 * static_cast<double>(top) * (1.0 + 1e-3 * nd(rng)));
            for (const auto kind : {admiral::r2r_kind::dct3, admiral::r2r_kind::dst3}) {
                const bool sine = kind == admiral::r2r_kind::dst3;
                INFO((sine ? "dst3" : "dct3"));
                // cos(pi k (2 j + 1) / (2 n)), or sin(pi (k + 1) (2 j + 1) / (2 n)) for the sine.
                std::vector<dd> rre(n);
                for (std::size_t k = 0; k < n; ++k)
                    for (std::size_t j = 0; j < n; ++j) {
                        const auto wj = w[((sine ? k + 1 : k) * (2 * j + 1)) % (4 * n)];
                        rre[k] += dd{std::ldexp(static_cast<double>(x[j]), -shift)} *
                                  dd{sine ? wj.imag() : wj.real()};
                    }
                for (auto& v : rre) v = v / dd{dn};
                std::vector<T> out(n);
                admiral::plan_r2r<T>(n, kind, 1, opts).inverse(x.data(), out.data());
                require_real_close(out.data(), rre, shift, tol);
            }
        }
    }
    REQUIRE((!kRouteKnown || odd_c2r >= 1));
}

// The normalized inverse c2c applies 1 / n to its input. A scale after the transform
// overflows when the bins sum above the type maximum, while the exact output is finite.
// The input is a tone of modulus 0.85 of the maximum, so the exact output is the tone
// modulus at bin 1. Each butterfly stage grows the partial sums of bin 1, so a scale that
// moves after any stage overflows. Every catalog size runs the codelet and good-thomas
// kernels, so each kernel arm runs on each ISA. The public plan runs the sizes up to 64 that
// elect one of the two kernels.
TEMPLATE_TEST_CASE("normalized inverse c2c stays finite and accurate with bins near the type maximum",
                   "[norm_gate][accuracy][overflow]", float, double) {
    using T = TestType;
    using P = admiral::detail::plan_impl<T>;
    using R = typename P::route_kind;
    const admiral::options opts{1, admiral::effort::estimate};
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const T top = std::numeric_limits<T>::max();
    const int shift = std::numeric_limits<T>::max_exponent - 100;
    const double cpeak = 0.85 * static_cast<double>(top);
    // A null name is the public plan.
    std::vector<std::tuple<std::size_t, const char*, R>> cases;
    for (std::size_t n = 2; n <= 768; ++n) {
        if (P::route_available(R::codelet, n)) cases.emplace_back(n, "codelet", R::codelet);
        if (P::route_available(R::good_thomas, n)) cases.emplace_back(n, "good_thomas", R::good_thomas);
        if (!kRouteKnown || n > 64) continue;
        const R auto_route = P(n, false).route();
        if (auto_route == R::codelet || auto_route == R::good_thomas)
            cases.emplace_back(n, nullptr, auto_route);
    }
    // Every catalog has the codelets 2 to 16, also the trimmed catalog of a sanitizer build.
    REQUIRE(cases.size() >= 15);
    for (const auto& [n, name, route] : cases)
        SECTION(std::string(name ? name : "plan") + " n=" + std::to_string(n)) {
        const double dn = static_cast<double>(n);
        const double tol = 4.0 * eps * std::log2(dn);
        std::mt19937 rng(4321u);
        std::normal_distribution<double> nd(0.0, 1.0);
        // x_j = cpeak cis(-2 pi j / n), with noise 1e-3 of it. w[j] is cis(2 pi j / n).
        const std::vector<std::complex<double>> w = root_table(n, false);
        std::vector<std::complex<T>> x(n);
        for (std::size_t j = 0; j < n; ++j)
            x[j] = {static_cast<T>(cpeak * (w[j].real() + 1e-3 * nd(rng))),
                    static_cast<T>(cpeak * (1e-3 * nd(rng) - w[j].imag()))};
        std::vector<dd> rre(n), rim(n);
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                const std::complex<double> wj = w[(j * k) % n];
                const dd xr{std::ldexp(static_cast<double>(x[j].real()), -shift)};
                const dd xi{std::ldexp(static_cast<double>(x[j].imag()), -shift)};
                rre[k] += xr * dd{wj.real()} - xi * dd{wj.imag()};
                rim[k] += xr * dd{wj.imag()} + xi * dd{wj.real()};
            }
            rre[k] = rre[k] / dd{dn};
            rim[k] = rim[k] / dd{dn};
        }
        for (const bool in_place : {false, true}) SECTION(in_place ? "in place" : "out of place") {
            std::vector<std::complex<T>> out(x);
            if (name == nullptr) {
                const admiral::plan<T> p(n, opts);
                if (in_place) p.inverse(out.data());
                else          p.inverse(x.data(), out.data());
            } else {
                const P p(n, false, route);
                if (in_place) p.execute(out.data(), out.data());
                else          p.execute(x.data(), out.data());
            }
            std::vector<std::complex<double>> got(n);
            bool finite = true;
            for (std::size_t k = 0; k < n; ++k) {
                finite = finite && std::isfinite(out[k].real()) && std::isfinite(out[k].imag());
                got[k] = {std::ldexp(static_cast<double>(out[k].real()), -shift),
                          std::ldexp(static_cast<double>(out[k].imag()), -shift)};
            }
            REQUIRE(finite);
            const double e = rel_err(got.data(), rre, rim);
            INFO("rel_err " << e);
            REQUIRE(e <= tol);
        }
    }
}
