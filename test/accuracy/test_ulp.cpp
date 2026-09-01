
#include "utils/reference.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <admiral/admiral.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace {

using ld = long double;
using cld = std::complex<ld>;
namespace num = admiral::detail::numbers;

constexpr std::size_t kSizes[] = {2, 3, 5, 7, 8, 10, 13, 16, 20, 32, 60, 64, 105, 120,
                                  101, 128, 210, 243, 256, 500, 512, 1000, 1009, 1024,
                                  1080, 2048};

constexpr ld kBound = 4;

template<typename T>
std::vector<std::complex<T>> random_signal(std::size_t n, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<T> u(-1, 1);
    std::vector<std::complex<T>> x(n);
    for (auto& v : x) v = {u(rng), u(rng)};
    return x;
}

template<typename T>
std::vector<cld> widen(const std::vector<std::complex<T>>& x) {
    std::vector<cld> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) out[i] = {ld(x[i].real()), ld(x[i].imag())};
    return out;
}

std::vector<cld> naive_dft(const std::vector<cld>& x, int sign, bool scale = true) {
    const std::size_t n = x.size();
    std::vector<cld> y(n);
    for (std::size_t k = 0; k < n; ++k) {
        cld acc{0, 0};
        for (std::size_t j = 0; j < n; ++j) {
            const ld ang = ld(sign) * 2 * num::pi_v<ld> * turn_fraction(k, j, n);
            acc += x[j] * cld(std::cos(ang), std::sin(ang));
        }
        y[k] = (sign < 0 || !scale) ? acc : acc / ld(n);
    }
    return y;
}

std::vector<cld> factored_dft(const std::vector<cld>& x, std::size_t n1, std::size_t n2, int sign) {
    const std::size_t N = n1 * n2;
    std::vector<cld> a(N), line(n1), row(n2), y(N);
    for (std::size_t j2 = 0; j2 < n2; ++j2) {
        for (std::size_t j1 = 0; j1 < n1; ++j1) line[j1] = x[n2 * j1 + j2];
        const auto col = naive_dft(line, sign, false);
        for (std::size_t k1 = 0; k1 < n1; ++k1) {
            const ld ang = ld(sign) * 2 * num::pi_v<ld> * turn_fraction(j2, k1, N);
            a[k1 * n2 + j2] = col[k1] * cld(std::cos(ang), std::sin(ang));
        }
    }
    for (std::size_t k1 = 0; k1 < n1; ++k1) {
        for (std::size_t j2 = 0; j2 < n2; ++j2) row[j2] = a[k1 * n2 + j2];
        const auto out = naive_dft(row, sign, false);
        for (std::size_t k2 = 0; k2 < n2; ++k2)
            y[n1 * k2 + k1] = sign < 0 ? out[k2] : out[k2] / ld(N);
    }
    return y;
}

template<typename T>
double eps_units(const std::vector<cld>& reference, const std::vector<std::complex<T>>& got) {
    return relerrtwonorm(reference, widen(got)) / double(std::numeric_limits<T>::epsilon());
}

template<typename T>
ld bound_for(std::size_t n) {
    return kBound * std::sqrt(std::log2(ld(n)));
}

struct large_case {
    std::size_t n, n1, n2;
};
constexpr large_case kLarge[] = {{65536, 256, 256}, {100000, 400, 250}, {262144, 512, 512}};

struct margin {
    double frac = 0, eps = 0, bound = 0;
    std::size_t at = 0;
    void see(std::size_t n, double measured, double b) {
        if (b > 0 && measured / b > frac) { frac = measured / b, eps = measured, bound = b, at = n; }
    }
};
#define REPORT_MARGIN(m)                                                       \
    WARN("worst margin " << (m).eps << " eps of " << (m).bound << " allowed (" \
                         << 100.0 * (m).frac << "% of budget) at N=" << (m).at)

std::uint64_t hash_bytes(const void* p, std::size_t bytes) {
    constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    const auto* b = static_cast<const unsigned char*>(p);
    std::uint64_t h = kFnvOffsetBasis;
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

}

TEMPLATE_TEST_CASE("ULP: 1-D forward matches a long-double DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : kSizes) {
        const std::uint32_t seeds = n <= 256 ? 3 : 1;
        for (std::uint32_t seed = 1; seed <= seeds; ++seed) {
            const auto x = random_signal<T>(n, seed);
            std::vector<std::complex<T>> y(n);
            admiral::plan<T> p(n);
            p.forward(x.data(), y.data());

            const double measured = eps_units(naive_dft(widen(x), -1), y);
            m.see(n, measured, double(bound_for<T>(n)));
            CAPTURE(n, seed, measured, double(bound_for<T>(n)));
            REQUIRE(measured <= double(bound_for<T>(n)));
        }
    }
    REPORT_MARGIN(m);
}

TEMPLATE_TEST_CASE("ULP: 1-D inverse matches a long-double DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : kSizes) {
        const auto x = random_signal<T>(n, 7);
        std::vector<std::complex<T>> y(n);
        admiral::plan<T> p(n);
        p.inverse(x.data(), y.data());

        const double measured = eps_units(naive_dft(widen(x), +1), y);
        m.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(m);
}

TEMPLATE_TEST_CASE("ULP: N-D forward matches a long-double DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    const std::vector<std::vector<std::size_t>> shapes = {{8, 8}, {12, 9}, {5, 7, 3}, {16, 15}};
    margin m;
    for (const auto& shape : shapes) {
        std::size_t ntot = 1;
        for (const auto e : shape) ntot *= e;

        const auto x = random_signal<T>(ntot, 11);
        std::vector<std::complex<T>> y(ntot);
        const admiral::span<const std::size_t> extents{shape};
        admiral::plan<T> p(extents);
        p.forward(x.data(), y.data());

        auto ref = widen(x);
        std::size_t inner = ntot;
        for (std::size_t ax = 0; ax < shape.size(); ++ax) {
            const std::size_t len = shape[ax];
            inner /= len;
            const std::size_t outer = ntot / (len * inner);
            for (std::size_t o = 0; o < outer; ++o) {
                for (std::size_t i = 0; i < inner; ++i) {
                    std::vector<cld> line(len);
                    for (std::size_t k = 0; k < len; ++k)
                        line[k] = ref[(o * len + k) * inner + i];
                    const auto out = naive_dft(line, -1);
                    for (std::size_t k = 0; k < len; ++k)
                        ref[(o * len + k) * inner + i] = out[k];
                }
            }
        }

        const double measured = eps_units(ref, y);
        m.see(ntot, measured, double(bound_for<T>(ntot)));
        CAPTURE(shape.size(), ntot, measured, double(bound_for<T>(ntot)));
        REQUIRE(measured <= double(bound_for<T>(ntot)));
    }
    REPORT_MARGIN(m);
}

TEST_CASE("ULP: the factored reference agrees with the naive one", "[accuracy][ulp]") {
    for (const auto [n, n1, n2] : {large_case{64, 8, 8}, {120, 12, 10}, {256, 16, 16},
                                   {1000, 40, 25}, {2048, 64, 32}}) {
        REQUIRE(n1 * n2 == n);
        const auto x = widen(random_signal<double>(n, 5));
        for (const int sign : {-1, +1}) {
            const auto want = naive_dft(x, sign);
            const auto got = factored_dft(x, n1, n2, sign);
            constexpr double oracle_eps = double(std::numeric_limits<long double>::epsilon());
            const double err = relerrtwonorm(want, got) / oracle_eps;
            CAPTURE(n, n1, n2, sign, err);
            REQUIRE(err <= 32.0);
        }
    }
}

TEMPLATE_TEST_CASE("ULP: large 1-D forward matches a factored long-double DFT",
                   "[accuracy][ulp]", float, double) {
    using T = TestType;
    margin m;
    for (const auto [n, n1, n2] : {kLarge[0], kLarge[1]}) {
        const auto x = random_signal<T>(n, 3);
        std::vector<std::complex<T>> y(n);
        admiral::plan<T> p(n);
        p.forward(x.data(), y.data());

        const double measured = eps_units(factored_dft(widen(x), n1, n2, -1), y);
        m.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(m);
}

TEMPLATE_TEST_CASE("ULP: 262144 forward matches a factored long-double DFT",
                   "[accuracy][ulp][.large]", float, double) {
    using T = TestType;
    const auto [n, n1, n2] = kLarge[2];
    const auto x = random_signal<T>(n, 3);
    std::vector<std::complex<T>> y(n);
    admiral::plan<T> p(n);
    p.forward(x.data(), y.data());

    const double measured = eps_units(factored_dft(widen(x), n1, n2, -1), y);
    margin m;
    m.see(n, measured, double(bound_for<T>(n)));
    REPORT_MARGIN(m);
    CAPTURE(n, measured, double(bound_for<T>(n)));
    REQUIRE(measured <= double(bound_for<T>(n)));
}

TEMPLATE_TEST_CASE("ULP: multi-tone input has a closed-form spectrum", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin mg;
    for (const std::size_t n : {64u, 120u, 243u, 1009u, 4096u, 15120u, 65537u, 65536u, 100000u, 262144u}) {
        std::mt19937 rng(static_cast<std::uint32_t>(n));
        std::uniform_real_distribution<double> amp(-1, 1);
        std::uniform_int_distribution<std::size_t> bin(0, n - 1);

        std::vector<std::size_t> ks;
        std::vector<cld> as;
        while (ks.size() < 4) {
            const std::size_t k = bin(rng);
            if (std::find(ks.begin(), ks.end(), k) != ks.end()) continue;
            ks.push_back(k);
            as.emplace_back(ld(amp(rng)), ld(amp(rng)));
        }

        std::vector<std::complex<T>> x(n), y(n);
        for (std::size_t j = 0; j < n; ++j) {
            cld acc{0, 0};
            for (std::size_t m = 0; m < ks.size(); ++m) {
                const ld ang = 2 * num::pi_v<ld> * turn_fraction(ks[m], j, n);
                acc += as[m] * cld(std::cos(ang), std::sin(ang));
            }
            x[j] = {T(acc.real()), T(acc.imag())};
        }
        std::vector<cld> want(n, cld{0, 0});
        for (std::size_t m = 0; m < ks.size(); ++m) want[ks[m]] = as[m] * ld(n);

        admiral::plan<T> p(n);
        p.forward(x.data(), y.data());

        const double measured = eps_units(want, y);
        mg.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(mg);
}

TEST_CASE("ULP: golden hash of a fixed f64 transform", "[accuracy][ulp][.golden]") {
    constexpr std::size_t n = 65536;
    constexpr std::uint64_t kGolden = 7717305452990564281ull;

    const auto x = random_signal<double>(n, 20260801);
    std::vector<std::complex<double>> y(n);
    admiral::plan<double> p(n);
    p.forward(x.data(), y.data());

    const std::uint64_t h = hash_bytes(y.data(), y.size() * sizeof(y[0]));
    WARN("golden hash N=" << n << " f64 forward = " << h);
    CAPTURE(h, kGolden);
    REQUIRE(h == kGolden);
}

TEMPLATE_TEST_CASE("ULP: r2c matches a long-double DFT", "[accuracy][ulp]", float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : {8u, 16u, 25u, 26u, 27u, 34u, 60u, 128u, 210u, 1000u, 1024u}) {
        std::mt19937 rng(13);
        std::uniform_real_distribution<T> u(-1, 1);
        std::vector<T> re(n);
        for (auto& v : re) v = u(rng);

        admiral::plan_r2c<T> p({n});
        std::vector<std::complex<T>> spec(p.cplx_size());
        p.forward(re.data(), spec.data());

        std::vector<cld> xl(n);
        for (std::size_t i = 0; i < n; ++i) xl[i] = {ld(re[i]), 0};
        auto full = naive_dft(xl, -1);
        full.resize(p.cplx_size());

        const double measured = eps_units(full, spec);
        m.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(m);
}

TEMPLATE_TEST_CASE("ULP: c2r matches a long-double inverse DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : {8u, 16u, 25u, 26u, 27u, 34u, 60u, 128u, 210u, 1000u, 1024u}) {
        std::mt19937 rng(29);
        std::uniform_real_distribution<ld> u(-1, 1);

        admiral::plan_r2c<T> p({n});
        const std::size_t nh = p.cplx_size();
        std::vector<cld> full(n);
        for (std::size_t k = 0; k < nh; ++k) {
            const ld im = (k == 0 || (n % 2 == 0 && k == n / 2)) ? 0.0L : u(rng);
            full[k] = {u(rng), im};
        }
        for (std::size_t k = nh; k < n; ++k) full[k] = std::conj(full[n - k]);

        auto xr = naive_dft(full, +1);

        std::vector<std::complex<T>> spec(nh);
        for (std::size_t k = 0; k < nh; ++k) spec[k] = {T(full[k].real()), T(full[k].imag())};
        std::vector<T> got(n);
        p.inverse(spec.data(), got.data());

        ld err = 0, nrm = 0;
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(std::fabs(xr[i].imag()) <= 64.0L * std::numeric_limits<ld>::epsilon());
            const ld d = ld(got[i]) - xr[i].real();
            err += d * d;
            nrm += xr[i].real() * xr[i].real();
        }
        const double measured = double(
            std::sqrt(err / nrm) / static_cast<ld>(std::numeric_limits<T>::epsilon()));
        m.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(m);
}
