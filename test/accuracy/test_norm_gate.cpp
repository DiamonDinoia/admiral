// Norm-gate accuracy test for one-shot plan builds (spec change 4, plan-fast).
// The transform's relative L2 error against a long-double naive reference DFT must sit at
// machine precision in norm: rel_err <= 4 * eps * log2(n), on random normal input, three
// seeds per size, for r2c and inverse c2c in the numpy one-shot configuration
// (effort::estimate, nthreads 0). The per-call pocketfft ratio is reported in
// logs/plan-fast/REPORT.md; the absolute bound here is the suite gate.
#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <vector>

#include "catch2/catch_test_macros.hpp"

#include <admiral/admiral.hpp>

namespace {

constexpr long double kPiL = 3.14159265358979323846264338327950288L;

template <typename C>
double rel_err(const C* got, const std::vector<std::complex<long double>>& ref) {
    long double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const std::complex<long double> g(static_cast<long double>(got[i].real()),
                                          static_cast<long double>(got[i].imag()));
        num += std::norm(g - ref[i]);
        den += std::norm(ref[i]);
    }
    return static_cast<double>(std::sqrt(num / den));
}

}  // namespace

TEST_CASE("one-shot plan transforms stay at machine precision in norm", "[norm_gate][accuracy]") {
    const admiral::options opts{.nthreads = 0, .eff = admiral::effort::estimate};
    const double eps = std::numeric_limits<double>::epsilon();
    for (const std::size_t n : {200u, 360u, 500u, 600u, 768u, 1000u, 1200u, 1536u, 2000u, 3000u, 4800u,
                                6000u}) {
        for (const unsigned seed : {1234u, 5678u, 9012u}) {
            std::mt19937 rng(seed);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> x(n);
            for (auto& v : x) v = nd(rng);
            const std::size_t m = n / 2 + 1;
            std::vector<std::complex<long double>> ref(m, 0);
            for (std::size_t k = 0; k < m; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    const long double a =
                        -2.0L * kPiL * static_cast<long double>(j) *
                        static_cast<long double>(k) / static_cast<long double>(n);
                    ref[k] += static_cast<long double>(x[j]) *
                              std::complex<long double>{std::cos(a), std::sin(a)};
                }
            std::vector<std::complex<double>> out(m);
            admiral::plan_r2c<double> p(n, opts);
            p.forward(x.data(), out.data());
            const double e = rel_err(out.data(), ref);
            INFO("r2c n=" << n << " seed=" << seed << " rel_err " << e);
            REQUIRE(e <= 4.0 * eps * std::log2(static_cast<double>(n)));
        }
    }
    for (const std::size_t n : {500u, 1000u}) {
        for (const unsigned seed : {1234u, 5678u, 9012u}) {
            std::mt19937 rng(seed);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<std::complex<double>> x(n);
            for (auto& v : x) v = {nd(rng), nd(rng)};
            std::vector<std::complex<long double>> ref(n, 0);
            for (std::size_t k = 0; k < n; ++k)
                for (std::size_t j = 0; j < n; ++j) {
                    const long double a =
                        2.0L * kPiL * static_cast<long double>(j) *
                        static_cast<long double>(k) / static_cast<long double>(n);
                    ref[k] += std::complex<long double>(
                                  static_cast<long double>(x[j].real()),
                                  static_cast<long double>(x[j].imag())) *
                              std::complex<long double>{std::cos(a), std::sin(a)};
                }
            // admiral inverse carries numpy's 1/n scaling.
            for (auto& v : ref) v /= static_cast<long double>(n);
            std::vector<std::complex<double>> out(n);
            admiral::plan<double> p(n, opts);
            p.inverse(x.data(), out.data());
            const double e = rel_err(out.data(), ref);
            INFO("c2c inverse n=" << n << " seed=" << seed << " rel_err " << e);
            REQUIRE(e <= 4.0 * eps * std::log2(static_cast<double>(n)));
        }
    }
}
