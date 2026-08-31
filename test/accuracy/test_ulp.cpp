// Accuracy at the ULP level, against a long-double naive DFT.
//
// The rest of the suite uses one flat budget, 32 eps, sized to pass every route.
// The bound here is 4*eps*sqrt(log2 N), ~10x tighter at these sizes. The bound grows
// with N the way the measured error does, so a change that costs a few ULP shows up.
//
// The reference is O(N^2) in long double and uses long-double trig. The reference
// has no FFT structure of its own, so the reference cannot share a bug with the code
// under test. The O(N^2) cost caps the sizes, hence one representative per route
// rather than a sweep.

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

// Route coverage: codelet catalog, smooth mixed-radix, coprime -> `good_thomas`,
// prime -> `rader`, a prime both decline -> `bluestein`, and a `four_step` split.
// 10 and 20 reach the merged coprime radix 10 = 5*2 (twiddle-free PFA with a
// power-of-two sub-DFT); f64 elects bare-10 at 10 and 2-10 at 20, f32 10-2 at 20,
// so the pair covers the radix leading and trailing. 500 is the only f64 r25.
constexpr std::size_t kSizes[] = {2, 3, 5, 7, 8, 10, 13, 16, 20, 32, 60, 64, 105, 120,
                                  101, 128, 210, 243, 256, 500, 512, 1000, 1009, 1024,
                                  1080, 2048};

// Machine-fit: 4x the worst measured over every size above and 3 seeds. Raise
// `kBound` only with the measurement that justifies the raise.
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

// `sign` = -1 forward (unscaled), +1 inverse (scaled by 1/n), as admiral does.
// `scale` exists only for `factored_dft` below, which composes sub-transforms and
// must apply the 1/n exactly once, at the end.
std::vector<cld> naive_dft(const std::vector<cld>& x, int sign, bool scale = true) {
    const std::size_t n = x.size();
    std::vector<cld> y(n);
    for (std::size_t k = 0; k < n; ++k) {
        cld acc{0, 0};
        for (std::size_t j = 0; j < n; ++j) {
            // `turn_fraction` reduces `k*j` mod n in INTEGERS before dividing.
            // The alternative, `ld(k)*ld(j)/ld(n)`, leaves an absolute angle error
            // that grows like n*eps_ld: 1 double eps of oracle error at n=2048,
            // ~20 at 65536, most of the budget these tests measure.
            const ld ang = ld(sign) * 2 * num::pi_v<ld> * turn_fraction(k, j, n);
            acc += x[j] * cld(std::cos(ang), std::sin(ang));
        }
        y[k] = (sign < 0 || !scale) ? acc : acc / ld(n);
    }
    return y;
}

// Long-double reference for N = n1*n2, both small enough for `naive_dft`. One
// Cooley-Tukey split with j = n2*j1 + j2 and k = n1*k2 + k1:
//   1. n2 column DFTs of length n1
//   2. an explicit long-double twiddle
//   3. n1 row DFTs of length n2
// The split costs O(N*(n1+n2)) instead of O(N^2), which puts 65536 and 262144 in reach.
//
// Unlike `naive_dft`, `factored_dft` has FFT structure, so `factored_dft` could in
// principle share a bug with the code under test. Agreement with `naive_dft` at every
// overlapping size pins the split (below).
std::vector<cld> factored_dft(const std::vector<cld>& x, std::size_t n1, std::size_t n2, int sign) {
    const std::size_t N = n1 * n2;
    std::vector<cld> a(N), line(n1), row(n2), y(N);
    for (std::size_t j2 = 0; j2 < n2; ++j2) {
        for (std::size_t j1 = 0; j1 < n1; ++j1) line[j1] = x[n2 * j1 + j2];
        const auto col = naive_dft(line, sign, false);
        for (std::size_t k1 = 0; k1 < n1; ++k1) {
            // `turn_fraction` reduces `j2*k1` mod N in integers before dividing, so
            // the angle carries no rounding of its own.
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

// The sizes every change in the current performance work targets: `kSizes` cannot
// reach them (the O(N^2) reference is the whole cost), but `factored_dft` can.
struct large_case {
    std::size_t n, n1, n2;
};
constexpr large_case kLarge[] = {{65536, 256, 256}, {100000, 400, 250}, {262144, 512, 512}};

// Tightest cell in a case. The case reports the cell every run: a change that moves
// the margin from 1.2 eps to 3.9 eps still passes, and "passed" says nothing when the
// run exists to compare against a pre-change margin.
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

// FNV-1a over the raw result bytes. Not a portability claim; see the [.golden] case.
std::uint64_t hash_bytes(const void* p, std::size_t bytes) {
    constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ull;
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    const auto* b = static_cast<const unsigned char*>(p);
    std::uint64_t h = kFnvOffsetBasis;
    for (std::size_t i = 0; i < bytes; ++i) h = (h ^ b[i]) * kFnvPrime;
    return h;
}

}  // namespace

TEMPLATE_TEST_CASE("ULP: 1-D forward matches a long-double DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : kSizes) {
        // The O(N^2) reference is the whole cost of this test, so only the cheap
        // sizes get repeated seeds.
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

// The N-D driver transforms one axis at a time, so the driver error is the 1-D error
// accumulated over the axes. The bound therefore uses the total element count.
TEMPLATE_TEST_CASE("ULP: N-D forward matches a long-double DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    // Separable, so the reference is a 1-D DFT along each axis in turn.
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

        // Reference: for each axis, DFT every line along it. Strides are row-major
        // with the last axis fastest, as admiral documents.
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

// `factored_dft` is only usable as an oracle if `factored_dft` agrees with the oracle
// `factored_dft` replaces. Every size here is one `naive_dft` can still afford. The
// case checks each size against the split `factored_dft` uses at large sizes (a
// square-ish one).
TEST_CASE("ULP: the factored reference agrees with the naive one", "[accuracy][ulp]") {
    for (const auto [n, n1, n2] : {large_case{64, 8, 8}, {120, 12, 10}, {256, 16, 16},
                                   {1000, 40, 25}, {2048, 64, 32}}) {
        REQUIRE(n1 * n2 == n);
        const auto x = widen(random_signal<double>(n, 5));
        for (const int sign : {-1, +1}) {
            const auto want = naive_dft(x, sign);
            const auto got = factored_dft(x, n1, n2, sign);
            // Units of the ORACLE's own eps, not double's: long double is 80-bit on
            // x86 and is double on Apple Silicon, so a fixed double-eps bound would
            // encode the x86 mantissa width. The real property: the factored oracle
            // tracks the naive one to a few ulp of the platform's long double. The
            // error grows like sqrt(log N), so extrapolating to 65536 is safe. 32
            // matches the flat 32-eps budget `fft_tol` uses.
            // Where long double == double, the oracle's few-eps noise sits on top of
            // the measured engine error. The consumers keep less headroom there but
            // are not measuring below the oracle floor.
            constexpr double oracle_eps = double(std::numeric_limits<long double>::epsilon());
            const double err = relerrtwonorm(want, got) / oracle_eps;
            CAPTURE(n, n1, n2, sign, err);
            REQUIRE(err <= 32.0);
        }
    }
}

// Large-N ULP, the sizes the performance work targets. 262144 is ~2.7e8
// long-double trig evaluations per precision, so it is hidden; 65536 and 100000 run
// by default. `ctest` picks up only the non-hidden case.
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

// A closed-form oracle that costs O(N) and runs at any size, the ones no
// long-double reference can reach included. The input is a sum of pure tones; the
// unnormalized forward of exp(+2*pi*i*k*j/N) is exactly N at bin k and 0 elsewhere.
// The closed form pins POSITION as well as value: unlike Parseval, the closed form
// cannot pass on a permuted spectrum. The only error the closed form carries is the
// rounding of the generated input to T, ~0.3 eps in relative L2. The unnormalized DFT
// is sqrt(N) times a unitary, so an input perturbation maps to the same relative
// output perturbation.
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
            if (std::find(ks.begin(), ks.end(), k) != ks.end()) continue;  // one tone per bin
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

// Bit-exactness against a recorded value. A tolerance test structurally cannot answer
// "did this refactor change a single bit?", because 1 ULP and 0 ULP both pass. The
// recorded value can.
//
// HOST-SPECIFIC BY CONSTRUCTION: the value depends on the toolchain, the ISA level and
// `-ffast-math` reassociation. The case is hidden ([.golden]) and never runs under
// `ctest`. Run the case by hand around a refactor that is meant to be bit-neutral:
//     ./admiral_tests "[golden]"
// If the toolchain, `ADM_TARGET_ARCH` or a deliberate numerical change moves the value,
// refresh the constant from the number the failure prints.
TEST_CASE("ULP: golden hash of a fixed f64 transform", "[accuracy][ulp][.golden]") {
    constexpr std::size_t n = 65536;
    // Machine-fit on the host toolchain and ISA: the case is hidden, so `ctest`
    // never refreshes `kGolden` by itself. When the case fails, refresh `kGolden`
    // from the number the failure prints.
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

// r2c is the complex engine plus a post-pass, so its own error budget is the same.
TEMPLATE_TEST_CASE("ULP: r2c matches a long-double DFT", "[accuracy][ulp]", float, double) {
    using T = TestType;
    margin m;
    // 26 and 34 halve to a prime > 11, which does not factor into `dif_radix_set`.
    // `multipass_supported` then declines, and the even path runs unbatched. 25/27
    // are odd and skip the recombination pass.
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
        full.resize(p.cplx_size());  // half spectrum: bins 0..n/2

        const double measured = eps_units(full, spec);
        m.see(n, measured, double(bound_for<T>(n)));
        CAPTURE(n, measured, double(bound_for<T>(n)));
        REQUIRE(measured <= double(bound_for<T>(n)));
    }
    REPORT_MARGIN(m);
}

// c2r's direct oracle: an r2c-then-c2r round trip cannot see a Hermitian-completion
// slip the unpack masks. Mirror of the r2c case: conjugate-symmetric LD spectrum ->
// known real signal (c2r folds its 1/N normalization in, so scale the oracle).
TEMPLATE_TEST_CASE("ULP: c2r matches a long-double inverse DFT", "[accuracy][ulp]",
                   float, double) {
    using T = TestType;
    margin m;
    for (const std::size_t n : {8u, 16u, 25u, 26u, 27u, 34u, 60u, 128u, 210u, 1000u, 1024u}) {
        std::mt19937 rng(29);
        std::uniform_real_distribution<ld> u(-1, 1);

        admiral::plan_r2c<T> p({n});
        const std::size_t nh = p.cplx_size();  // n/2+1
        std::vector<cld> full(n);
        for (std::size_t k = 0; k < nh; ++k) {
            // k=0 and (even n) k=n/2 must be real for Hermitian symmetry.
            const ld im = (k == 0 || (n % 2 == 0 && k == n / 2)) ? 0.0L : u(rng);
            full[k] = {u(rng), im};
        }
        for (std::size_t k = nh; k < n; ++k) full[k] = std::conj(full[n - k]);

        auto xr = naive_dft(full, +1);  // scale=true: 1/n inside, matching c2r

        std::vector<std::complex<T>> spec(nh);
        for (std::size_t k = 0; k < nh; ++k) spec[k] = {T(full[k].real()), T(full[k].imag())};
        std::vector<T> got(n);
        p.inverse(spec.data(), got.data());

        ld err = 0, nrm = 0;  // eps-normalized rel-L2, same statistic as `eps_units`
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
