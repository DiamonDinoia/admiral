// test_fft_coverage.cpp
//
// Coverage-driven tests, added after an llvm-cov sweep flagged cold code the
// existing suite never reached:
//   * the compile-time number-theory helpers in ct_math.hpp (exercised at
//     RUNTIME here — they are constexpr, not consteval, so a runtime call
//     forces a runtime instantiation the kernel-generation callers never do);
//   * every routing path (four_step / rader / bluestein / good_thomas /
//     iterative_dif terminal) via an exhaustive integer-N roundtrip and a
//     naive-DFT cross-check — ideas mirrored from the FFTW3, pocketfft and
//     numpy.fft test suites (exhaustive-N, prime-N/Bluestein, impulse/DC
//     known-answers, Parseval, Hermitian symmetry);
//   * the C API error-argument paths and the entire r2c/c2r C surface.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/admiral.hpp>
#include <admiral/admiral.h>
#include <admiral/detail/rader.hpp>
#include <admiral/detail/ct_math.hpp>
#include <admiral/detail/math.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <string>
#include <vector>

using namespace Catch::Matchers;

// ---------------------------------------------------------------------------
// Local helpers (mirrors the per-file convention in test_fft.cpp / _codelet).
// ---------------------------------------------------------------------------

// O(eps * sqrt(N) * log N) round-trip / accuracy budget, scaled by amplitude.
template<typename T>
T fft_tol(std::size_t N, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(static_cast<T>(N)) + T(1);
    return eps * std::sqrt(static_cast<T>(N)) * log2N * T(64) * scale;
}

// Direct O(N^2) DFT reference with long double accumulation. Forward uses
// exp(-2*pi*i*k*n/N) (yafft's sign convention), unnormalized.
template<typename T>
std::vector<std::complex<T>> reference_dft(const std::vector<std::complex<T>>& x, bool forward) {
    const std::size_t N = x.size();
    const long double sign = forward ? -1.0L : 1.0L;
    const long double two_pi = 2.0L * std::numbers::pi_v<long double>;
    std::vector<std::complex<T>> out(N);
    for (std::size_t k = 0; k < N; ++k) {
        long double sr = 0.0L, si = 0.0L;
        for (std::size_t n = 0; n < N; ++n) {
            const long double ang = sign * two_pi * static_cast<long double>((k * n) % N) / static_cast<long double>(N);
            const long double c = std::cos(ang), s = std::sin(ang);
            const long double xr = static_cast<long double>(x[n].real());
            const long double xi = static_cast<long double>(x[n].imag());
            sr += xr * c - xi * s;
            si += xr * s + xi * c;
        }
        out[k] = std::complex<T>(static_cast<T>(sr), static_cast<T>(si));
    }
    return out;
}

template<typename T>
bool approx_equal(const std::vector<std::complex<T>>& a,
                  const std::vector<std::complex<T>>& b, T tol) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::abs(a[i] - b[i]) > tol) return false;
    return true;
}

template<typename T>
std::vector<std::complex<T>> make_signal(std::size_t N) {
    std::vector<std::complex<T>> v(N);
    for (std::size_t i = 0; i < N; ++i)
        v[i] = std::complex<T>(std::cos(T(0.3) * T(i) + T(1)), std::sin(T(0.17) * T(i)));
    return v;
}

// ===========================================================================
// ct_math.hpp — number-theory helpers, exercised at runtime.
// ===========================================================================
TEST_CASE("ct_math number theory helpers", "[coverage][ct_math]") {
    using namespace admiral::detail;

    SECTION("smallest_radix / codelet_radix peel the expected factor") {
        // Pass runtime (volatile) args so these constexpr fns are evaluated at
        // runtime — the kernel-generation callers only ever hit them constexpr.
        auto rt = [](unsigned n) { volatile unsigned v = n; return smallest_radix(v); };
        REQUIRE(rt(4) == 4);
        REQUIRE(rt(8) == 4);   // 8 % 4 == 0
        REQUIRE(rt(6) == 2);   // even, not mult of 4
        REQUIRE(rt(9) == 3);
        REQUIRE(rt(25) == 5);
        REQUIRE(rt(49) == 7);
        REQUIRE(rt(121) == 11);
        REQUIRE(rt(13) == 13); // prime-to-our-set
        auto cr = [](unsigned n) { volatile unsigned v = n; return codelet_radix(v); };
        REQUIRE(cr(32) == 8);  // special-cased radix-8 peel
        REQUIRE(cr(60) == 5);  // special-cased
        REQUIRE(cr(16) == 4);
    }

    SECTION("ct_is_prime") {
        auto ip = [](unsigned n) { volatile unsigned v = n; return ct_is_prime(v); };
        REQUIRE_FALSE(ip(0));
        REQUIRE_FALSE(ip(1));
        REQUIRE(ip(2));
        REQUIRE(ip(97));
        REQUIRE_FALSE(ip(91));  // 7*13
        REQUIRE(ip(65537));     // Fermat prime
    }

    SECTION("ct_powmod") {
        auto pm = [](unsigned b, unsigned e, unsigned m) {
            volatile unsigned vb = b, ve = e, vm = m; return ct_powmod(vb, ve, vm);
        };
        REQUIRE(pm(2, 10, 1000) == 24);     // 1024 mod 1000
        REQUIRE(pm(3, 0, 7) == 1);          // anything^0
        REQUIRE(pm(7, 4, 5) == 1);          // 7^4=2401, mod5=1
    }

    SECTION("ct_primitive_root generates the full multiplicative group") {
        auto pr = [](unsigned p) { volatile unsigned v = p; return ct_primitive_root(v); };
        for (unsigned p : {13u, 17u, 97u, 101u}) {
            const unsigned g = pr(p);
            REQUIRE(g >= 2);
            // g has order p-1: powers g^1..g^(p-1) hit every residue 1..p-1 once.
            std::vector<bool> seen(p, false);
            unsigned x = 1;
            for (unsigned i = 1; i < p; ++i) {
                x = static_cast<unsigned>((static_cast<unsigned long long>(x) * g) % p);
                REQUIRE_FALSE(seen[x]);
                seen[x] = true;
            }
        }
    }

    SECTION("is_rader_prime") {
        auto rp = [](unsigned n) { volatile unsigned v = n; return is_rader_prime(v); };
        REQUIRE_FALSE(rp(11));  // handled by the radix butterfly, not Rader
        REQUIRE(rp(13));
        REQUIRE(rp(97));
        REQUIRE_FALSE(rp(100)); // composite
    }
}

// ===========================================================================
// Exhaustive integer-N: every route (codelet / iterative_dif / four_step /
// rader / bluestein / good_thomas) is hit somewhere in 2..384. Round-trip for
// all; naive-DFT cross-check for the O(N^2)-affordable range. (FFTW3/pocketfft.)
// ===========================================================================
TEMPLATE_TEST_CASE("exhaustive integer-N round-trip and forward vs naive DFT",
                   "[coverage][sweep]", float, double) {
    using T = TestType;
    for (std::size_t N = 2; N <= 384; ++N) {
        const auto in = make_signal<T>(N);

        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        INFO("round-trip N=" << N);
        REQUIRE(approx_equal(in, recov, fft_tol<T>(N, T(2))));

        // Forward output cross-checked against the direct DFT (masks nothing a
        // round-trip would: sign, direction, twiddle-index bugs all surface).
        if (N <= 200) {
            const auto ref = reference_dft<T>(in, /*forward=*/true);
            INFO("vs-naive N=" << N);
            REQUIRE(approx_equal(spec, ref, fft_tol<T>(N, T(N))));
        }
    }
}

// ===========================================================================
// Impulse / DC known-answers + Parseval across routing boundaries. Catches
// off-by-one twiddle indexing that a round-trip hides. (FFTW3/numpy.)
// ===========================================================================
TEMPLATE_TEST_CASE("known-answer impulse, DC, and Parseval", "[coverage][known]", float, double) {
    using T = TestType;
    const std::array<std::size_t, 14> sizes = {1, 2, 3, 5, 7, 8, 12, 16, 60, 64, 67, 120, 128, 251};
    for (std::size_t N : sizes) {
        INFO("N=" << N);

        // delta[0] -> flat spectrum (all bins == 1).
        {
            std::vector<std::complex<T>> x(N, std::complex<T>(0, 0));
            x[0] = std::complex<T>(1, 0);
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            for (std::size_t k = 0; k < N; ++k) {
                REQUIRE_THAT(X[k].real(), WithinAbs(1.0, 1e-4));
                REQUIRE_THAT(X[k].imag(), WithinAbs(0.0, 1e-4));
            }
        }

        // constant (DC) -> single non-zero bin at k=0 of value N.
        {
            std::vector<std::complex<T>> x(N, std::complex<T>(T(2), T(0)));
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            REQUIRE_THAT(X[0].real(), WithinAbs(2.0 * static_cast<double>(N), 1e-3 * static_cast<double>(N)));
            for (std::size_t k = 1; k < N; ++k)
                REQUIRE(std::abs(X[k]) < fft_tol<T>(N, T(4 * N)));
        }

        // Parseval: ||X||^2 == N * ||x||^2.
        {
            const auto x = make_signal<T>(N);
            std::vector<std::complex<T>> X(N);
            admiral::forward(std::span<const std::complex<T>>(x), std::span(X));
            long double ex = 0, eX = 0;
            for (auto& v : x) ex += static_cast<long double>(std::norm(v));
            for (auto& v : X) eX += static_cast<long double>(std::norm(v));
            REQUIRE_THAT(static_cast<double>(eX),
                         WithinRel(static_cast<double>(ex) * static_cast<double>(N), 1e-3));
        }
    }
}

// ===========================================================================
// Large N: terminal-base DIF path (bn 8/16/32/64) and deep Bluestein for a
// large prime, verified by round-trip + Parseval (naive DFT is infeasible).
// ===========================================================================
TEST_CASE("large-N round-trip (terminal DIF and deep Bluestein)", "[coverage][large]") {
    using T = double;
    for (std::size_t N : {std::size_t{8192}, std::size_t{16384}, std::size_t{32768},
                          std::size_t{65536}, std::size_t{8191}, std::size_t{65537}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        REQUIRE(approx_equal(in, recov, fft_tol<T>(N, T(4))));

        long double ex = 0, eX = 0;
        for (auto& v : in) ex += static_cast<long double>(std::norm(v));
        for (auto& v : spec) eX += static_cast<long double>(std::norm(v));
        REQUIRE_THAT(static_cast<double>(eX),
                     WithinRel(static_cast<double>(ex) * static_cast<double>(N), 1e-6));
    }
}

// ===========================================================================
// f32 batched four-step: sizes 128..768 route to four_step_batched only on
// AVX2 (W=8 for float); the W-parametric split table + dispatch are dead on
// AVX-512 (W=16 falls back to iterative_dif). Round-trip + forward-vs-naive
// exercise both leaves of the split. (Covers four_step.hpp on the v3 build.)
// ===========================================================================
TEST_CASE("f32 batched four-step routed sizes", "[coverage][four_step]") {
    using T = float;
    for (std::size_t N : {std::size_t{128}, std::size_t{256}, std::size_t{384},
                          std::size_t{448}, std::size_t{512}, std::size_t{640},
                          std::size_t{768}}) {
        INFO("N=" << N);
        const auto in = make_signal<T>(N);
        std::vector<std::complex<T>> spec(N), recov(N);
        admiral::forward(std::span<const std::complex<T>>(in), std::span(spec));
        admiral::inverse(std::span<const std::complex<T>>(spec), std::span(recov));
        REQUIRE(approx_equal(in, recov, fft_tol<T>(N, T(4))));
        const auto ref = reference_dft<T>(in, /*forward=*/true);
        REQUIRE(approx_equal(spec, ref, fft_tol<T>(N, T(N))));
    }
}

// ===========================================================================
// Rader with a FOUR-STEP inner convolution. For p=79, L=p-1=78 is neither a
// power of two nor codelet-supported, so pick_inner routes the length-L
// convolution through four_step (rader.hpp four_step branch — never hit by the
// routed sweep, whose small primes use the codelet/iterative_dif inner).
// rader_plan is direction-fixed and unnormalized, matching reference_dft.
// ===========================================================================
TEST_CASE("Rader prime with four-step inner convolution", "[coverage][rader]") {
    using namespace admiral::detail;
    const std::size_t p = 79;
    for (bool fwd : {true, false}) {
        INFO("forward=" << fwd);
        const auto in = make_signal<double>(p);
        rader_plan<double> plan(p, fwd);
        std::vector<std::complex<double>> out(p);
        plan.execute(in.data(), out.data());
        const auto ref = reference_dft<double>(in, fwd);
        REQUIRE(approx_equal(out, ref, fft_tol<double>(p, double(p))));
    }
}

// ===========================================================================
// C API — error-argument paths and the r2c/c2r surface (all cold before).
// ===========================================================================
TEST_CASE("C API error strings cover every status", "[coverage][c_api]") {
    REQUIRE(std::string(adm_error_string(ADM_SUCCESS)) == "Success");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_NULL_POINTER)) == "Null pointer argument");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_SIZE)) == "Invalid size (must be > 0)");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_OUT_OF_MEMORY)) == "Out of memory");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_PLAN)) == "Invalid plan");
    REQUIRE(std::string(adm_error_string(static_cast<adm_status>(999))) == "Unknown error");
}

TEST_CASE("C API transform argument validation", "[coverage][c_api]") {
    admf_complex f{1, 0};
    adm_complex d{1, 0};

    REQUIRE(admf_forward(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_forward(&f, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_inverse(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_inverse(&f, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward(&d, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_inverse(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_inverse(&d, 0) == ADM_ERROR_INVALID_SIZE);

    // A valid single-element float transform (exercises the success path too).
    REQUIRE(admf_forward(&f, 1) == ADM_SUCCESS);
}

TEST_CASE("C API N-D and plan argument validation", "[coverage][c_api]") {
    const std::array<size_t, 2> shape = {4, 4};
    const std::array<size_t, 2> bad = {4, 0};
    std::vector<adm_complex> buf(16);

    REQUIRE(adm_forward_nd(nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward_nd(buf.data(), nullptr, 2) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(buf.data(), shape.data(), 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(buf.data(), bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_inverse_nd(nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);

    // Plan constructors.
    adm_plan p = nullptr;
    REQUIRE(admf_plan_both(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_plan_both(&p, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_plan_both(nullptr, 4) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_both(&p, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_plan_nd(nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_plan_nd(&p, bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_plan_nd(nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_nd(&p, bad.data(), 2) == ADM_ERROR_INVALID_SIZE);

    // Execute with a null / null-data / wrong-type plan.
    admf_complex fdata{1, 0};
    adm_complex ddata{1, 0};
    REQUIRE(admf_plan_execute_forward(nullptr, &fdata) == ADM_ERROR_INVALID_PLAN);
    REQUIRE(adm_plan_execute_inverse(nullptr, &ddata) == ADM_ERROR_INVALID_PLAN);

    adm_plan dplan = nullptr;
    REQUIRE(adm_plan_both(&dplan, 4) == ADM_SUCCESS);
    REQUIRE(admf_plan_execute_forward(dplan, &fdata) == ADM_ERROR_INVALID_PLAN);   // wrong type
    REQUIRE(admf_plan_execute_inverse(dplan, &fdata) == ADM_ERROR_INVALID_PLAN);   // wrong type
    REQUIRE(adm_plan_execute_forward(dplan, nullptr) == ADM_ERROR_NULL_POINTER); // null data
    REQUIRE(adm_plan_execute_inverse(dplan, nullptr) == ADM_ERROR_NULL_POINTER);
    adm_plan_destroy(dplan);

    REQUIRE(adm_plan_size(nullptr) == 0);
}

TEMPLATE_TEST_CASE("C API r2c/c2r round-trip and validation", "[coverage][c_api][real]", float, double) {
    using T = TestType;
    const std::array<size_t, 2> shape = {4, 6};
    const size_t real_n = 4 * 6;
    const size_t cplx_n = 4 * (6 / 2 + 1);

    std::vector<T> in(real_n), out(real_n);
    for (size_t i = 0; i < real_n; ++i) in[i] = std::cos(T(0.4) * T(i)) + T(1);

    using C = std::conditional_t<std::is_same_v<T, float>, admf_complex, adm_complex>;
    std::vector<C> spec(cplx_n);

    adm_status s_fwd, s_inv;
    if constexpr (std::is_same_v<T, float>) {
        s_fwd = admf_r2c_nd(in.data(), spec.data(), shape.data(), 2);
        s_inv = admf_c2r_nd(spec.data(), out.data(), shape.data(), 2);
    } else {
        s_fwd = adm_r2c_nd(in.data(), spec.data(), shape.data(), 2);
        s_inv = adm_c2r_nd(spec.data(), out.data(), shape.data(), 2);
    }
    REQUIRE(s_fwd == ADM_SUCCESS);
    REQUIRE(s_inv == ADM_SUCCESS);
    for (size_t i = 0; i < real_n; ++i)
        REQUIRE_THAT(out[i], WithinAbs(static_cast<double>(in[i]), 1e-4));

    // Argument validation (null + degenerate shape).
    const std::array<size_t, 2> bad = {4, 0};
    if constexpr (std::is_same_v<T, float>) {
        REQUIRE(admf_r2c_nd(nullptr, spec.data(), shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_r2c_nd(in.data(), nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_r2c_nd(in.data(), spec.data(), bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(admf_c2r_nd(nullptr, out.data(), shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_c2r_nd(spec.data(), out.data(), bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
    } else {
        REQUIRE(adm_r2c_nd(nullptr, spec.data(), shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_r2c_nd(in.data(), nullptr, shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_r2c_nd(in.data(), spec.data(), bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(adm_c2r_nd(nullptr, out.data(), shape.data(), 2) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_c2r_nd(spec.data(), out.data(), bad.data(), 2) == ADM_ERROR_INVALID_SIZE);
    }
}
