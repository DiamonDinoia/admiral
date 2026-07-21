// Coverage for the large-N four_step_large route (f64, N*16 > 12 MiB) and for
// pointer-alignment variants of the public execute() path.
//
// Two gaps this closes:
//   1. No ctest exercised the four_step_large (Bailey four-step) route — it only
//      triggers for f64 above ~786k points, larger than any prior test size.
//      Forward is checked against an analytical single-tone reference (exact at
//      machine precision for any N), plus round-trip identity, across the
//      L3-transpose-fuse boundary and a non-power-of-two unbalanced split.
//   2. No test drove unaligned/aligned INPUT and OUTPUT buffers through the
//      store-align peel in dif_col_pass_last / four_step_large. User data is not
//      guaranteed cache-line aligned, so we run every (in,out) alignment pairing.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <admiral/admiral.hpp>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <vector>

namespace {

template<typename T>
T fft_tol(std::size_t N, T scale = T(1)) {
    const T eps = std::numeric_limits<T>::epsilon();
    const T log2N = std::log2(static_cast<T>(N)) + T(1);
    return eps * std::sqrt(static_cast<T>(N)) * log2N * T(64) * scale;
}

template<typename T>
std::vector<std::complex<T>> make_input(std::size_t n, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<T>> v(n);
    for (auto& x : v) x = {static_cast<T>(dist(gen)), static_cast<T>(dist(gen))};
    return v;
}

template<typename T>
T max_magnitude(const std::vector<std::complex<T>>& v) {
    T m = T(1);
    for (const auto& x : v) m = std::max(m, std::abs(x));
    return m;
}

// Analytical forward reference: input is a single complex exponential at bin K.
// admiral::forward is X[k] = sum_n x[n]*exp(-2*pi*i*k*n/N), unscaled.
// For x[n] = exp(+2*pi*i*K*n/N), the DFT is exactly N at k==K and 0 elsewhere.
template<typename T>
std::vector<std::complex<T>> tone_input(std::size_t N, std::size_t K) {
    std::vector<std::complex<T>> v(N);
    for (std::size_t n = 0; n < N; ++n) {
        const T ph = T(2) * std::numbers::pi_v<T> * T(K) * T(n) / T(N);
        v[n] = {std::cos(ph), std::sin(ph)};
    }
    return v;
}

template<typename T>
std::vector<std::complex<T>> tone_spectrum(std::size_t N, std::size_t K) {
    std::vector<std::complex<T>> ref(N, std::complex<T>(0, 0));
    ref[K] = std::complex<T>(static_cast<T>(N), 0);
    return ref;
}

template<typename T>
void require_close(const std::vector<std::complex<T>>& got,
                   const std::vector<std::complex<T>>& ref, T tol) {
    REQUIRE(got.size() == ref.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (std::abs(got[i] - ref[i]) > tol) {
            INFO("mismatch at index " << i << " got=(" << got[i].real() << ","
                 << got[i].imag() << ") ref=(" << ref[i].real() << "," << ref[i].imag()
                 << ") tol=" << tol);
            REQUIRE(std::abs(got[i] - ref[i]) <= tol);
        }
    }
}

// Owns an over-aligned buffer and hands out a pointer that is 64-byte aligned
// plus `off_elems` complex elements. off_elems=0 => cache-line aligned;
// off_elems in {1,2,3} => 16/32/48-byte misaligned (complex<double> is 16 B),
// which exercises the store-align peel's head/tail masked-window paths.
template<typename T>
struct offset_buffer {
    std::vector<std::complex<T>> storage;
    std::complex<T>* ptr;
    explicit offset_buffer(std::size_t n, std::size_t off_elems)
        : storage(n + 64 / sizeof(std::complex<T>) + off_elems) {
        auto base = reinterpret_cast<std::uintptr_t>(storage.data());
        std::size_t pad = (64 - (base % 64)) % 64 / sizeof(std::complex<T>);
        ptr = storage.data() + pad + off_elems;
    }
};

// four_step_large routes for f64 when N*sizeof(complex<double>) > 12 MiB
// (N > 786432). L3 transpose-fuse kicks in when N*16 > L3 (~45 MB, N ~ 2.8M).
constexpr std::size_t kBelowFuse = 1048576;   // 16 MB, no transpose fuse
constexpr std::size_t kNonPow2   = 1990656;   // 30 MB, unbalanced split (1728*1152)
constexpr std::size_t kAboveFuse = 3888000;   // 59 MB, transpose fuse active

} // namespace

TEST_CASE("four_step_large forward vs analytical (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kNonPow2}) {
        CAPTURE(N);
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<double>(N, K);
        const auto ref = tone_spectrum<double>(N, K);

        auto plan = admiral::plan<double>(N);
        auto out = in;
        plan.forward(std::span(out));

        require_close(out, ref, fft_tol<double>(N, static_cast<double>(N)));
    }
}

TEST_CASE("four_step_large round-trip identity (double)", "[large][fourstep]") {
    for (const std::size_t N : {kBelowFuse, kNonPow2, kAboveFuse}) {
        CAPTURE(N);
        const auto in = make_input<double>(N, 77u + unsigned(N));
        auto plan = admiral::plan<double>(N);

        auto data = in;
        plan.forward(std::span(data));
        plan.inverse(std::span(data));

        require_close(data, in, fft_tol<double>(N, max_magnitude(in)));
    }
}

// Every (input, output) cache-line-alignment pairing through the OOP execute
// path. Covers both a four_step_large size (store-align peel in the col pass)
// and an iterative_dif size (peel in dif_col_pass_last). Forward validated
// against the analytical tone reference so a mis-peeled store is caught, not
// just a self-consistent one.
TEMPLATE_TEST_CASE("execute() input/output alignment variants vs analytical",
                   "[large][align]", float, double) {
    using T = TestType;
    // four_step_large is f64-only; for f32 this size routes iterative_dif — still
    // a valid alignment probe of the shared peel.
    for (const std::size_t N : {std::size_t{4096}, std::size_t{1048576}}) {
        CAPTURE(N);
        const std::size_t K = N / 4 + 7;
        const auto in  = tone_input<T>(N, K);
        const auto ref = tone_spectrum<T>(N, K);
        const T tol = fft_tol<T>(N, static_cast<T>(N));
        auto plan = admiral::plan<T>(N);

        // off=0 aligned, off=1 misaligned by one complex (16 B for f64 / 8 B for f32).
        for (const std::size_t in_off : {std::size_t{0}, std::size_t{1}}) {
            for (const std::size_t out_off : {std::size_t{0}, std::size_t{1}}) {
                CAPTURE(in_off, out_off);
                offset_buffer<T> src(N, in_off);
                offset_buffer<T> dst(N, out_off);
                std::copy(in.begin(), in.end(), src.ptr);

                plan.forward(src.ptr, dst.ptr);

                std::vector<std::complex<T>> got(dst.ptr, dst.ptr + N);
                require_close(got, ref, tol);

                // src must be untouched by the OOP path regardless of alignment.
                for (std::size_t i = 0; i < N; ++i) REQUIRE(src.ptr[i] == in[i]);
            }
        }
    }
}
