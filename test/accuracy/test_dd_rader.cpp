// Validates the dd Rader path against the project's reference_dft for prime N.
// For each prime and transform type, the admiral result must agree with the
// reference DFT within the standard fft_tol bound.

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "admiral/admiral.hpp"
#include "utils/reference.hpp"

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
                              << static_cast<double>(err) / digits_eps<lt_t>()
                              << " eps");
    REQUIRE(err <= static_cast<long double>(fft_tol<lt_t>()));
}

std::vector<lc_t> make_input(std::size_t N) {
    std::vector<lc_t> x(N);
    for (std::size_t j = 0; j < N; ++j)
        x[j] = {static_cast<lt_t>(j) * 0.1L, static_cast<lt_t>(N - j) * 0.1L};
    return x;
}

}  // namespace

const std::size_t primes[] = {97, 193, 257};

// Both cases compare against an exact long-double reference. Tagged [longdouble] so the
// valgrind legs drop them: valgrind runs x87 long double at 53 bits, so the reference then
// fails there with no memcheck error.
TEST_CASE("dd_rader: c2c forward matches reference DFT", "[longdouble]") {
    for (const std::size_t N : primes) {
        CAPTURE(N);
        auto x = make_input(N);
        auto ref = reference_dft<lt_t, lt_t>(x, true);
        std::vector<lc_t> got = x;
        std::size_t shape[] = {N};
        admiral::forward(got.data(), shape);
        require_close_lt(ref, got);
    }
}

TEST_CASE("dd_rader: c2c inverse matches reference DFT", "[longdouble]") {
    for (const std::size_t N : primes) {
        CAPTURE(N);
        auto X = make_input(N);
        auto ref = reference_dft<lt_t, lt_t>(X, false);
        for (auto& v : ref) v /= static_cast<lt_t>(N);  // admiral inverse divides by N
        std::vector<lc_t> got = X;
        std::size_t shape[] = {N};
        admiral::inverse(got.data(), shape);
        require_close_lt(ref, got);
    }
}
