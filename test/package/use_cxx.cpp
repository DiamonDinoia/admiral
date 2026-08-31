// Installed C++ interface: only `<admiral/admiral.hpp>`, no detail headers, no xsimd,
// no poet. A single unit spike transforms to a pure tone, so a wrong sign, a wrong
// scale factor or a plan that silently did nothing all fail here.
#include <admiral/admiral.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

template<typename T>
int check(const char* what, std::size_t n, std::size_t q) {
    std::vector<std::complex<T>> x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto phase = T(2) * admiral::detail::numbers::pi_v<T> * T(q) * T(i) / T(n);
        x[i] = {std::cos(phase), std::sin(phase)};
    }
    admiral::plan<T> p(n);
    p.forward(x.data(), y.data());
    // Forward is exp(-2*pi*i*k*n/N), so exp(+2*pi*i*q*n/N) transforms to N at bin q.
    T worst = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const auto want = std::complex<T>(k == q ? T(n) : T(0), T(0));
        worst = std::max(worst, std::abs(y[k] - want));
    }
    p.inverse(y.data(), x.data());  // round trip must return the spike
    for (std::size_t i = 0; i < n; ++i) {
        const auto phase = T(2) * admiral::detail::numbers::pi_v<T> * T(q) * T(i) / T(n);
        worst = std::max(worst, std::abs(x[i] - std::complex<T>(std::cos(phase), std::sin(phase))));
    }
    // Componentwise absolute error on a spectrum peaked at n, so the bound scales
    // with n: max|got-ref| <= 32*eps*||ref||_inf. That is the suite's flat 32*eps
    // discipline of `fft_tol`, in relative L2 there. Spelled out, not shared: an
    // installed-package test may include only `<admiral/admiral.hpp>`.
    const T tol = T(n) * T(32) * std::numeric_limits<T>::epsilon();
    std::printf("%-6s n=%zu worst=%.3g tol=%.3g\n", what, n, double(worst), double(tol));
    return worst <= tol ? 0 : 1;
}

}  // namespace

int main() {
    int rc = 0;
    rc |= check<double>("f64", 1024, 3);
    rc |= check<float>("f32", 1024, 3);
    rc |= check<double>("f64", 1050, 7);  // not a power of two
    // Free functions with an explicit options aggregate: `nthreads` = 2.
    std::vector<std::complex<double>> a(512, {1, 0}), b(512);
    admiral::forward<double>(a, b, {2});
    const double dc_tol = 512.0 * 32.0 * std::numeric_limits<double>::epsilon();
    if (std::abs(b[0] - std::complex<double>(512, 0)) > dc_tol) { std::puts("forward() wrong"); rc = 1; }
    std::puts(rc ? "FAIL" : "ok");
    return rc;
}
