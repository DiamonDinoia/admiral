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
    T worst = 0;
    for (std::size_t k = 0; k < n; ++k) {
        const auto want = std::complex<T>(k == q ? T(n) : T(0), T(0));
        worst = std::max(worst, std::abs(y[k] - want));
    }
    p.inverse(y.data(), x.data());
    for (std::size_t i = 0; i < n; ++i) {
        const auto phase = T(2) * admiral::detail::numbers::pi_v<T> * T(q) * T(i) / T(n);
        worst = std::max(worst, std::abs(x[i] - std::complex<T>(std::cos(phase), std::sin(phase))));
    }
    const T tol = T(n) * T(32) * std::numeric_limits<T>::epsilon();
    std::printf("%-6s n=%zu worst=%.3g tol=%.3g\n", what, n, double(worst), double(tol));
    return worst <= tol ? 0 : 1;
}

}

int main() {
    int rc = 0;
    rc |= check<double>("f64", 1024, 3);
    rc |= check<float>("f32", 1024, 3);
    rc |= check<double>("f64", 1050, 7);
    std::vector<std::complex<double>> a(512, {1, 0}), b(512);
    admiral::forward<double>(a, b, {2});
    const double dc_tol = 512.0 * 32.0 * std::numeric_limits<double>::epsilon();
    if (std::abs(b[0] - std::complex<double>(512, 0)) > dc_tol) { std::puts("forward() wrong"); rc = 1; }
    std::puts(rc ? "FAIL" : "ok");
    return rc;
}
