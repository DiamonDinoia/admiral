#include <admiral/admiral.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main() {
    std::vector<std::complex<double>> x(1024);

    for (std::size_t i = 0; i < x.size(); ++i)
        x[i] = {std::sin(0.01 * double(i)), std::cos(0.02 * double(i))};
    const auto input = x;

    admiral::plan<double> p(x.size());
    p.forward(x);
    p.inverse(x);

    admiral::plan<double> p2d({64, 64});
    admiral::plan<double> p8(1 << 20, {.nthreads = 8});

    admiral::forward<double>(x, x);
    admiral::inverse<double>(x, x);

    double err = 0;
    for (std::size_t i = 0; i < x.size(); ++i) err = std::max(err, std::abs(x[i] - input[i]));
    if (err > 1e-10) {
        std::fprintf(stderr, "round trip failed: err=%g\n", err);
        return 1;
    }
    (void)p2d;
    (void)p8;
    return 0;
}
