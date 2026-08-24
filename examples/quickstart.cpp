// The README Quick start, runnable. Returns nonzero if the round trip fails.
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
    p.forward(x);                  // in place
    p.inverse(x);                  // divides by 1024

    admiral::plan<double> p2d({64, 64});              // N-D; r2c/DCT too, see header
    admiral::plan<double> p8(1 << 20, {.nthreads = 8});

    admiral::forward<double>(x, x);   // one-shot, no plan
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
