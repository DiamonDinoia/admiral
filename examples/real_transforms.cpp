#include <admiral/admiral.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main() {
    const std::size_t n = 512;
    std::vector<double> in(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = std::sin(0.1 * double(i));

    admiral::plan_r2c<double> p(n);
    std::vector<std::complex<double>> spec(p.cplx_size());
    p.forward(in, spec);
    p.inverse(spec, in);

    double err = 0;
    for (std::size_t i = 0; i < n; ++i)
        err = std::max(err, std::abs(in[i] - std::sin(0.1 * double(i))));
    if (err > 1e-12) {
        std::fprintf(stderr, "real round trip failed: err=%g\n", err);
        return 1;
    }

    admiral::plan_r2c<double> p2({64, 64});
    std::printf("64x64 real -> %zu complex (%zux%zu expected)\n", p2.cplx_size(),
                std::size_t{64}, std::size_t{33});
    return 0;
}
