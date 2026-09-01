#include <admiral/admiral.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main() {
    const std::size_t rows = 128, cols = 96;
    std::vector<std::complex<double>> a(rows * cols), b(rows * cols);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = {std::sin(0.01 * double(i)), std::cos(0.03 * double(i))};
    b = a;

    admiral::axis_plan<double> ax0({rows, cols}, 0, true);
    admiral::axis_plan<double> ax1({rows, cols}, 1, true);
    ax0.execute(a.data(), {}, {});
    ax1.execute(a.data(), {}, {});

    admiral::plan<double> full({rows, cols});
    full.forward(b);

    double err = 0;
    for (std::size_t i = 0; i < a.size(); ++i) err = std::max(err, std::abs(a[i] - b[i]));
    if (err > 1e-10) {
        std::fprintf(stderr, "axis chain != full plan: err=%g\n", err);
        return 1;
    }
    return 0;
}
