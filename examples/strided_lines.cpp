#include <admiral/admiral.hpp>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

int main() {
    const std::size_t rows = 256, cols = 96;
    std::vector<std::complex<double>> src(rows * cols), dst(rows * cols);
    for (std::size_t i = 0; i < src.size(); ++i)
        src[i] = {std::sin(0.3 * double(i)), std::cos(0.7 * double(i) + 1.0)};

    admiral::strides_plan<double> columns(rows, cols,
                                          cols, 1,
                                          1, rows);
    columns.forward(src.data(), dst.data());

    admiral::plan<double> ref({rows});
    std::vector<std::complex<double>> line(rows);
    double err = 0;
    for (std::size_t l = 0; l < cols; ++l) {
        for (std::size_t p = 0; p < rows; ++p) line[p] = src[p * cols + l];
        ref.forward(line);
        for (std::size_t p = 0; p < rows; ++p)
            err = std::max(err, std::abs(dst[l * rows + p] - line[p]));
    }
    if (err > 1e-10) {
        std::fprintf(stderr, "strides_plan != per-column plan: err=%g\n", err);
        return 1;
    }
    return 0;
}
