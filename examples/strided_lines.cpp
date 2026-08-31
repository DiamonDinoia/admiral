// `strides_plan` transforms a batch of strided lines out of place, under FFTW's
// `plan_many(rank = 1)` geometry. Source and destination carry independent (stride,
// dist) pairs, so one call transforms every column of a row-major matrix and writes
// the result column-major.
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

    // A line is one column: its elements are `cols` apart, and consecutive columns
    // are 1 apart. The destination packs each transformed column contiguously.
    admiral::strides_plan<double> columns(/*len=*/rows, /*nbatch=*/cols,
                                          /*in_stride=*/cols, /*in_dist=*/1,
                                          /*out_stride=*/1, /*out_dist=*/rows);
    columns.forward(src.data(), dst.data());

    // Oracle: gather one column by hand and run a plain 1-D plan on it.
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
