// DCT-II and its exact inverse DCT-III (`dct2`/`dct3`; the `dst` pair works the same).
#include <admiral/admiral.hpp>

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    const std::size_t n = 256;
    std::vector<double> in(n), out(n);
    for (std::size_t i = 0; i < n; ++i) in[i] = std::cos(0.05 * double(i));
    const auto input = in;

    admiral::plan_r2r<double> dct2(n, admiral::r2r_kind::dct2);
    dct2.forward(in, out);
    dct2.inverse(out, in);                       // back to the input exactly

    double err = 0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(in[i] - input[i]));
    if (err > 1e-11) {
        std::fprintf(stderr, "r2r round trip failed: err=%g\n", err);
        return 1;
    }

    // Batched form: transforms contiguous lines of n at once (N-D: transpose between axes).
    admiral::plan_r2r<double> batched(n, admiral::r2r_kind::dct2, 8);
    std::vector<double> block(8 * n, 1.0), bout(8 * n);
    batched.forward(block, bout);
    return 0;
}
