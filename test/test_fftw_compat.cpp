// FFTW3 compatibility shim: verify the covered surface matches FFTW conventions
// (unnormalized both directions) and agrees with the native admiral:: API.
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <complex>
#include <vector>

#include <admiral/admiral.hpp>
#include <admiral/fftw3.h>

namespace {
std::vector<std::complex<double>> signal(int N) {
    std::vector<std::complex<double>> v(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) v[static_cast<std::size_t>(i)] = {std::sin(0.3 * i), std::cos(0.17 * i)};
    return v;
}
double maxdiff(const std::vector<std::complex<double>>& a, const std::vector<std::complex<double>>& b) {
    double m = 0;
    for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}
}  // namespace

TEST_CASE("fftw shim: forward matches native unnormalized forward", "[fftw]") {
    for (int N : {8, 12, 15, 64, 100}) {
        auto in = signal(N);
        std::vector<std::complex<double>> shim(static_cast<std::size_t>(N)), native = in;

        fftw_plan p = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(in.data()),
                                       reinterpret_cast<fftw_complex*>(shim.data()),
                                       FFTW_FORWARD, FFTW_ESTIMATE);
        REQUIRE(p != nullptr);
        fftw_execute(p);
        fftw_destroy_plan(p);

        admiral::plan<double> np(static_cast<std::size_t>(N));
        np.forward(std::span(native), 1.0);  // fct=1 -> unnormalized, FFTW convention

        REQUIRE(maxdiff(shim, native) < 1e-11 * N);
    }
}

TEST_CASE("fftw shim: forward then backward is N*x (unnormalized)", "[fftw]") {
    const int N = 96;
    auto x = signal(N);
    std::vector<std::complex<double>> buf = x;

    fftw_plan fwd = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(buf.data()),
                                     reinterpret_cast<fftw_complex*>(buf.data()),
                                     FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan bwd = fftw_plan_dft_1d(N, reinterpret_cast<fftw_complex*>(buf.data()),
                                     reinterpret_cast<fftw_complex*>(buf.data()),
                                     FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(fwd);
    fftw_execute(bwd);
    fftw_destroy_plan(fwd);
    fftw_destroy_plan(bwd);

    for (std::size_t i = 0; i < x.size(); ++i) x[i] *= double(N);
    REQUIRE(maxdiff(buf, x) < 1e-9 * N);
}

TEST_CASE("fftw shim: 2-D plan and execute_dft on fresh arrays", "[fftw]") {
    const int n0 = 6, n1 = 10, N = n0 * n1;
    auto a = signal(N), b = signal(N);
    for (auto& z : b) z += std::complex<double>(1.0, -0.5);  // make b distinct
    std::vector<std::complex<double>> oa(static_cast<std::size_t>(N)), ob(static_cast<std::size_t>(N));

    fftw_plan p = fftw_plan_dft_2d(n0, n1, reinterpret_cast<fftw_complex*>(a.data()),
                                   reinterpret_cast<fftw_complex*>(oa.data()),
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);                                                              // a -> oa
    fftw_execute_dft(p, reinterpret_cast<fftw_complex*>(b.data()),
                     reinterpret_cast<fftw_complex*>(ob.data()));                 // b -> ob
    fftw_destroy_plan(p);

    admiral::plan<double> np({std::size_t(n0), std::size_t(n1)});
    auto ra = a, rb = b;
    np.forward(std::span(ra), 1.0);
    np.forward(std::span(rb), 1.0);
    REQUIRE(maxdiff(oa, ra) < 1e-11 * N);
    REQUIRE(maxdiff(ob, rb) < 1e-11 * N);
}

TEST_CASE("fftw shim: aligned malloc and single precision", "[fftw]") {
    auto* buf = fftwf_alloc_complex(32);
    REQUIRE(buf != nullptr);
    REQUIRE(reinterpret_cast<std::uintptr_t>(buf) % 64 == 0);
    for (int i = 0; i < 32; ++i) { buf[i][0] = float(i); buf[i][1] = 0.f; }

    fftwf_plan fwd = fftwf_plan_dft_1d(32, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_plan bwd = fftwf_plan_dft_1d(32, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftwf_execute(fwd);
    fftwf_execute(bwd);
    fftwf_destroy_plan(fwd);
    fftwf_destroy_plan(bwd);

    for (int i = 0; i < 32; ++i) REQUIRE(std::abs(buf[i][0] - float(i) * 32.f) < 1e-2f * 32);
    fftwf_free(buf);
}
