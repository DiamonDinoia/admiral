#include <catch2/catch_test_macros.hpp>
#include <admiral/detail/cxx_compat.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "utils/reference.hpp"

#include <admiral/admiral.h>
#include <cmath>
#include <cstring>
#include <vector>

using namespace Catch::Matchers;

TEST_CASE("C API forward/inverse transforms (double)", "[c_api]") {
    const size_t N = 16;

    std::vector<adm_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(2.0 * admiral::detail::numbers::pi * double(i) / double(N));
        input[i].imag = std::cos(2.0 * admiral::detail::numbers::pi * double(i) / double(N));
    }

    std::vector<adm_complex> original = input;

    adm_status status = adm_forward(input.data(), N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    status = adm_inverse(input.data(), N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    require_close_c(input.data(), original.data(), N, fft_tol<double>());
}

TEST_CASE("C API forward/inverse transforms (float)", "[c_api]") {
    const size_t N = 16;

    std::vector<admf_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(float(i) * 0.1f);
        input[i].imag = std::cos(float(i) * 0.1f);
    }

    std::vector<admf_complex> original = input;

    adm_status status = admf_forward(input.data(), N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    status = admf_inverse(input.data(), N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    require_close_c(input.data(), original.data(), N, fft_tol<float>());
}

TEST_CASE("C API error handling for transforms", "[c_api]") {
    std::vector<adm_complex> data(16);

    SECTION("Null pointer") {
        REQUIRE(adm_forward(nullptr, 16, nullptr) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_inverse(nullptr, 16, nullptr) == ADM_ERROR_NULL_POINTER);
    }

    SECTION("Zero size") {
        REQUIRE(adm_forward(data.data(), 0, nullptr) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(adm_inverse(data.data(), 0, nullptr) == ADM_ERROR_INVALID_SIZE);
    }
}

TEST_CASE("C API bidirectional plan (double)", "[c_api][plan]") {
    const size_t N = 32;

    adm_plan plan = nullptr;
    adm_status status = adm_plan_1d(&plan, N, nullptr);
    REQUIRE(status == ADM_SUCCESS);
    REQUIRE(plan != nullptr);
    REQUIRE(adm_plan_size(plan) == N);

    std::vector<adm_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(double(i) * 0.1);
        input[i].imag = std::cos(double(i) * 0.1);
    }

    std::vector<adm_complex> original = input;

    status = adm_plan_execute_forward(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    status = adm_plan_execute_inverse(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    require_close_c(input.data(), original.data(), N, fft_tol<double>());

    adm_plan_destroy(plan);
}

TEST_CASE("C API plan round-trip (float)", "[c_api][plan]") {
    const size_t N = 64;

    adm_plan plan = nullptr;
    adm_status status = admf_plan_1d(&plan, N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    std::vector<admf_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(float(i) * 0.1f);
        input[i].imag = std::cos(float(i) * 0.2f);
    }

    std::vector<admf_complex> original = input;

    status = admf_plan_execute_forward(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    status = admf_plan_execute_inverse(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    require_close_c(input.data(), original.data(), N, fft_tol<float>());

    adm_plan_destroy(plan);
}

TEST_CASE("C API plan error handling", "[c_api][plan]") {
    SECTION("Null plan pointer") {
        REQUIRE(adm_plan_1d(nullptr, 32, nullptr) == ADM_ERROR_NULL_POINTER);
    }

    SECTION("Zero size") {
        adm_plan plan = nullptr;
        REQUIRE(adm_plan_1d(&plan, 0, nullptr) == ADM_ERROR_INVALID_SIZE);
        adm_plan_destroy(plan);   // a failed call still writes an owned error record
    }

    SECTION("Null plan in execute") {
        std::vector<adm_complex> data(32);
        REQUIRE(adm_plan_execute_forward(nullptr, data.data()) == ADM_ERROR_INVALID_PLAN);
        REQUIRE(adm_plan_execute_inverse(nullptr, data.data()) == ADM_ERROR_INVALID_PLAN);
    }

    SECTION("Null data in execute") {
        adm_plan plan = nullptr;
        REQUIRE(adm_plan_1d(&plan, 32, nullptr) == ADM_SUCCESS);
        REQUIRE(adm_plan_execute_forward(plan, nullptr) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_plan_execute_inverse(plan, nullptr) == ADM_ERROR_NULL_POINTER);
        adm_plan_destroy(plan);
    }
}

TEST_CASE("C API plan reuse", "[c_api][plan]") {
    const size_t N = 32;

    adm_plan plan = nullptr;
    adm_status status = adm_plan_1d(&plan, N, nullptr);
    REQUIRE(status == ADM_SUCCESS);

    for (size_t batch = 0; batch < 5; ++batch) {
        std::vector<adm_complex> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i].real = std::sin(double(batch + i) * 0.1);
            input[i].imag = std::cos(double(batch + i) * 0.2);
        }

        std::vector<adm_complex> original = input;

        status = adm_plan_execute_forward(plan, input.data());
        REQUIRE(status == ADM_SUCCESS);

        status = adm_plan_execute_inverse(plan, input.data());
        REQUIRE(status == ADM_SUCCESS);

        require_close_c(input.data(), original.data(), N, fft_tol<double>());
    }

    adm_plan_destroy(plan);
}

// Sizes over pow2, primes, composites, a prime power, and Bluestein primes.
static const std::vector<size_t> kSweepSizes = {
    4, 8, 16, 32, 64, 128, 256,        // power-of-2
    3, 5, 7, 11, 13, 17, 31,           // small primes
    6, 10, 12, 15, 24, 30, 100,        // composites
    121,                               // 11^2 (non-7-smooth)
    127, 251                           // Bluestein primes
};

TEST_CASE("C API round-trip size sweep (double)", "[c_api][sweep]") {
    for (size_t N : kSweepSizes) {
        CAPTURE(N);
        std::vector<adm_complex> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i].real = std::sin(double(i) * 0.3 + 0.1);
            input[i].imag = std::cos(double(i) * 0.7 - 0.2);
        }
        std::vector<adm_complex> original = input;

        REQUIRE(adm_forward(input.data(), N, nullptr) == ADM_SUCCESS);
        REQUIRE(adm_inverse(input.data(), N, nullptr) == ADM_SUCCESS);

        require_close_c(input.data(), original.data(), N, fft_tol<double>());
    }
}

TEST_CASE("C API round-trip size sweep (float)", "[c_api][sweep]") {
    for (size_t N : kSweepSizes) {
        CAPTURE(N);
        std::vector<admf_complex> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i].real = std::sin(float(i) * 0.3f + 0.1f);
            input[i].imag = std::cos(float(i) * 0.7f - 0.2f);
        }
        std::vector<admf_complex> original = input;

        REQUIRE(admf_forward(input.data(), N, nullptr) == ADM_SUCCESS);
        REQUIRE(admf_inverse(input.data(), N, nullptr) == ADM_SUCCESS);

        require_close_c(input.data(), original.data(), N, fft_tol<float>());
    }
}

// N-D C API.

TEST_CASE("C API N-D round-trip identity (double)", "[c_api][nd]") {
    // 2D and 3D shapes over the catalog / 7-smooth / fallback column routes.
    const std::vector<std::vector<size_t>> shapes = {
        {8, 8}, {16, 32}, {12, 15}, {17, 19}, {31, 9}, {4, 5, 6}, {8, 8, 8}};
    for (const auto& shape : shapes) {
        const size_t Ntot = shape_product(shape);
        CAPTURE(Ntot);

        std::vector<adm_complex> input(Ntot);
        for (size_t i = 0; i < Ntot; ++i) {
            input[i].real = std::sin(double(i) * 0.3 + 0.1);
            input[i].imag = std::cos(double(i) * 0.7 - 0.2);
        }
        std::vector<adm_complex> original = input;

        REQUIRE(adm_forward_nd(input.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
        REQUIRE(adm_inverse_nd(input.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
        require_close_c(input.data(), original.data(), Ntot, fft_tol<double>());
    }
}

TEST_CASE("C API N-D round-trip identity (float)", "[c_api][nd]") {
    const std::vector<std::vector<size_t>> shapes = {{16, 16}, {12, 20}, {8, 4, 2}};
    for (const auto& shape : shapes) {
        const size_t Ntot = shape_product(shape);
        CAPTURE(Ntot);

        std::vector<admf_complex> input(Ntot);
        for (size_t i = 0; i < Ntot; ++i) {
            input[i].real = std::sin(float(i) * 0.3f + 0.1f);
            input[i].imag = std::cos(float(i) * 0.7f - 0.2f);
        }
        std::vector<admf_complex> original = input;

        REQUIRE(admf_forward_nd(input.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
        REQUIRE(admf_inverse_nd(input.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
        require_close_c(input.data(), original.data(), Ntot, fft_tol<float>());
    }
}

// A rank-1 axis is the identity along that axis, so shapes {1, N} and {N, 1} must
// reproduce the 1D transform of size N. This case verifies the contract through the
// C API alone.
TEST_CASE("C API N-D shape {1,N}/{N,1} matches 1D (double)", "[c_api][nd]") {
    for (size_t N : {8u, 16u, 13u, 31u}) {
        CAPTURE(N);
        std::vector<adm_complex> base(N);
        for (size_t i = 0; i < N; ++i) {
            base[i].real = std::sin(double(i) * 0.5 + 0.2);
            base[i].imag = std::cos(double(i) * 0.4 - 0.3);
        }

        std::vector<adm_complex> one_d = base;
        REQUIRE(adm_forward(one_d.data(), N, nullptr) == ADM_SUCCESS);

        for (const std::vector<size_t>& shape : {std::vector<size_t>{1, N},
                                                 std::vector<size_t>{N, 1}}) {
            std::vector<adm_complex> nd = base;
            REQUIRE(adm_forward_nd(nd.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
            require_close_c(nd.data(), one_d.data(), N, fft_tol<double>());
        }
    }
}

TEST_CASE("C API N-D reusable plan matches one-shot (double)", "[c_api][nd][plan]") {
    const std::vector<size_t> shape = {16, 16};
    const size_t Ntot = shape_product(shape);

    std::vector<adm_complex> base(Ntot);
    for (size_t i = 0; i < Ntot; ++i) {
        base[i].real = std::sin(double(i) * 0.11 + 0.7);
        base[i].imag = std::cos(double(i) * 0.13 - 0.5);
    }

    adm_plan plan = nullptr;
    REQUIRE(adm_plan_nd(&plan, shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
    REQUIRE(plan != nullptr);
    REQUIRE(adm_plan_size(plan) == Ntot);

    std::vector<adm_complex> via_plan = base;
    std::vector<adm_complex> via_oneshot = base;
    REQUIRE(adm_plan_execute_forward(plan, via_plan.data()) == ADM_SUCCESS);
    REQUIRE(adm_forward_nd(via_oneshot.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
    require_close_c(via_plan.data(), via_oneshot.data(), Ntot, fft_tol<double>());

    // Round-trip through the same reusable plan returns the original.
    REQUIRE(adm_plan_execute_inverse(plan, via_plan.data()) == ADM_SUCCESS);
    require_close_c(via_plan.data(), base.data(), Ntot, fft_tol<double>());

    adm_plan_destroy(plan);
}

TEST_CASE("C API N-D rejects invalid arguments", "[c_api][nd]") {
    const size_t shape[2] = {8, 8};
    const size_t bad_shape[2] = {8, 0};  // zero extent
    std::vector<adm_complex> data(64);

    REQUIRE(adm_forward_nd(nullptr, shape, 2, nullptr) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward_nd(data.data(), nullptr, 2, nullptr) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(data.data(), shape, 0, nullptr) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(data.data(), bad_shape, 2, nullptr) == ADM_ERROR_INVALID_SIZE);

    adm_plan plan = nullptr;
    REQUIRE(adm_plan_nd(nullptr, shape, 2, nullptr) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_nd(&plan, bad_shape, 2, nullptr) == ADM_ERROR_INVALID_SIZE);
    adm_plan_destroy(plan);   // each failed call overwrote the previous record
    REQUIRE(adm_plan_nd(&plan, shape, 0, nullptr) == ADM_ERROR_INVALID_SIZE);
    adm_plan_destroy(plan);
}

TEST_CASE("C API N-D reusable plan matches one-shot (float)", "[c_api][nd][plan]") {
    const std::vector<size_t> shape = {8, 6, 2};
    const size_t Ntot = shape_product(shape);

    std::vector<admf_complex> base(Ntot);
    for (size_t i = 0; i < Ntot; ++i) {
        base[i].real = std::sin(float(i) * 0.11f + 0.7f);
        base[i].imag = std::cos(float(i) * 0.13f - 0.5f);
    }

    adm_plan plan = nullptr;
    REQUIRE(admf_plan_nd(&plan, shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
    REQUIRE(plan != nullptr);
    REQUIRE(adm_plan_size(plan) == Ntot);

    std::vector<admf_complex> via_plan = base;
    std::vector<admf_complex> via_oneshot = base;
    REQUIRE(admf_plan_execute_forward(plan, via_plan.data()) == ADM_SUCCESS);
    REQUIRE(admf_forward_nd(via_oneshot.data(), shape.data(), shape.size(), nullptr) == ADM_SUCCESS);
    require_close_c(via_plan.data(), via_oneshot.data(), Ntot, fft_tol<float>());

    REQUIRE(admf_plan_execute_inverse(plan, via_plan.data()) == ADM_SUCCESS);
    require_close_c(via_plan.data(), base.data(), Ntot, fft_tol<float>());

    adm_plan_destroy(plan);
}

TEST_CASE("C API threaded plans (double)", "[c_api][plan][threads]") {
    // 1<<16 f64 (1 MiB of complex) crosses the threaded `four_step_large` line.
    // All three plans here take the default nullptr options (`nthreads` = 0, auto);
    // an auto-threaded plan runs on its per-plan pool. The three results must agree.
    const size_t N = 1 << 16;
    std::vector<adm_complex> in(N);
    for (size_t i = 0; i < N; ++i) {
        in[i].real = std::sin(0.7 * double(i)) - 0.3;
        in[i].imag = std::cos(0.3 * double(i)) + 0.2;
    }
    adm_plan p1 = nullptr, p4 = nullptr;
    REQUIRE(adm_plan_1d(&p1, N, nullptr) == ADM_SUCCESS);
    REQUIRE(adm_plan_1d(&p4, N, nullptr) == ADM_SUCCESS);
    std::vector<adm_complex> a = in, b = in;
    REQUIRE(adm_plan_execute_forward(p1, a.data()) == ADM_SUCCESS);
    REQUIRE(adm_plan_execute_forward(p4, b.data()) == ADM_SUCCESS);
    require_close_c(b.data(), a.data(), N, fft_tol<double>(8));
    adm_plan pa = nullptr;
    REQUIRE(adm_plan_1d(&pa, N, nullptr) == ADM_SUCCESS);
    std::vector<adm_complex> c = in;
    REQUIRE(adm_plan_execute_forward(pa, c.data()) == ADM_SUCCESS);
    require_close_c(c.data(), a.data(), N, fft_tol<double>(8));
    // Threaded N-D: the N-D runtime owns one pool per plan.
    adm_plan pn = nullptr;
    const size_t shape[2] = {64, 512};
    REQUIRE(adm_plan_nd(&pn, shape, 2, nullptr) == ADM_SUCCESS);
    std::vector<adm_complex> d(64 * 512);
    for (size_t i = 0; i < d.size(); ++i) {
        d[i].real = std::sin(0.1 * double(i));
        d[i].imag = std::cos(0.2 * double(i));
    }
    const std::vector<adm_complex> d0 = d;
    REQUIRE(adm_plan_execute_forward(pn, d.data()) == ADM_SUCCESS);
    REQUIRE(adm_plan_execute_inverse(pn, d.data()) == ADM_SUCCESS);
    require_close_c(d.data(), d0.data(), d.size(), fft_tol<double>(2));
    adm_plan_destroy(p1);
    adm_plan_destroy(p4);
    adm_plan_destroy(pa);
    adm_plan_destroy(pn);
}
