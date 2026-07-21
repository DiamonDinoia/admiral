#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <admiral/admiral.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <limits>

using namespace Catch::Matchers;

// PI constant
constexpr double pi = 3.141592653589793238462643383279502884;

// Helper to check if two C complex arrays are approximately equal
template<typename T>
bool c_arrays_approx_equal(const T* a, const T* b, size_t size, double tolerance) {
    for (size_t i = 0; i < size; ++i) {
        double diff_real = std::abs(a[i].real - b[i].real);
        double diff_imag = std::abs(a[i].imag - b[i].imag);
        if (diff_real > tolerance || diff_imag > tolerance) {
            return false;
        }
    }
    return true;
}

// Relative error budget for an N-point round-trip in precision T (see test_fft.cpp).
template<typename T>
double fft_tol(std::size_t N, double scale = 1.0) {
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const double log2N = std::log2(static_cast<double>(N)) + 1.0;
    return eps * std::sqrt(static_cast<double>(N)) * log2N * 64.0 * scale;
}

TEST_CASE("C API error strings", "[c_api]") {
    REQUIRE(std::string(adm_error_string(ADM_SUCCESS)) == "Success");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_NULL_POINTER)) == "Null pointer argument");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_SIZE)) == "Invalid size (must be > 0)");
}

TEST_CASE("C API forward/inverse transforms (double)", "[c_api]") {
    const size_t N = 16;
    const double tolerance = 1e-10;

    // Create test data
    std::vector<adm_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(2.0 * pi * double(i) / double(N));
        input[i].imag = std::cos(2.0 * pi * double(i) / double(N));
    }

    // Save original
    std::vector<adm_complex> original = input;

    // Forward transform
    adm_status status = adm_forward(input.data(), N);
    REQUIRE(status == ADM_SUCCESS);

    // Inverse transform
    status = adm_inverse(input.data(), N);
    REQUIRE(status == ADM_SUCCESS);

    // Should recover original
    REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, tolerance));
}

TEST_CASE("C API forward/inverse transforms (float)", "[c_api]") {
    const size_t N = 16;
    const float tolerance = 1e-6f;

    // Create test data
    std::vector<admf_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(float(i) * 0.1f);
        input[i].imag = std::cos(float(i) * 0.1f);
    }

    // Save original
    std::vector<admf_complex> original = input;

    // Forward transform
    adm_status status = admf_forward(input.data(), N);
    REQUIRE(status == ADM_SUCCESS);

    // Inverse transform
    status = admf_inverse(input.data(), N);
    REQUIRE(status == ADM_SUCCESS);

    // Should recover original
    REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, tolerance));
}

TEST_CASE("C API error handling for transforms", "[c_api]") {
    std::vector<adm_complex> data(16);

    SECTION("Null pointer") {
        REQUIRE(adm_forward(nullptr, 16) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_inverse(nullptr, 16) == ADM_ERROR_NULL_POINTER);
    }

    SECTION("Zero size") {
        REQUIRE(adm_forward(data.data(), 0) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(adm_inverse(data.data(), 0) == ADM_ERROR_INVALID_SIZE);
    }
}

TEST_CASE("C API forward plan (double)", "[c_api][plan]") {
    const size_t N = 32;

    // Create plan
    adm_plan plan = nullptr;
    adm_status status = adm_plan_both(&plan, N);
    REQUIRE(status == ADM_SUCCESS);
    REQUIRE(plan != nullptr);
    REQUIRE(adm_plan_size(plan) == N);

    // Create test data
    std::vector<adm_complex> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i].real = std::sin(double(i));
        data[i].imag = std::cos(double(i));
    }

    // Execute plan
    status = adm_plan_execute_forward(plan, data.data());
    REQUIRE(status == ADM_SUCCESS);

    // Clean up
    adm_plan_destroy(plan);
}

TEST_CASE("C API inverse plan (double)", "[c_api][plan]") {
    const size_t N = 32;

    // Create plan
    adm_plan plan = nullptr;
    adm_status status = adm_plan_both(&plan, N);
    REQUIRE(status == ADM_SUCCESS);
    REQUIRE(plan != nullptr);

    // Create test data
    std::vector<adm_complex> data(N);
    for (size_t i = 0; i < N; ++i) {
        data[i].real = std::sin(double(i));
        data[i].imag = std::cos(double(i));
    }

    // Execute plan
    status = adm_plan_execute_inverse(plan, data.data());
    REQUIRE(status == ADM_SUCCESS);

    // Clean up
    adm_plan_destroy(plan);
}

TEST_CASE("C API bidirectional plan (double)", "[c_api][plan]") {
    const size_t N = 32;
    const double tolerance = 1e-10;

    // Create plan
    adm_plan plan = nullptr;
    adm_status status = adm_plan_both(&plan, N);
    REQUIRE(status == ADM_SUCCESS);
    REQUIRE(plan != nullptr);

    // Create test data
    std::vector<adm_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(double(i) * 0.1);
        input[i].imag = std::cos(double(i) * 0.1);
    }

    std::vector<adm_complex> original = input;

    // Forward then inverse
    status = adm_plan_execute_forward(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    status = adm_plan_execute_inverse(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    // Should recover original
    REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, tolerance));

    // Clean up
    adm_plan_destroy(plan);
}

TEST_CASE("C API plan round-trip (float)", "[c_api][plan]") {
    const size_t N = 64;
    const float tolerance = 1e-6f;

    // Create bidirectional plan
    adm_plan plan = nullptr;
    adm_status status = admf_plan_both(&plan, N);
    REQUIRE(status == ADM_SUCCESS);

    // Test data
    std::vector<admf_complex> input(N);
    for (size_t i = 0; i < N; ++i) {
        input[i].real = std::sin(float(i) * 0.1f);
        input[i].imag = std::cos(float(i) * 0.2f);
    }

    std::vector<admf_complex> original = input;

    // Forward then inverse
    status = admf_plan_execute_forward(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    status = admf_plan_execute_inverse(plan, input.data());
    REQUIRE(status == ADM_SUCCESS);

    // Should recover original
    REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, tolerance));

    // Clean up
    adm_plan_destroy(plan);
}

TEST_CASE("C API plan error handling", "[c_api][plan]") {
    SECTION("Null plan pointer") {
        REQUIRE(adm_plan_both(nullptr, 32) == ADM_ERROR_NULL_POINTER);
    }

    SECTION("Zero size") {
        adm_plan plan = nullptr;
        REQUIRE(adm_plan_both(&plan, 0) == ADM_ERROR_INVALID_SIZE);
    }

    SECTION("Null plan in execute") {
        std::vector<adm_complex> data(32);
        REQUIRE(adm_plan_execute_forward(nullptr, data.data()) == ADM_ERROR_INVALID_PLAN);
        REQUIRE(adm_plan_execute_inverse(nullptr, data.data()) == ADM_ERROR_INVALID_PLAN);
    }

    SECTION("Null data in execute") {
        adm_plan plan = nullptr;
        adm_plan_both(&plan, 32);
        REQUIRE(adm_plan_execute_forward(plan, nullptr) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_plan_execute_inverse(plan, nullptr) == ADM_ERROR_NULL_POINTER);
        adm_plan_destroy(plan);
    }
}

TEST_CASE("C API plan reuse", "[c_api][plan]") {
    const size_t N = 32;
    const double tolerance = 1e-10;

    // Create bidirectional plan
    adm_plan plan = nullptr;
    adm_status status = adm_plan_both(&plan, N);
    REQUIRE(status == ADM_SUCCESS);

    // Use plan multiple times with different data
    for (size_t batch = 0; batch < 5; ++batch) {
        std::vector<adm_complex> input(N);
        for (size_t i = 0; i < N; ++i) {
            input[i].real = std::sin(double(batch + i) * 0.1);
            input[i].imag = std::cos(double(batch + i) * 0.2);
        }

        std::vector<adm_complex> original = input;

        // Forward then inverse
        status = adm_plan_execute_forward(plan, input.data());
        REQUIRE(status == ADM_SUCCESS);

        status = adm_plan_execute_inverse(plan, input.data());
        REQUIRE(status == ADM_SUCCESS);

        // Should recover original
        REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, tolerance));
    }

    adm_plan_destroy(plan);
}

// Sizes spanning pow2, primes, composites, a prime power, and Bluestein primes.
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

        REQUIRE(adm_forward(input.data(), N) == ADM_SUCCESS);
        REQUIRE(adm_inverse(input.data(), N) == ADM_SUCCESS);

        REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, fft_tol<double>(N)));
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

        REQUIRE(admf_forward(input.data(), N) == ADM_SUCCESS);
        REQUIRE(admf_inverse(input.data(), N) == ADM_SUCCESS);

        REQUIRE(c_arrays_approx_equal(input.data(), original.data(), N, fft_tol<float>(N)));
    }
}

// ============================================================================
// N-D C API
// ============================================================================

TEST_CASE("C API N-D round-trip identity (double)", "[c_api][nd]") {
    // 2D and 3D shapes spanning catalog / 7-smooth / fallback column routes.
    const std::vector<std::vector<size_t>> shapes = {
        {8, 8}, {16, 32}, {12, 15}, {17, 19}, {31, 9}, {4, 5, 6}, {8, 8, 8}};
    for (const auto& shape : shapes) {
        size_t Ntot = 1;
        for (size_t e : shape) Ntot *= e;
        CAPTURE(Ntot);

        std::vector<adm_complex> input(Ntot);
        for (size_t i = 0; i < Ntot; ++i) {
            input[i].real = std::sin(double(i) * 0.3 + 0.1);
            input[i].imag = std::cos(double(i) * 0.7 - 0.2);
        }
        std::vector<adm_complex> original = input;

        REQUIRE(adm_forward_nd(input.data(), shape.data(), shape.size()) == ADM_SUCCESS);
        REQUIRE(adm_inverse_nd(input.data(), shape.data(), shape.size()) == ADM_SUCCESS);
        REQUIRE(c_arrays_approx_equal(input.data(), original.data(), Ntot, fft_tol<double>(Ntot)));
    }
}

TEST_CASE("C API N-D round-trip identity (float)", "[c_api][nd]") {
    const std::vector<std::vector<size_t>> shapes = {{16, 16}, {12, 20}, {8, 4, 2}};
    for (const auto& shape : shapes) {
        size_t Ntot = 1;
        for (size_t e : shape) Ntot *= e;
        CAPTURE(Ntot);

        std::vector<admf_complex> input(Ntot);
        for (size_t i = 0; i < Ntot; ++i) {
            input[i].real = std::sin(float(i) * 0.3f + 0.1f);
            input[i].imag = std::cos(float(i) * 0.7f - 0.2f);
        }
        std::vector<admf_complex> original = input;

        REQUIRE(admf_forward_nd(input.data(), shape.data(), shape.size()) == ADM_SUCCESS);
        REQUIRE(admf_inverse_nd(input.data(), shape.data(), shape.size()) == ADM_SUCCESS);
        REQUIRE(c_arrays_approx_equal(input.data(), original.data(), Ntot, fft_tol<float>(Ntot)));
    }
}

// A rank-1 axis is the identity along that axis, so an N-D transform of shape
// {1, N} (or {N, 1}) must reproduce the 1D transform of size N — verified
// purely through the C API.
TEST_CASE("C API N-D shape {1,N}/{N,1} matches 1D (double)", "[c_api][nd]") {
    for (size_t N : {8u, 16u, 13u, 31u}) {
        CAPTURE(N);
        std::vector<adm_complex> base(N);
        for (size_t i = 0; i < N; ++i) {
            base[i].real = std::sin(double(i) * 0.5 + 0.2);
            base[i].imag = std::cos(double(i) * 0.4 - 0.3);
        }

        std::vector<adm_complex> one_d = base;
        REQUIRE(adm_forward(one_d.data(), N) == ADM_SUCCESS);

        for (const std::vector<size_t>& shape : {std::vector<size_t>{1, N},
                                                 std::vector<size_t>{N, 1}}) {
            std::vector<adm_complex> nd = base;
            REQUIRE(adm_forward_nd(nd.data(), shape.data(), shape.size()) == ADM_SUCCESS);
            REQUIRE(c_arrays_approx_equal(nd.data(), one_d.data(), N, fft_tol<double>(N)));
        }
    }
}

TEST_CASE("C API N-D reusable plan matches one-shot (double)", "[c_api][nd][plan]") {
    const std::vector<size_t> shape = {16, 16};
    const size_t Ntot = 16 * 16;

    std::vector<adm_complex> base(Ntot);
    for (size_t i = 0; i < Ntot; ++i) {
        base[i].real = std::sin(double(i) * 0.11 + 0.7);
        base[i].imag = std::cos(double(i) * 0.13 - 0.5);
    }

    adm_plan plan = nullptr;
    REQUIRE(adm_plan_nd(&plan, shape.data(), shape.size()) == ADM_SUCCESS);
    REQUIRE(plan != nullptr);
    REQUIRE(adm_plan_size(plan) == Ntot);

    std::vector<adm_complex> via_plan = base;
    std::vector<adm_complex> via_oneshot = base;
    REQUIRE(adm_plan_execute_forward(plan, via_plan.data()) == ADM_SUCCESS);
    REQUIRE(adm_forward_nd(via_oneshot.data(), shape.data(), shape.size()) == ADM_SUCCESS);
    REQUIRE(c_arrays_approx_equal(via_plan.data(), via_oneshot.data(), Ntot, fft_tol<double>(Ntot)));

    // Round-trip through the same reusable plan returns the original.
    REQUIRE(adm_plan_execute_inverse(plan, via_plan.data()) == ADM_SUCCESS);
    REQUIRE(c_arrays_approx_equal(via_plan.data(), base.data(), Ntot, fft_tol<double>(Ntot)));

    adm_plan_destroy(plan);
}

TEST_CASE("C API N-D rejects invalid arguments", "[c_api][nd]") {
    const size_t shape[2] = {8, 8};
    const size_t bad_shape[2] = {8, 0};  // zero extent
    std::vector<adm_complex> data(64);

    REQUIRE(adm_forward_nd(nullptr, shape, 2) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward_nd(data.data(), nullptr, 2) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(data.data(), shape, 0) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(data.data(), bad_shape, 2) == ADM_ERROR_INVALID_SIZE);

    adm_plan plan = nullptr;
    REQUIRE(adm_plan_nd(nullptr, shape, 2) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_nd(&plan, bad_shape, 2) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_plan_nd(&plan, shape, 0) == ADM_ERROR_INVALID_SIZE);
}
