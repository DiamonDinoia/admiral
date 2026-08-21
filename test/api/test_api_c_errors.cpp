// C API error paths: every adm_status string, and the argument validation on the
// 1-D / N-D / plan / r2c-c2r entry points. The happy paths live in test_api_c.cpp;
// these are the branches a working caller never takes.

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <admiral/admiral.h>

#include "utils/reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

using namespace Catch::Matchers;

TEST_CASE("C API error strings cover every status", "[coverage][c_api]") {
    REQUIRE(std::string(adm_error_string(ADM_SUCCESS)) == "Success");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_NULL_POINTER)) == "Null pointer argument");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_SIZE)) == "Invalid size (must be > 0)");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_OUT_OF_MEMORY)) == "Out of memory");
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_PLAN)) == "Invalid plan");
    // 3 is inside adm_status' value range (enumerators -4..0 => 3 signed bits => -4..3)
    // but is not an enumerator, so this hits the default branch with defined behaviour.
    REQUIRE(std::string(adm_error_string(static_cast<adm_status>(3))) == "Unknown error");
}

TEST_CASE("C API transform argument validation", "[coverage][c_api]") {
    admf_complex f{1, 0};
    adm_complex d{1, 0};

    REQUIRE(admf_forward(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_forward(&f, 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_inverse(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_inverse(&f, 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward(&d, 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_inverse(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_inverse(&d, 0, NULL) == ADM_ERROR_INVALID_SIZE);

    // A valid single-element float transform (exercises the success path too).
    REQUIRE(admf_forward(&f, 1, NULL) == ADM_SUCCESS);
}

TEST_CASE("C API N-D and plan argument validation", "[coverage][c_api]") {
    const std::array<size_t, 2> shape = {4, 4};
    const std::array<size_t, 2> bad = {4, 0};
    std::vector<adm_complex> buf(16);

    REQUIRE(adm_forward_nd(nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_forward_nd(buf.data(), nullptr, 2, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(buf.data(), shape.data(), 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_forward_nd(buf.data(), bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_inverse_nd(nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);

    // Plan constructors.
    adm_plan p = nullptr;
    REQUIRE(admf_plan_1d(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_plan_1d(&p, 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_plan_1d(nullptr, 4, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_1d(&p, 0, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(admf_plan_nd(nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(admf_plan_nd(&p, bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);
    REQUIRE(adm_plan_nd(nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
    REQUIRE(adm_plan_nd(&p, bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);

    // Execute with a null / null-data / wrong-type plan.
    admf_complex fdata{1, 0};
    adm_complex ddata{1, 0};
    REQUIRE(admf_plan_execute_forward(nullptr, &fdata) == ADM_ERROR_INVALID_PLAN);
    REQUIRE(adm_plan_execute_inverse(nullptr, &ddata) == ADM_ERROR_INVALID_PLAN);

    adm_plan dplan = nullptr;
    REQUIRE(adm_plan_1d(&dplan, 4, NULL) == ADM_SUCCESS);
    REQUIRE(admf_plan_execute_forward(dplan, &fdata) == ADM_ERROR_INVALID_PLAN);   // wrong type
    REQUIRE(admf_plan_execute_inverse(dplan, &fdata) == ADM_ERROR_INVALID_PLAN);   // wrong type
    REQUIRE(adm_plan_execute_forward(dplan, nullptr) == ADM_ERROR_NULL_POINTER);   // null data
    REQUIRE(adm_plan_execute_inverse(dplan, nullptr) == ADM_ERROR_NULL_POINTER);
    adm_plan_destroy(dplan);

    REQUIRE(adm_plan_size(nullptr) == 0);
}

TEMPLATE_TEST_CASE("C API r2c/c2r round-trip and validation", "[coverage][c_api][real]",
                   float, double) {
    using T = TestType;
    const std::array<size_t, 2> shape = {4, 6};
    const size_t real_n = 4 * 6;
    const size_t cplx_n = 4 * (6 / 2 + 1);

    std::vector<T> in(real_n), out(real_n);
    for (size_t i = 0; i < real_n; ++i) in[i] = std::cos(T(0.4) * T(i)) + T(1);

    using C = std::conditional_t<std::is_same_v<T, float>, admf_complex, adm_complex>;
    std::vector<C> spec(cplx_n);

    adm_status s_fwd, s_inv;
    if constexpr (std::is_same_v<T, float>) {
        s_fwd = admf_r2c_nd(in.data(), spec.data(), shape.data(), 2, NULL);
        s_inv = admf_c2r_nd(spec.data(), out.data(), shape.data(), 2, NULL);
    } else {
        s_fwd = adm_r2c_nd(in.data(), spec.data(), shape.data(), 2, NULL);
        s_inv = adm_c2r_nd(spec.data(), out.data(), shape.data(), 2, NULL);
    }
    REQUIRE(s_fwd == ADM_SUCCESS);
    REQUIRE(s_inv == ADM_SUCCESS);
    require_close(out, in, fft_tol<T>(2.0));  // r2c then c2r compounds

    // Argument validation (null + degenerate shape).
    const std::array<size_t, 2> bad = {4, 0};
    if constexpr (std::is_same_v<T, float>) {
        REQUIRE(admf_r2c_nd(nullptr, spec.data(), shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_r2c_nd(in.data(), nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_r2c_nd(in.data(), spec.data(), bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(admf_c2r_nd(nullptr, out.data(), shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(admf_c2r_nd(spec.data(), out.data(), bad.data(), 2, NULL)
                == ADM_ERROR_INVALID_SIZE);
    } else {
        REQUIRE(adm_r2c_nd(nullptr, spec.data(), shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_r2c_nd(in.data(), nullptr, shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_r2c_nd(in.data(), spec.data(), bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);
        REQUIRE(adm_c2r_nd(nullptr, out.data(), shape.data(), 2, NULL) == ADM_ERROR_NULL_POINTER);
        REQUIRE(adm_c2r_nd(spec.data(), out.data(), bad.data(), 2, NULL) == ADM_ERROR_INVALID_SIZE);
    }
}

// adm_options is the C mirror of admiral::options: same three knobs, NULL for the
// defaults. The C layer must reject an eff outside the enum rather than cast it,
// because C cannot stop a caller from inventing one.
TEST_CASE("adm_options reaches threads, effort and debug", "[c_api][options]") {
    constexpr size_t N = 240;
    std::vector<adm_complex> data(N);
    for (size_t i = 0; i < N; ++i) data[i] = {double(i % 13) - 6.0, double(i % 7) - 3.0};
    const auto ref = data;

    const adm_options threaded = {.nthreads = 2, .eff = ADM_EFFORT_ESTIMATE, .debug = 0};
    const adm_options measured = {.nthreads = 1, .eff = ADM_EFFORT_AUTOMATIC, .debug = 0};
    const adm_options traced = {.nthreads = 1, .eff = ADM_EFFORT_ESTIMATE, .debug = 2};

    for (const adm_options* o : {&threaded, &measured, &traced}) {
        REQUIRE(adm_forward(data.data(), N, o) == ADM_SUCCESS);
        REQUIRE(adm_inverse(data.data(), N, o) == ADM_SUCCESS);
        for (size_t i = 0; i < N; ++i) {
            REQUIRE_THAT(data[i].real, Catch::Matchers::WithinAbs(ref[i].real, 1e-9));
            REQUIRE_THAT(data[i].imag, Catch::Matchers::WithinAbs(ref[i].imag, 1e-9));
        }
    }

    adm_plan plan = nullptr;
    REQUIRE(adm_plan_1d(&plan, N, &measured) == ADM_SUCCESS);
    REQUIRE(adm_plan_size(plan) == N);
    adm_plan_destroy(plan);

    // 3 is the largest value adm_effort's range can hold and is not an enumerator, so it
    // is the only unnamed value a cast can produce without an unspecified result.
    const adm_options bad = {.nthreads = 1, .eff = static_cast<adm_effort>(3), .debug = 0};
    REQUIRE(adm_forward(data.data(), N, &bad) == ADM_ERROR_INVALID_OPTION);
    REQUIRE(adm_plan_1d(&plan, N, &bad) == ADM_ERROR_INVALID_OPTION);
    REQUIRE(std::string(adm_error_string(ADM_ERROR_INVALID_OPTION)) != "Unknown error");
}
