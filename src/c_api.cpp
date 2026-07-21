#include "admiral/admiral.h"
#include "admiral/admiral.hpp"
#include <complex>
#include <span>
#include <new>
#include <cstring>

// ============================================================================
// Error Messages
// ============================================================================

const char* adm_error_string(adm_status status) {
    switch (status) {
        case ADM_SUCCESS: return "Success";
        case ADM_ERROR_NULL_POINTER: return "Null pointer argument";
        case ADM_ERROR_INVALID_SIZE: return "Invalid size (must be > 0)";
        case ADM_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case ADM_ERROR_INVALID_PLAN: return "Invalid plan";
        default: return "Unknown error";
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

// Convert C complex to C++ complex (zero-copy view). Internal to this TU.
namespace {
template<typename T>
std::span<std::complex<T>> to_cpp_span(void* data, size_t size) {
    return std::span<std::complex<T>>(
        reinterpret_cast<std::complex<T>*>(data), size
    );
}

// Validate an N-D shape and return the total element count (product of extents).
// Returns 0 to signal an invalid shape: a null pointer, zero rank, or any
// zero/degenerate extent — all of which the caller maps to ADM_ERROR_INVALID_SIZE.
size_t nd_total_or_zero(const size_t* shape, size_t ndim) {
    if (shape == nullptr || ndim == 0) return 0;
    size_t total = 1;
    for (size_t d = 0; d < ndim; ++d) {
        if (shape[d] == 0) return 0;
        total *= shape[d];
    }
    return total;
}

// One-shot N-D transform (single direction), mirroring admiral::forward /
// inverse: build a runtime-rank engine for `shape` and execute in place.
template<typename T>
adm_status run_nd(void* data, const size_t* shape, size_t ndim, bool forward) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (nd_total_or_zero(shape, ndim) == 0) return ADM_ERROR_INVALID_SIZE;
    try {
        admiral::detail::nd_runtime_plan<T>(std::span<const size_t>(shape, ndim), forward)
            .execute(reinterpret_cast<std::complex<T>*>(data));
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}
}  // anonymous namespace

// ============================================================================
// Core Transform Functions
// ============================================================================

adm_status admf_forward(admf_complex* data, size_t size) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (size == 0) return ADM_ERROR_INVALID_SIZE;

    try {
        auto span = to_cpp_span<float>(data, size);
        admiral::forward(span, span);
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

adm_status adm_forward(adm_complex* data, size_t size) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (size == 0) return ADM_ERROR_INVALID_SIZE;

    try {
        auto span = to_cpp_span<double>(data, size);
        admiral::forward(span, span);
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

adm_status admf_inverse(admf_complex* data, size_t size) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (size == 0) return ADM_ERROR_INVALID_SIZE;

    try {
        auto span = to_cpp_span<float>(data, size);
        admiral::inverse(span, span);
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

adm_status adm_inverse(adm_complex* data, size_t size) {
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (size == 0) return ADM_ERROR_INVALID_SIZE;

    try {
        auto span = to_cpp_span<double>(data, size);
        admiral::inverse(span, span);
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

// ============================================================================
// N-D Transform Functions (one-shot, in-place)
// ============================================================================

adm_status admf_forward_nd(admf_complex* data, const size_t* shape, size_t ndim) {
    return run_nd<float>(data, shape, ndim, /*forward=*/true);
}

adm_status adm_forward_nd(adm_complex* data, const size_t* shape, size_t ndim) {
    return run_nd<double>(data, shape, ndim, /*forward=*/true);
}

adm_status admf_inverse_nd(admf_complex* data, const size_t* shape, size_t ndim) {
    return run_nd<float>(data, shape, ndim, /*forward=*/false);
}

adm_status adm_inverse_nd(adm_complex* data, const size_t* shape, size_t ndim) {
    return run_nd<double>(data, shape, ndim, /*forward=*/false);
}

// ============================================================================
// Real-to-complex N-D Transforms (r2c / c2r, one-shot, out-of-place)
// ============================================================================

namespace {
template<typename T>
adm_status run_r2c(const void* in, void* out, const size_t* shape, size_t ndim) {
    if (in == nullptr || out == nullptr) return ADM_ERROR_NULL_POINTER;
    if (nd_total_or_zero(shape, ndim) == 0) return ADM_ERROR_INVALID_SIZE;
    try {
        admiral::detail::nd_real_plan<T>(std::span<const size_t>(shape, ndim))
            .forward(reinterpret_cast<const T*>(in), reinterpret_cast<std::complex<T>*>(out));
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) { return ADM_ERROR_OUT_OF_MEMORY; }
    catch (...) { return ADM_ERROR_INVALID_SIZE; }
}

template<typename T>
adm_status run_c2r(void* spec, void* out, const size_t* shape, size_t ndim) {
    if (spec == nullptr || out == nullptr) return ADM_ERROR_NULL_POINTER;
    if (nd_total_or_zero(shape, ndim) == 0) return ADM_ERROR_INVALID_SIZE;
    try {
        admiral::detail::nd_real_plan<T>(std::span<const size_t>(shape, ndim))
            .inverse(reinterpret_cast<std::complex<T>*>(spec), reinterpret_cast<T*>(out));
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) { return ADM_ERROR_OUT_OF_MEMORY; }
    catch (...) { return ADM_ERROR_INVALID_SIZE; }
}
}  // anonymous namespace

adm_status admf_r2c_nd(const float* in, admf_complex* out, const size_t* shape, size_t ndim) {
    return run_r2c<float>(in, out, shape, ndim);
}
adm_status adm_r2c_nd(const double* in, adm_complex* out, const size_t* shape, size_t ndim) {
    return run_r2c<double>(in, out, shape, ndim);
}
adm_status admf_c2r_nd(admf_complex* spec, float* out, const size_t* shape, size_t ndim) {
    return run_c2r<float>(spec, out, shape, ndim);
}
adm_status adm_c2r_nd(adm_complex* spec, double* out, const size_t* shape, size_t ndim) {
    return run_c2r<double>(spec, out, shape, ndim);
}

// ============================================================================
// Plan Implementation
// ============================================================================

// Type-erased plan wrapper
struct adm_plan_s {
    enum class Type { BOTH_FLOAT, BOTH_DOUBLE };
    Type type;
    void* plan_ptr;
    size_t size;

    ~adm_plan_s() {
        switch (type) {
            case Type::BOTH_FLOAT:
                delete static_cast<admiral::plan<float>*>(plan_ptr);
                break;
            case Type::BOTH_DOUBLE:
                delete static_cast<admiral::plan<double>*>(plan_ptr);
                break;
        }
    }
};

// Shared bidirectional-plan factory for every (precision, rank, nthreads) combo.
// The unified admiral::plan<T> IS the N-D engine (runtime shape), and a 1D plan is
// just a rank-1 shape — so `both` and `nd` are the same construction over a
// `shape[ndim]` span, differing only in ndim. nthreads is passed straight to
// admiral::plan (1 = serial, 0 = auto); the BOTH_* wrapper type and total element
// count drive execution/destruction through the existing paths unchanged.
namespace {
template<typename T>
adm_status make_plan(adm_plan* plan, const size_t* shape, size_t ndim, size_t total,
                     size_t nthreads, adm_plan_s::Type type) {
    if (plan == nullptr) return ADM_ERROR_NULL_POINTER;
    if (total == 0) return ADM_ERROR_INVALID_SIZE;
    try {
        auto cpp_plan = new admiral::plan<T>(std::span<const size_t>(shape, ndim), nthreads);
        *plan = new adm_plan_s{type, cpp_plan, total};
        return ADM_SUCCESS;
    } catch (const std::bad_alloc&) {
        return ADM_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}
}  // anonymous namespace

// Bidirectional 1D plans (serial + threaded). A 1D plan is a rank-1 shape.
adm_status admf_plan_both(adm_plan* plan, size_t size) {
    return make_plan<float>(plan, &size, 1, size, 1, adm_plan_s::Type::BOTH_FLOAT);
}
adm_status adm_plan_both(adm_plan* plan, size_t size) {
    return make_plan<double>(plan, &size, 1, size, 1, adm_plan_s::Type::BOTH_DOUBLE);
}
adm_status admf_plan_both_threaded(adm_plan* plan, size_t size, size_t nthreads) {
    return make_plan<float>(plan, &size, 1, size, nthreads, adm_plan_s::Type::BOTH_FLOAT);
}
adm_status adm_plan_both_threaded(adm_plan* plan, size_t size, size_t nthreads) {
    return make_plan<double>(plan, &size, 1, size, nthreads, adm_plan_s::Type::BOTH_DOUBLE);
}

// Bidirectional N-D plans (serial + threaded).
adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim) {
    return make_plan<float>(plan, shape, ndim, nd_total_or_zero(shape, ndim), 1,
                            adm_plan_s::Type::BOTH_FLOAT);
}
adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim) {
    return make_plan<double>(plan, shape, ndim, nd_total_or_zero(shape, ndim), 1,
                             adm_plan_s::Type::BOTH_DOUBLE);
}
adm_status admf_plan_nd_threaded(adm_plan* plan, const size_t* shape, size_t ndim, size_t nthreads) {
    return make_plan<float>(plan, shape, ndim, nd_total_or_zero(shape, ndim), nthreads,
                            adm_plan_s::Type::BOTH_FLOAT);
}
adm_status adm_plan_nd_threaded(adm_plan* plan, const size_t* shape, size_t ndim, size_t nthreads) {
    return make_plan<double>(plan, shape, ndim, nd_total_or_zero(shape, ndim), nthreads,
                             adm_plan_s::Type::BOTH_DOUBLE);
}

// Execute forward (for bidirectional plans)
adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data) {
    if (plan == nullptr) return ADM_ERROR_INVALID_PLAN;
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (plan->type != adm_plan_s::Type::BOTH_FLOAT) return ADM_ERROR_INVALID_PLAN;

    try {
        auto span = to_cpp_span<float>(data, plan->size);
        static_cast<admiral::plan<float>*>(plan->plan_ptr)->forward(span);
        return ADM_SUCCESS;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data) {
    if (plan == nullptr) return ADM_ERROR_INVALID_PLAN;
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (plan->type != adm_plan_s::Type::BOTH_DOUBLE) return ADM_ERROR_INVALID_PLAN;

    try {
        auto span = to_cpp_span<double>(data, plan->size);
        static_cast<admiral::plan<double>*>(plan->plan_ptr)->forward(span);
        return ADM_SUCCESS;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

// Execute inverse (for bidirectional plans)
adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data) {
    if (plan == nullptr) return ADM_ERROR_INVALID_PLAN;
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (plan->type != adm_plan_s::Type::BOTH_FLOAT) return ADM_ERROR_INVALID_PLAN;

    try {
        auto span = to_cpp_span<float>(data, plan->size);
        static_cast<admiral::plan<float>*>(plan->plan_ptr)->inverse(span);
        return ADM_SUCCESS;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data) {
    if (plan == nullptr) return ADM_ERROR_INVALID_PLAN;
    if (data == nullptr) return ADM_ERROR_NULL_POINTER;
    if (plan->type != adm_plan_s::Type::BOTH_DOUBLE) return ADM_ERROR_INVALID_PLAN;

    try {
        auto span = to_cpp_span<double>(data, plan->size);
        static_cast<admiral::plan<double>*>(plan->plan_ptr)->inverse(span);
        return ADM_SUCCESS;
    } catch (...) {
        return ADM_ERROR_INVALID_SIZE;
    }
}

// Destroy plan
void adm_plan_destroy(adm_plan plan) {
    delete plan;
}

// Get plan size
size_t adm_plan_size(adm_plan plan) {
    if (plan == nullptr) return 0;
    return plan->size;
}
