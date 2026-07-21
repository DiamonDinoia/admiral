#pragma once

#include <stddef.h>

// Public C API symbols are explicitly exported (the build sets hidden visibility
// project-wide; these must stay visible to consumers).
#if defined(_WIN32) || defined(__CYGWIN__)
#  define ADM_C_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_C_API __attribute__((visibility("default")))
#else
#  define ADM_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Complex number types (layout-compatible with std::complex)
typedef struct { float real; float imag; } admf_complex;
typedef struct { double real; double imag; } adm_complex;

// Error codes
typedef enum {
    ADM_SUCCESS = 0,
    ADM_ERROR_NULL_POINTER = -1,
    ADM_ERROR_INVALID_SIZE = -2,
    ADM_ERROR_OUT_OF_MEMORY = -3,
    ADM_ERROR_INVALID_PLAN = -4
} adm_status;

// Human-readable error messages
ADM_C_API const char* adm_error_string(adm_status status);

// ============================================================================
// Core Transform Functions (in-place, user manages memory)
// ============================================================================

// Forward FFT (modifies data in-place)
ADM_C_API adm_status admf_forward(admf_complex* data, size_t size);
ADM_C_API adm_status adm_forward(adm_complex* data, size_t size);

// Inverse FFT (modifies data in-place)
ADM_C_API adm_status admf_inverse(admf_complex* data, size_t size);
ADM_C_API adm_status adm_inverse(adm_complex* data, size_t size);

// ============================================================================
// N-D Transform Functions (row-column algorithm; in-place)
//
// `data` is a contiguous row-major tensor of shape[0]*...*shape[ndim-1] complex
// values, last axis fastest. `shape` lists the `ndim` extents. Normalization
// matches the 1D API: forward is unscaled, inverse divides by the total element
// count, so forward then inverse is the identity. Strided / non-contiguous
// layouts are not supported.
// ============================================================================

// One-shot forward N-D FFT (unscaled), in-place.
ADM_C_API adm_status admf_forward_nd(admf_complex* data, const size_t* shape, size_t ndim);
ADM_C_API adm_status adm_forward_nd(adm_complex* data, const size_t* shape, size_t ndim);

// One-shot inverse N-D FFT (scaled by 1/Ntot), in-place.
ADM_C_API adm_status admf_inverse_nd(admf_complex* data, const size_t* shape, size_t ndim);
ADM_C_API adm_status adm_inverse_nd(adm_complex* data, const size_t* shape, size_t ndim);

// ============================================================================
// Real-to-complex N-D Transforms (r2c / c2r), out-of-place.
//
// `shape` lists the `ndim` real extents (row-major, last axis fastest, real).
// The complex half-spectrum tensor has the innermost extent shape[ndim-1]
// replaced by shape[ndim-1]/2 + 1; its total element count is the product of the
// other extents times that (layout matches FFTW/ducc0). Normalization matches
// the complex API: forward (r2c) is unscaled, inverse (c2r) divides by the total
// real element count, so r2c then c2r is the identity. The c2r input is consumed.
// ============================================================================

// Forward N-D r2c (unscaled): real `in` -> complex half-spectrum `out`.
ADM_C_API adm_status admf_r2c_nd(const float* in, admf_complex* out, const size_t* shape, size_t ndim);
ADM_C_API adm_status adm_r2c_nd(const double* in, adm_complex* out, const size_t* shape, size_t ndim);

// Inverse N-D c2r (scaled by 1/Ntot): complex half-spectrum `spec` (consumed) -> real `out`.
ADM_C_API adm_status admf_c2r_nd(admf_complex* spec, float* out, const size_t* shape, size_t ndim);
ADM_C_API adm_status adm_c2r_nd(adm_complex* spec, double* out, const size_t* shape, size_t ndim);

// ============================================================================
// Plan-Based Interface (for repeated transforms)
// ============================================================================

// Opaque plan handle
typedef struct adm_plan_s* adm_plan;

// Create bidirectional plan (both forward and inverse)
ADM_C_API adm_status admf_plan_both(adm_plan* plan, size_t size);
ADM_C_API adm_status adm_plan_both(adm_plan* plan, size_t size);

// Threaded variants: nthreads caches worker threads in the plan, reused across
// every execute (nthreads = 1 is serial, 0 = auto = hardware concurrency capped).
// Kept as separate symbols so the serial ones above stay ABI-stable.
ADM_C_API adm_status admf_plan_both_threaded(adm_plan* plan, size_t size, size_t nthreads);
ADM_C_API adm_status adm_plan_both_threaded(adm_plan* plan, size_t size, size_t nthreads);

// Create a bidirectional N-D plan (reusable; contiguous row-major). `shape`
// lists `ndim` extents, last axis fastest. Execute with the existing
// adm_plan_execute_forward_* / adm_plan_execute_inverse_* functions; a 1D shape
// is equivalent to adm_plan_both_*. adm_plan_size returns the total element count.
ADM_C_API adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim);
ADM_C_API adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim);

// Threaded N-D plan variants (see adm_plan_both_*_threaded for nthreads semantics).
ADM_C_API adm_status admf_plan_nd_threaded(adm_plan* plan, const size_t* shape, size_t ndim, size_t nthreads);
ADM_C_API adm_status adm_plan_nd_threaded(adm_plan* plan, const size_t* shape, size_t ndim, size_t nthreads);

// Execute specific direction for bidirectional plans
ADM_C_API adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data);
ADM_C_API adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data);
ADM_C_API adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data);
ADM_C_API adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data);

// Destroy plan
ADM_C_API void adm_plan_destroy(adm_plan plan);

// Get plan size
ADM_C_API size_t adm_plan_size(adm_plan plan);

#ifdef __cplusplus
}
#endif
