#pragma once

// ============================================================================
// Admiral C API — complex and real FFTs, 1-D and N-D.
//
//   #include <admiral/admiral.h>
//
//   adm_plan p;
//   adm_plan_1d(&p, 1024, NULL);         // 1024 points, default options
//   adm_plan_execute_forward(p, data);   // data: adm_complex[1024], in place
//   adm_plan_execute_inverse(p, data);   // divides by 1024
//   adm_plan_destroy(p);
//
// Naming. Double precision keeps the plain name, single precision adds an `f`
// to the prefix: adm_forward / admf_forward. The two sets are otherwise the
// same.
//
// Layout. Contiguous row-major, last axis fastest — the same layout as FFTW.
// Strided and non-contiguous data is not supported.
//
// Sign and scale. Forward uses exp(-2*pi*i*k*n/N) and is unscaled. Inverse uses
// exp(+2*pi*i*k*n/N) and divides by the element count, so forward then inverse
// gives back the input. FFTW leaves both directions unscaled; when porting FFTW
// code, either scale the output or use <admiral/fftw3.h>, which keeps FFTW's
// convention.
//
// Errors. Every call returns an adm_status, and ADM_SUCCESS is 0. The result is
// nodiscard, so ignoring it is a warning. adm_error_string() turns a status
// into text.
//
// Options. Everything that builds a plan -- the one-shots included -- takes a
// trailing `const adm_options*`. NULL means the defaults, so a caller who wants
// none of them passes nothing.
//
// Threads. nthreads = 1 is serial, 0 means "one per physical core". A plan owns
// its workers and reuses them on every execute. Do not execute one plan from
// two threads at the same time; separate plans are independent.
// ============================================================================

#include <stddef.h>

// The build hides every symbol by default, so the public C entry points opt back
// in explicitly.
#if defined(_WIN32) || defined(__CYGWIN__)
#  define ADM_C_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_C_API __attribute__((visibility("default")))
#else
#  define ADM_C_API
#endif

// [[nodiscard]] is not valid C before C23, so older compilers get the GNU
// attribute instead.
#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define ADM_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_NODISCARD __attribute__((warn_unused_result))
#else
#  define ADM_NODISCARD
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interleaved complex, layout-compatible with std::complex and FFTW's
// double[2], so an existing buffer casts instead of copying.
typedef struct { float real; float imag; } admf_complex;
typedef struct { double real; double imag; } adm_complex;

typedef enum {
    ADM_SUCCESS = 0,
    ADM_ERROR_NULL_POINTER = -1,
    ADM_ERROR_INVALID_SIZE = -2,   // null shape, zero size/extent, or extents overflow size_t
    ADM_ERROR_OUT_OF_MEMORY = -3,
    ADM_ERROR_INVALID_PLAN = -4,   // null plan, or wrong precision for this call
    ADM_ERROR_INVALID_OPTION = -5  // an adm_options field is outside its range
} adm_status;

// How hard the planner works before it commits to a route. estimate ranks
// candidates with a cost model; the measuring efforts time them, which costs plan
// time and is not reproducible across machines.
typedef enum {
    ADM_EFFORT_ESTIMATE = 0,
    ADM_EFFORT_AUTOMATIC = 1,
    ADM_EFFORT_MEASURE = 2
} adm_effort;

// Everything a plan is built with. Pass NULL for the defaults, or name every field
// -- C leaves the rest indeterminate, and a zeroed struct is not the default either:
// nthreads = 0 asks for one worker per physical core.
//   const adm_options o = {.nthreads = 8, .eff = ADM_EFFORT_AUTOMATIC, .debug = 0};
typedef struct {
    size_t nthreads;   // 1 serial, 0 one per physical core
    adm_effort eff;
    unsigned debug;    // stderr trace verbosity: 0 silent, 1 route, 2 route + shape
} adm_options;

// Never null; unknown codes give "Unknown error".
ADM_NODISCARD ADM_C_API const char* adm_error_string(adm_status status);

// ============================================================================
// One-shot transforms
//
// These plan, transform and discard the plan. Use them for a size transformed
// once; for repeated transforms build a plan instead, which keeps the twiddle
// tables and the worker threads alive between calls. Zero sizes are rejected;
// the C++ one-shots treat an empty span as a no-op, this surface does not.
// ============================================================================

// 1-D, in place. `size` is the number of complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_forward(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward(adm_complex* data, size_t size,
                                               const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse(adm_complex* data, size_t size,
                                               const adm_options* opts);

// N-D, in place. `shape` lists `ndim` extents and `data` holds their product in
// complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_forward_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);

// ============================================================================
// Real transforms (r2c / c2r), N-D, out of place
//
// `shape` gives the extents of the REAL tensor. A real signal has a conjugate-
// symmetric spectrum, so only half of it is stored: the complex tensor has the
// same extents except the innermost, which becomes shape[ndim-1]/2 + 1. Its
// element count is therefore
//
//     shape[0] * ... * shape[ndim-2] * (shape[ndim-1]/2 + 1)
//
// This is FFTW's and ducc0's half-spectrum layout, so buffers port directly.
// r2c is unscaled and c2r divides by the real element count. c2r overwrites its
// input spectrum.
// ============================================================================

ADM_NODISCARD ADM_C_API adm_status admf_r2c_nd(const float* in, admf_complex* out,
                                               const size_t* shape, size_t ndim,
                                               const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_r2c_nd(const double* in, adm_complex* out,
                                              const size_t* shape, size_t ndim,
                                              const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_c2r_nd(admf_complex* spec, float* out, const size_t* shape,
                                               size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_c2r_nd(adm_complex* spec, double* out, const size_t* shape,
                                              size_t ndim, const adm_options* opts);

// ============================================================================
// Plans (repeated transforms)
//
// A plan holds the twiddle tables, the routing decision and the worker threads
// for one shape. It runs both directions, so create it once and call
// execute_forward and execute_inverse any number of times. Destroy it with
// adm_plan_destroy.
// ============================================================================

typedef struct adm_plan_s* adm_plan;

// 1-D plan over `n` complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);

// N-D plan over `shape[ndim]`. ndim == 1 is the same plan adm_plan_1d builds.
ADM_NODISCARD ADM_C_API adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                               const adm_options* opts);

// Transform `data` in place. The precision must match the one the plan was
// created with, or the call returns ADM_ERROR_INVALID_PLAN.
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data);
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data);

// Total complex element count the plan expects. 0 for a null plan.
ADM_NODISCARD ADM_C_API size_t adm_plan_size(adm_plan plan);

// Safe on a null plan.
ADM_C_API void adm_plan_destroy(adm_plan plan);

#ifdef __cplusplus
}
#endif
