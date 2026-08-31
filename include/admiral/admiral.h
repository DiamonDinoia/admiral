#pragma once

// ============================================================================
// Admiral C API: complex and real FFTs, 1-D and N-D.
//
//   #include <admiral/admiral.h>
//
//   adm_plan p;
//   adm_plan_1d(&p, 1024, NULL);         // 1024 points, default options
//   adm_plan_execute_forward(p, data);   // data: adm_complex[1024], in place
//   adm_plan_execute_inverse(p, data);   // divides by 1024
//   adm_plan_destroy(p);
//
// Naming. Double precision keeps the plain name, single precision adds an f to
// the prefix: `adm_forward` / `admf_forward`. The two sets are otherwise the
// same.
//
// Layout. Contiguous row-major, last axis fastest, the same layout as FFTW. No
// strided or non-contiguous data.
//
// Sign and scale. Forward uses exp(-2*pi*i*k*n/N), unscaled; inverse uses
// exp(+2*pi*i*k*n/N) and divides by the element count, so a round-trip returns
// the input. FFTW leaves both directions unscaled: when porting FFTW code,
// scale the output or use `<admiral/fftw3.h>`, which keeps FFTW's convention.
//
// Errors. Every call returns an `adm_status`, and `ADM_SUCCESS` is 0. The
// result is nodiscard. `adm_error_string()` turns a status into text.
//
// Options. Everything that builds a plan, one-shots included, takes a trailing
// const `adm_options*`; `NULL` means the defaults.
//
// Threads. `nthreads` = 0 (default) is auto: serial for small transforms,
// capped at one worker per physical core. A plan owns and reuses its workers
// across executes. Do not execute one plan from two threads at once; separate
// plans are independent.
// ============================================================================

#include <stddef.h>

// The build hides every symbol by default, so the public C entry points opt back
// in.
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

/// Interleaved complex, layout-compatible with `std::complex` and FFTW's
/// `double[2]`, so an existing buffer casts instead of a copy.
typedef struct { float real; float imag; } admf_complex;
typedef struct { double real; double imag; } adm_complex;

typedef enum {
    ADM_SUCCESS = 0,
    ADM_ERROR_NULL_POINTER = -1,
    ADM_ERROR_INVALID_SIZE = -2,   ///< null shape, zero size/extent, size mismatch,
                                   //  or an extent product that overflows `size_t`
    ADM_ERROR_OUT_OF_MEMORY = -3,
    ADM_ERROR_INVALID_PLAN = -4,   ///< null plan, or wrong precision for this call
    ADM_ERROR_INVALID_OPTION = -5, ///< an `adm_options` field is outside its range
    ADM_ERROR_INTERNAL = -6        ///< a fault outside the caller's arguments; a failed
                                   //  plan reports the detail via `adm_plan_error_message()`
} adm_status;

/// How hard the planner works before it commits to a route. `estimate` ranks
/// candidates with a cost model; the measuring efforts time them, which costs
/// plan time and is not reproducible across machines.
typedef enum {
    ADM_EFFORT_ESTIMATE = 0,
    ADM_EFFORT_AUTOMATIC = 1,
    ADM_EFFORT_MEASURE = 2
} adm_effort;

/// Everything a plan is built with. Pass `NULL` for the defaults, or name every
/// field, since C leaves the rest indeterminate. A zeroed struct IS the default:
/// `nthreads` 0 is auto, `eff` 0 is `ADM_EFFORT_ESTIMATE`, `debug` 0 is silent.
///   const adm_options o = {.nthreads = 8, .eff = ADM_EFFORT_AUTOMATIC, .debug = 0};
typedef struct {
    size_t nthreads;   ///< 0 auto (size-aware, capped at one per physical core), 1 serial
    adm_effort eff;
    unsigned debug;    ///< `stderr` trace: 0 silent, 1 route, 2 + shape, 3 + cost ranking
} adm_options;

/// Never null; unknown codes give "Unknown error".
ADM_NODISCARD ADM_C_API const char* adm_error_string(adm_status status);

// One-shot transforms: plan, transform, discard. For repeated transforms build a
// plan instead. One-shots route with `ADM_EFFORT_ESTIMATE` and ignore `opts->eff`.
// The calls reject a zero size; the C++ one-shots treat an empty span as a no-op.

/// 1-D, in place. `size` is the number of complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_forward(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward(adm_complex* data, size_t size,
                                               const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse(adm_complex* data, size_t size,
                                               const adm_options* opts);

/// N-D, in place. `shape` lists `ndim` extents and `data` holds their product
/// in complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_forward_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);

// Real transforms (r2c / c2r), N-D, out of place. `shape` gives the REAL
// tensor's extents. A conjugate-symmetric spectrum stores half, so the complex
// tensor has the same extents except the innermost, which becomes
// shape[ndim-1]/2 + 1, the FFTW/ducc0 half-spectrum layout. The complex element
// count is the product of the first ndim-1 extents times (shape[ndim-1]/2 + 1).
// r2c is unscaled; c2r divides by the real element count and overwrites its
// input spectrum.

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

// Plans (repeated transforms). A plan holds the twiddle tables, the route and
// the workers for one shape and runs both directions. Destroy a plan with
// `adm_plan_destroy`. Failed creation still writes a handle carrying the
// failure (`adm_plan_status`, `adm_plan_error_message`), never a hidden null
// for execute.

typedef struct adm_plan_s* adm_plan;

/// 1-D plan over `n` complex elements.
ADM_NODISCARD ADM_C_API adm_status admf_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);

/// N-D plan over shape[ndim]. ndim == 1 is the same plan `adm_plan_1d` builds.
ADM_NODISCARD ADM_C_API adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                               const adm_options* opts);

/// Transform `data` in place. The precision must match the plan's; otherwise
/// the call returns `ADM_ERROR_INVALID_PLAN`.
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data);
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data);

/// Total complex element count the plan expects. 0 for a null plan or a failed
/// creation.
ADM_NODISCARD ADM_C_API size_t adm_plan_size(adm_plan plan);

/// Status of the plan: `ADM_SUCCESS` when usable. A failed creation still
/// writes a handle that carries the failure instead of `NULL`. Re-read the
/// handle's status here, read the reason with `adm_plan_error_message()`, destroy
/// as usual. The execute calls on a failed plan re-return the creation failure.
ADM_NODISCARD ADM_C_API adm_status adm_plan_status(adm_plan plan);

/// Reason for `adm_plan_status()` != `ADM_SUCCESS` (the rejection reason or the
/// caught exception's message); "" for a healthy plan; "null plan" for `NULL`.
/// Valid while the plan lives.
ADM_NODISCARD ADM_C_API const char* adm_plan_error_message(adm_plan plan);

/// Safe on a null plan.
ADM_C_API void adm_plan_destroy(adm_plan plan);

#ifdef __cplusplus
}
#endif
