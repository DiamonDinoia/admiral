#pragma once

#include <stddef.h>

#include <admiral/detail/api.h>

#define ADM_C_API ADM_VISIBILITY

#if defined(__cplusplus) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L)
#  define ADM_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#  define ADM_NODISCARD __attribute__((warn_unused_result))
#else
#  define ADM_NODISCARD
#endif

// Executing one `adm_plan` from several threads is safe: an execute holds no shared mutable
// state. Pool dispatches serialize, so a shared plan adds no throughput.
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { float real; float imag; } admf_complex;
typedef struct { double real; double imag; } adm_complex;

typedef enum {
    ADM_SUCCESS = 0,
    ADM_ERROR_NULL_POINTER = -1,
    ADM_ERROR_INVALID_SIZE = -2,
    ADM_ERROR_OUT_OF_MEMORY = -3,
    ADM_ERROR_INVALID_PLAN = -4,
    ADM_ERROR_INVALID_OPTION = -5,
    ADM_ERROR_INTERNAL = -6
} adm_status;

typedef enum {
    ADM_EFFORT_ESTIMATE = 0,
    ADM_EFFORT_AUTOMATIC = 1,
    ADM_EFFORT_MEASURE = 2
} adm_effort;

// `nthreads` 0 auto-selects the width from the transform size, 1 forces serial, n forces n.
// `eff` ADM_EFFORT_ESTIMATE routes from the cost model; the other two also time candidates.
typedef struct {
    size_t nthreads;
    adm_effort eff;
    unsigned debug;
} adm_options;

// Never returns NULL; an unrecognised `status` yields "Unknown error".
ADM_NODISCARD ADM_C_API const char* adm_error_string(adm_status status);

ADM_NODISCARD ADM_C_API adm_status admf_forward(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward(adm_complex* data, size_t size,
                                               const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse(admf_complex* data, size_t size,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse(adm_complex* data, size_t size,
                                               const adm_options* opts);

ADM_NODISCARD ADM_C_API adm_status admf_forward_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_forward_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status admf_inverse_nd(admf_complex* data, const size_t* shape,
                                                   size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_inverse_nd(adm_complex* data, const size_t* shape,
                                                  size_t ndim, const adm_options* opts);

ADM_NODISCARD ADM_C_API adm_status admf_r2c_nd(const float* in, admf_complex* out,
                                               const size_t* shape, size_t ndim,
                                               const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_r2c_nd(const double* in, adm_complex* out,
                                              const size_t* shape, size_t ndim,
                                              const adm_options* opts);
// At `ndim` >= 2 these overwrite `spec` while running. Copy `spec` first to keep it.
ADM_NODISCARD ADM_C_API adm_status admf_c2r_nd(admf_complex* spec, float* out, const size_t* shape,
                                               size_t ndim, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_c2r_nd(adm_complex* spec, double* out, const size_t* shape,
                                              size_t ndim, const adm_options* opts);

typedef struct adm_plan_s* adm_plan;

// A failed create still writes a handle instead of NULL. Read the reason with
// `adm_plan_status`, then release the handle with `adm_plan_destroy` as after a success.
ADM_NODISCARD ADM_C_API adm_status admf_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_1d(adm_plan* plan, size_t n, const adm_options* opts);

ADM_NODISCARD ADM_C_API adm_status admf_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                                const adm_options* opts);
ADM_NODISCARD ADM_C_API adm_status adm_plan_nd(adm_plan* plan, const size_t* shape, size_t ndim,
                                               const adm_options* opts);

// `data` holds exactly `adm_plan_size(plan)` complex elements. Executing a plan that failed
// creation re-returns that plan's own status, not ADM_ERROR_INVALID_PLAN.
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_forward(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_forward(adm_plan plan, adm_complex* data);
ADM_NODISCARD ADM_C_API adm_status admf_plan_execute_inverse(adm_plan plan, admf_complex* data);
ADM_NODISCARD ADM_C_API adm_status adm_plan_execute_inverse(adm_plan plan, adm_complex* data);

// Returns 0 for NULL and for a plan that failed creation.
ADM_NODISCARD ADM_C_API size_t adm_plan_size(adm_plan plan);

ADM_NODISCARD ADM_C_API adm_status adm_plan_status(adm_plan plan);

ADM_NODISCARD ADM_C_API const char* adm_plan_error_message(adm_plan plan);

// Accepts NULL.
ADM_C_API void adm_plan_destroy(adm_plan plan);

#ifdef __cplusplus
}
#endif
