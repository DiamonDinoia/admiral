/* Installed C surface, compiled as C. This is also the test for the README's
 * "project(app C CXX)" note: the archive is C++ behind a C API, so if the export set
 * ever stops carrying the C++ runtime this link fails. */
#include <admiral/admiral.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024
#define Q 3

int main(void) {
    adm_complex* x = malloc(N * sizeof *x);
    if (!x) return 1;
    for (size_t i = 0; i < N; ++i) {
        const double phase = 2.0 * M_PI * (double)Q * (double)i / (double)N;
        x[i].real = cos(phase);
        x[i].imag = sin(phase);
    }

    adm_plan p;
    adm_status s = adm_plan_1d(&p, N, NULL);
    if (s != ADM_SUCCESS) { printf("plan: %s\n", adm_error_string(s)); return 1; }
    if (adm_plan_size(p) != N) { puts("plan_size wrong"); return 1; }

    s = adm_plan_execute_forward(p, x);
    if (s != ADM_SUCCESS) { printf("forward: %s\n", adm_error_string(s)); return 1; }

    int rc = 0;
    double worst = 0.0;
    for (size_t k = 0; k < N; ++k) {
        /* Forward is exp(-2*pi*i*k*n/N), so the +Q tone peaks at bin Q. */
        const double want = (k == Q) ? (double)N : 0.0;
        const double e = hypot(x[k].real - want, x[k].imag);
        if (e > worst) worst = e;
    }

    /* Inverse divides by N, so the round trip returns the input spike. */
    s = adm_plan_execute_inverse(p, x);
    if (s != ADM_SUCCESS) { printf("inverse: %s\n", adm_error_string(s)); return 1; }
    for (size_t i = 0; i < N; ++i) {
        const double phase = 2.0 * M_PI * (double)Q * (double)i / (double)N;
        const double e = hypot(x[i].real - cos(phase), x[i].imag - sin(phase));
        if (e > worst) worst = e;
    }
    adm_plan_destroy(p);

    /* max|got-ref| <= 32*eps*||ref||_inf, and the spectrum peak is N, so the flat
     * 32*eps the C++ suite asserts in the relative L2 norm. */
    const double tol = (double)N * 32.0 * DBL_EPSILON;
    printf("c worst=%.3g tol=%.3g\n", worst, tol);
    if (worst > tol) rc = 1;

    /* Errors are statuses, not aborts. */
    if (adm_plan_1d(&p, 0, NULL) == ADM_SUCCESS) { puts("zero size accepted"); rc = 1; }
    if (adm_forward(NULL, N, NULL) != ADM_ERROR_NULL_POINTER) { puts("null not rejected"); rc = 1; }

    free(x);
    puts(rc ? "FAIL" : "ok");
    return rc;
}
