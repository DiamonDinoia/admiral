/* Installed FFTW shim, compiled as C: FFTW's names, FFTW's convention (both
 * directions unscaled), so the round trip comes back scaled by N. */
#include <admiral/fftw3.h>

#include <float.h>
#include <math.h>
#include <stdio.h>

#define N 512
#define Q 5

int main(void) {
    fftw_complex* in = fftw_alloc_complex(N);
    fftw_complex* out = fftw_alloc_complex(N);
    if (!in || !out) return 1;
    for (int i = 0; i < N; ++i) {
        const double phase = 2.0 * M_PI * (double)Q * (double)i / (double)N;
        in[i][0] = cos(phase);
        in[i][1] = sin(phase);
    }

    fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    if (!p) { puts("plan_dft_1d returned NULL"); return 1; }
    fftw_execute(p);

    int rc = 0;
    double worst = 0.0;
    for (int k = 0; k < N; ++k) {
        /* `FFTW_FORWARD` is exp(-2*pi*i*k*n/N), so the +Q tone peaks at bin Q. */
        const double want = (k == Q) ? (double)N : 0.0;
        const double e = hypot(out[k][0] - want, out[k][1]);
        if (e > worst) worst = e;
    }

    /* Unscaled inverse: the round trip is N times the input, unlike the adm_ API. */
    fftw_plan q = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(q);
    for (int i = 0; i < N; ++i) {
        const double phase = 2.0 * M_PI * (double)Q * (double)i / (double)N;
        const double e = hypot(in[i][0] - (double)N * cos(phase), in[i][1] - (double)N * sin(phase));
        if (e > worst) worst = e;
    }

    fftw_destroy_plan(p);
    fftw_destroy_plan(q);
    fftw_free(in);
    fftw_free(out);
    fftw_cleanup();

    /* Both directions unscaled, so the round trip's reference has magnitude N and the
     * spectrum peak is N too: one factor of N. max|got-ref| <= 32*eps*N is the
     * componentwise form of the suite-wide 32*eps. */
    const double tol = (double)N * 32.0 * DBL_EPSILON;
    printf("fftw worst=%.3g tol=%.3g\n", worst, tol);
    if (worst > tol) rc = 1;
    puts(rc ? "FAIL" : "ok");
    return rc;
}
