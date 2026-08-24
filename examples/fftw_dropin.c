/* The FFTW shim: existing FFTW code compiles unchanged against <admiral/fftw3.h>. */
#include <admiral/fftw3.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = 1024;
    fftw_complex* in = fftw_alloc_complex(N);
    fftw_complex* out = fftw_alloc_complex(N);
    fftw_plan p = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int i = 0; i < N; ++i) in[i][0] = sin(0.01 * (double)i), in[i][1] = 0.0;

    fftw_plan q = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(q);      /* complex in -> out */
    fftw_execute(p);      /* out -> in, divided by N */
    fftw_destroy_plan(q);
    fftw_destroy_plan(p);

    /* FFTW convention: both directions are unscaled, so a round trip multiplies
       the input by N. Scale back before comparing. */
    double err = 0;
    for (int i = 0; i < N; ++i) {
        const double d = fabs(in[i][0] / N - sin(0.01 * (double)i)) + fabs(in[i][1]);
        if (d > err) err = d;
    }
    fftw_free(in);
    fftw_free(out);
    fftw_cleanup();
    if (err > 1e-10) {
        fprintf(stderr, "FFTW round trip failed: err=%g\n", err);
        return 1;
    }
    return 0;
}
