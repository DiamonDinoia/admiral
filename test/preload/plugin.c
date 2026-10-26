/* Plugin that links FFTW, as a Python extension or a MEX file does. run.sh links it to real FFTW
 * and to standin.c. It loads it through Python ctypes with dlopen(RTLD_LOCAL), so its FFTW stays
 * out of the global scope. Under LD_PRELOAD the r2c planners bind to that FFTW, and execute and
 * destroy_plan bind to the shim. run_plugin returns 0 when the DC term of each precision is right. */
#include <fftw3.h>

#define DEFINE_R2C(NAME, TYPE, PFX)                                                           \
    static int NAME(void) {                                                                   \
        TYPE* r = PFX##_alloc_real(8);                                                        \
        PFX##_complex* c = PFX##_alloc_complex(5);                                            \
        if (!r || !c) return 1;                                                               \
        PFX##_plan p = PFX##_plan_dft_r2c_1d(8, r, c, FFTW_ESTIMATE);                         \
        if (!p) return 1;                                                                     \
        for (int i = 0; i < 8; ++i) r[i] = (TYPE)i;                                           \
        PFX##_execute(p);                                                                     \
        const int bad = c[0][0] != 28 || c[0][1] != 0;                                        \
        PFX##_destroy_plan(p);                                                                \
        PFX##_free(r);                                                                        \
        PFX##_free(c);                                                                        \
        return bad;                                                                           \
    }

DEFINE_R2C(r2c_d, double, fftw)
DEFINE_R2C(r2c_f, float, fftwf)

int run_plugin(void) { return r2c_d() | r2c_f(); }
