/* Plain FFTW program. It includes the real <fftw3.h> and knows nothing about admiral.
 * It uses the c2c subset only: 1-D, 2-D, 3-D and generic rank, both precisions, both
 * directions. It calls every function in the shim's export list.
 * Each array prints as "<tag> <index> <re> <im>" with full precision. */
#include <fftw3.h>

#include <math.h>
#include <stdio.h>

#define DIMS_1 60
#define DIMS_2 12, 10
#define DIMS_3 6, 8, 10

static double sample(long i, int part) {
    return sin(0.37 * (double)i + 1.1 * (double)part) + 0.25 * cos(0.011 * (double)(i * i));
}

/* One planner call per rank, so the binding check sees plan_dft_1d. */
#define PLAN_1(PFX, I, O, SIGN) PFX##_plan_dft_1d(n[0], I, O, SIGN, FFTW_ESTIMATE)
#define PLAN_2(PFX, I, O, SIGN) PFX##_plan_dft_2d(n[0], n[1], I, O, SIGN, FFTW_ESTIMATE)
#define PLAN_3(PFX, I, O, SIGN) PFX##_plan_dft_3d(n[0], n[1], n[2], I, O, SIGN, FFTW_ESTIMATE)
#define PLAN_G(PFX, I, O, SIGN) \
    PFX##_plan_dft((int)(sizeof n / sizeof n[0]), n, I, O, SIGN, FFTW_ESTIMATE)

/* PL picks the planner call, RANK sets the length of n. The forward input comes from
 * alloc_complex, the middle array from malloc, and a second input from alloc_real.
 * execute_dft runs the forward plan on that second input, which holds half the first. */
#define DEFINE_CASE(NAME, TYPE, PFX, FMT, PL, RANK, ...)                                      \
    static int NAME(const char* tag) {                                                        \
        const int n[RANK] = {__VA_ARGS__};                                                    \
        int total = 1;                                                                        \
        for (int d = 0; d < (RANK); ++d) total *= n[d];                                       \
        PFX##_complex* in = PFX##_alloc_complex((size_t)total);                               \
        PFX##_complex* mid = PFX##_malloc((size_t)total * sizeof(PFX##_complex));             \
        PFX##_complex* back = PFX##_alloc_complex((size_t)total);                             \
        PFX##_complex* in2 = (PFX##_complex*)PFX##_alloc_real(2 * (size_t)total);             \
        PFX##_complex* mid2 = PFX##_alloc_complex((size_t)total);                             \
        if (!in || !mid || !back || !in2 || !mid2) return 1;                                  \
        PFX##_plan fwd = PLAN_##PL(PFX, in, mid, FFTW_FORWARD);                               \
        PFX##_plan bwd = PLAN_##PL(PFX, mid, back, FFTW_BACKWARD);                            \
        if (!fwd || !bwd) return 1;                                                           \
        for (int i = 0; i < total; ++i) {                                                     \
            in[i][0] = (TYPE)sample(i, 0);                                                    \
            in[i][1] = (TYPE)sample(i, 1);                                                    \
            in2[i][0] = (TYPE)(0.5 * sample(i, 0));                                           \
            in2[i][1] = (TYPE)(0.5 * sample(i, 1));                                           \
        }                                                                                     \
        PFX##_execute(fwd);                                                                   \
        PFX##_execute(bwd);                                                                   \
        PFX##_execute_dft(fwd, in2, mid2);                                                    \
        for (int i = 0; i < total; ++i)                                                       \
            printf("%sf %d " FMT " " FMT "\n", tag, i, (double)mid[i][0], (double)mid[i][1]); \
        for (int i = 0; i < total; ++i)                                                       \
            printf("%sb %d " FMT " " FMT "\n", tag, i, (double)back[i][0], (double)back[i][1]); \
        for (int i = 0; i < total; ++i)                                                       \
            printf("%sx %d " FMT " " FMT "\n", tag, i, (double)mid2[i][0], (double)mid2[i][1]); \
        PFX##_destroy_plan(fwd);                                                              \
        PFX##_destroy_plan(bwd);                                                              \
        PFX##_free(in);                                                                       \
        PFX##_free(mid);                                                                      \
        PFX##_free(back);                                                                     \
        PFX##_free(in2);                                                                      \
        PFX##_free(mid2);                                                                     \
        return 0;                                                                             \
    }

DEFINE_CASE(case_d1, double, fftw, "%.17g", 1, 1, DIMS_1)
DEFINE_CASE(case_d2, double, fftw, "%.17g", 2, 2, DIMS_2)
DEFINE_CASE(case_d3, double, fftw, "%.17g", 3, 3, DIMS_3)
DEFINE_CASE(case_dg, double, fftw, "%.17g", G, 2, 9, 14)
DEFINE_CASE(case_f1, float, fftwf, "%.9g", 1, 1, DIMS_1)
DEFINE_CASE(case_f2, float, fftwf, "%.9g", 2, 2, DIMS_2)
DEFINE_CASE(case_f3, float, fftwf, "%.9g", 3, 3, DIMS_3)
DEFINE_CASE(case_fg, float, fftwf, "%.9g", G, 2, 9, 14)

int main(void) {
    int rc = 0;
    rc |= case_d1("d1");
    rc |= case_d2("d2");
    rc |= case_d3("d3");
    rc |= case_dg("dg");
    rc |= case_f1("f1");
    rc |= case_f2("f2");
    rc |= case_f3("f3");
    rc |= case_fg("fg");
    fftw_cleanup();
    fftwf_cleanup();
    return rc;
}
