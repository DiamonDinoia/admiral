/* Plain FFTW program that mixes plans from two libraries under LD_PRELOAD.
 * The r2c and guru planners are not in the shim, so real FFTW makes those plans.
 * The c2c planner is in the shim. Every plan runs through fftw_execute or fftw_execute_dft
 * and goes to fftw_destroy_plan. The program checks that no heap is left.
 * With an argument, it does not destroy one c2c plan, and the heap check must fail.
 * Each array prints as "<tag> <index> <re> <im>". */
#include <fftw3.h>

#include <malloc.h>
#include <math.h>
#include <stdio.h>

#define N 24

static int leak_one; /* main sets it: skip one destroy_plan */

static double sample(long i, int part) {
    return sin(0.41 * (double)i + 0.9 * (double)part) + 0.5 * cos(0.07 * (double)(i * i));
}

#define DEFINE_MIXED(NAME, TYPE, PFX, FMT)                                                    \
    static int NAME(const char* tag) {                                                        \
        PFX##_complex* a = PFX##_alloc_complex(N);                                            \
        PFX##_complex* b = PFX##_alloc_complex(N);                                            \
        PFX##_complex* c = PFX##_alloc_complex(N);                                            \
        PFX##_complex* d = PFX##_alloc_complex(N);                                            \
        PFX##_complex* e = PFX##_alloc_complex(N);                                            \
        TYPE* r = PFX##_alloc_real(N);                                                        \
        PFX##_complex* rc = PFX##_alloc_complex(N / 2 + 1);                                   \
        if (!a || !b || !c || !d || !e || !r || !rc) return 1;                                \
        const PFX##_iodim dim = {N, 1, 1};                                                    \
        PFX##_plan pr = PFX##_plan_dft_r2c_1d(N, r, rc, FFTW_ESTIMATE);                       \
        PFX##_plan pg = PFX##_plan_guru_dft(1, &dim, 0, NULL, a, b, FFTW_FORWARD, FFTW_ESTIMATE); \
        PFX##_plan pc = PFX##_plan_dft_1d(N, a, c, FFTW_FORWARD, FFTW_ESTIMATE);              \
        if (!pr || !pg || !pc) return 1;                                                      \
        for (int i = 0; i < N; ++i) {                                                         \
            r[i] = (TYPE)sample(i, 2);                                                        \
            a[i][0] = (TYPE)sample(i, 0);                                                     \
            a[i][1] = (TYPE)sample(i, 1);                                                     \
            d[i][0] = (TYPE)(0.5 * sample(i, 0));                                             \
            d[i][1] = (TYPE)(0.5 * sample(i, 1));                                             \
        }                                                                                     \
        PFX##_execute(pr);                                                                    \
        PFX##_execute(pg);                                                                    \
        PFX##_execute(pc);                                                                    \
        PFX##_execute_dft(pg, d, e);                                                          \
        for (int i = 0; i < N / 2 + 1; ++i)                                                   \
            printf("%sr %d " FMT " " FMT "\n", tag, i, (double)rc[i][0], (double)rc[i][1]);   \
        for (int i = 0; i < N; ++i)                                                           \
            printf("%sg %d " FMT " " FMT "\n", tag, i, (double)b[i][0], (double)b[i][1]);     \
        for (int i = 0; i < N; ++i)                                                           \
            printf("%sc %d " FMT " " FMT "\n", tag, i, (double)c[i][0], (double)c[i][1]);     \
        for (int i = 0; i < N; ++i)                                                           \
            printf("%sx %d " FMT " " FMT "\n", tag, i, (double)e[i][0], (double)e[i][1]);     \
        PFX##_destroy_plan(pr);                                                               \
        PFX##_destroy_plan(pg);                                                               \
        if (leak_one) leak_one = 0;                                                           \
        else PFX##_destroy_plan(pc);                                                          \
        PFX##_free(a);                                                                        \
        PFX##_free(b);                                                                        \
        PFX##_free(c);                                                                        \
        PFX##_free(d);                                                                        \
        PFX##_free(e);                                                                        \
        PFX##_free(r);                                                                        \
        PFX##_free(rc);                                                                       \
        return 0;                                                                             \
    }

DEFINE_MIXED(mixed_d, double, fftw, "%.17g")
DEFINE_MIXED(mixed_f, float, fftwf, "%.9g")

/* Heap in use. After every plan is destroyed and both cleanup calls ran, the heap in use must
 * return to its value at the start. A plan that nobody destroyed, or a cleanup that did not
 * reach real FFTW, leaves bytes behind. Real FFTW keeps its planner tables reachable, so a leak
 * checker misses the second case. The driver sets glibc.malloc.tcache_count=0, because the
 * tcache counts freed chunks as in use. */
static size_t heap_in_use(void) {
    const struct mallinfo2 mi = mallinfo2();
    return mi.uordblks + mi.hblkhd;
}

int main(int argc, char** argv) {
    static char buffer[1 << 16]; /* stdout must not allocate after the baseline */
    setvbuf(stdout, buffer, _IOFBF, sizeof buffer);
    fftw_free(fftw_malloc(1)); /* glibc keeps memory from its first allocation in use */
    leak_one = argc > 1;
    const size_t before = heap_in_use();
    int rc = 0;
    rc |= mixed_d("d");
    rc |= mixed_f("f");
    fftw_cleanup();
    fftwf_cleanup();
    const size_t after = heap_in_use();
    if (after != before) {
        fprintf(stderr, "mixed: heap in use went from %zu to %zu bytes\n", before, after);
        rc = 1;
    }
    return rc;
}
