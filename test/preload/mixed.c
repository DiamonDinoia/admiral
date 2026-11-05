/* Plain FFTW program that mixes plans from two libraries under LD_PRELOAD.
 * The r2c and guru planners are not in the shim, so real FFTW makes those plans.
 * The c2c planner is in the shim. Every plan runs through fftw_execute or fftw_execute_dft
 * and goes to fftw_destroy_plan. The program checks that no heap is left.
 * With "leak", it does not destroy one c2c plan, and the heap check must fail.
 * With "control", it makes no c2c plan and prints its heap delta. The driver passes that delta
 * as the baseline of a run: a platform allocation that hits both runs cancels out, and a leak in
 * the plan path of admiral hits the run only. A plain number argument is that baseline.
 * One warmup run precedes the baseline. The plan path pays one-time init at its first run (the
 * engine allocator sets its thread up and keeps one chunk in use to the thread end). The warmup
 * run pays that init, so the measured run shows only what each run leaves behind. A leak pays on
 * all runs and the check rejects it.
 * Each array prints as "<tag> <index> <re> <im>". The control prints no arrays. */
#include <fftw3.h>

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 24

static int leak_one; /* main sets it: skip one destroy_plan */
static int control;  /* main sets it: no c2c plan, print the heap delta */
static int warmup;   /* main sets it: the warmup run pays one-time init, prints no arrays */

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
        PFX##_plan pc = control ? NULL : PFX##_plan_dft_1d(N, a, c, FFTW_FORWARD, FFTW_ESTIMATE); \
        if (!pr || !pg || (!control && !pc)) return 1;                                        \
        for (int i = 0; i < N; ++i) {                                                         \
            r[i] = (TYPE)sample(i, 2);                                                        \
            a[i][0] = (TYPE)sample(i, 0);                                                     \
            a[i][1] = (TYPE)sample(i, 1);                                                     \
            d[i][0] = (TYPE)(0.5 * sample(i, 0));                                             \
            d[i][1] = (TYPE)(0.5 * sample(i, 1));                                             \
        }                                                                                     \
        PFX##_execute(pr);                                                                    \
        PFX##_execute(pg);                                                                    \
        if (!control) PFX##_execute(pc);                                                      \
        PFX##_execute_dft(pg, d, e);                                                          \
        if (!control && !warmup) {                                                            \
            for (int i = 0; i < N / 2 + 1; ++i)                                               \
                printf("%sr %d " FMT " " FMT "\n", tag, i, (double)rc[i][0], (double)rc[i][1]);   \
            for (int i = 0; i < N; ++i)                                                       \
                printf("%sg %d " FMT " " FMT "\n", tag, i, (double)b[i][0], (double)b[i][1]);     \
            for (int i = 0; i < N; ++i)                                                       \
                printf("%sc %d " FMT " " FMT "\n", tag, i, (double)c[i][0], (double)c[i][1]);     \
            for (int i = 0; i < N; ++i)                                                       \
                printf("%sx %d " FMT " " FMT "\n", tag, i, (double)e[i][0], (double)e[i][1]);     \
        }                                                                                     \
        PFX##_destroy_plan(pr);                                                               \
        PFX##_destroy_plan(pg);                                                               \
        if (!control) {                                                                       \
            if (leak_one && !warmup) leak_one = 0;                                            \
            else PFX##_destroy_plan(pc);                                                      \
        }                                                                                     \
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
 * return to the baseline. The baseline is 0 at default, or the delta of a control run in the
 * same environment: a platform allocation that survives in both runs cancels out then. A plan
 * that nobody destroyed, or a leak in the plan path of admiral, leaves bytes behind only in the
 * run and the check fails. Real FFTW keeps its planner tables reachable, so a leak checker
 * misses the second case. The driver sets glibc.malloc.tcache_count=0, because the
 * tcache counts freed chunks as in use. */
static size_t heap_in_use(void) {
    const struct mallinfo2 mi = mallinfo2();
    return mi.uordblks + mi.hblkhd;
}

int main(int argc, char** argv) {
    static char buffer[1 << 16]; /* stdout must not allocate after the baseline */
    setvbuf(stdout, buffer, _IOFBF, sizeof buffer);
    fftw_free(fftw_malloc(1)); /* glibc keeps memory from its first allocation in use */
    size_t baseline = 0;
    if (argc > 1 && strcmp(argv[1], "leak") == 0) leak_one = 1;
    else if (argc > 1 && strcmp(argv[1], "control") == 0) control = 1;
    else if (argc > 1) baseline = strtoull(argv[1], NULL, 10);
    /* One warmup run before the baseline, in all modes: it pays the one-time init of the paths
     * of the run. */
    warmup = 1;
    mixed_d("d");
    mixed_f("f");
    fftw_cleanup();
    fftwf_cleanup();
    warmup = 0;
    const size_t before = heap_in_use();
    int rc = 0;
    rc |= mixed_d("d");
    rc |= mixed_f("f");
    fftw_cleanup();
    fftwf_cleanup();
    const size_t after = heap_in_use();
    if (control) {
        printf("%zu\n", after - before);
        return rc;
    }
    if (after - before != baseline) {
        fprintf(stderr, "mixed: heap in use went from %zu to %zu bytes\n", before, after);
        rc = 1;
    }
    return rc;
}
