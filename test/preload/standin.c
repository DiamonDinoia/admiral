/* Stand-in FFTW. run.sh builds it under a soname that no FFTW uses, as auditwheel renames the
 * FFTW in a Python wheel, and links plugin.c to it. It has the calls that plugin.c makes. The shim
 * has no r2c planner, so the plans come from this library. Its r2c execute computes only the DC
 * term, which is the term that plugin.c checks.
 * With SEND_BACK=1, execute calls the first fftw_execute in the global scope instead. Under
 * LD_PRELOAD that is the shim, so the plan goes back to the shim, as from a second shim copy.
 * With SEND_BACK=2, the first execute makes that call from a new thread and joins it, then runs
 * the plan. The first thread is still inside the shim, so the shim must take the second call.
 * standin_cleanups counts the calls to its fftw_cleanup and fftwf_cleanup. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fftw3.h>
#include <pthread.h>
#include <stdlib.h>

#ifndef SEND_BACK
#define SEND_BACK 0
#endif

static int cleanups;
int standin_cleanups(void) { return cleanups; }

#define DEFINE_STANDIN(TYPE, PFX)                                                             \
    struct PFX##_plan_s {                                                                     \
        TYPE* in; /* a pointer first, as in FFTW: the shim reads a plan's first 8 bytes */    \
        PFX##_complex* out;                                                                   \
        int n;                                                                                \
    };                                                                                        \
    TYPE* PFX##_alloc_real(size_t n) { return malloc(n * sizeof(TYPE)); }                     \
    PFX##_complex* PFX##_alloc_complex(size_t n) { return malloc(n * sizeof(PFX##_complex)); } \
    void PFX##_free(void* p) { free(p); }                                                     \
    PFX##_plan PFX##_plan_dft_r2c_1d(int n, TYPE* in, PFX##_complex* out, unsigned flags) {   \
        PFX##_plan p = malloc(sizeof *p);                                                     \
        if (p) *p = (struct PFX##_plan_s){in, out, n};                                        \
        return p;                                                                             \
    }                                                                                         \
    static void* PFX##_send_back(void* p) {                                                   \
        ((void (*)(PFX##_plan))dlsym(RTLD_DEFAULT, #PFX "_execute"))(p);                      \
        return NULL;                                                                          \
    }                                                                                         \
    void PFX##_execute(const PFX##_plan p) {                                                  \
        static int sent; /* pthread_create orders this write before the new thread reads it */ \
        if (SEND_BACK == 1) {                                                                 \
            PFX##_send_back(p);                                                               \
            return;                                                                           \
        }                                                                                     \
        if (SEND_BACK == 2 && !sent) {                                                        \
            sent = 1;                                                                         \
            pthread_t t;                                                                      \
            if (pthread_create(&t, NULL, PFX##_send_back, p) || pthread_join(t, NULL)) abort(); \
        }                                                                                     \
        TYPE s = 0;                                                                           \
        for (int i = 0; i < p->n; ++i) s += p->in[i];                                         \
        p->out[0][0] = s;                                                                     \
        p->out[0][1] = 0;                                                                     \
    }                                                                                         \
    void PFX##_cleanup(void) { ++cleanups; }                                                  \
    void PFX##_destroy_plan(PFX##_plan p) { free(p); }

DEFINE_STANDIN(double, fftw)
DEFINE_STANDIN(float, fftwf)
