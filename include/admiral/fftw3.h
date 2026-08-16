#pragma once

// ============================================================================
// FFTW3-compatible header for Admiral.
//
// Include this instead of <fftw3.h> and link admiral::fftw. Existing FFTW code
// that stays inside the surface below compiles and runs unchanged — same
// spellings, same sign convention, same row-major layout, same unscaled result
// in both directions.
//
//   #include <admiral/fftw3.h>
//
//   fftw_complex* in  = fftw_alloc_complex(N);
//   fftw_complex* out = fftw_alloc_complex(N);
//   fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
//   fftw_execute(p);
//   fftw_destroy_plan(p);
//   fftw_free(in); fftw_free(out);
//
// What is covered
//   fftw_plan_dft, fftw_plan_dft_1d/2d/3d
//   fftw_execute, fftw_execute_dft, fftw_destroy_plan
//   fftw_malloc, fftw_free, fftw_alloc_real, fftw_alloc_complex, fftw_cleanup
//   the fftwf_ single-precision mirror of all of the above
//
// What is not
//   Real transforms (r2c, c2r, r2r) — use admiral::plan_r2c<T> from
//   <admiral/admiral.hpp>. The guru, advanced and split interfaces. Wisdom.
//   fftw_plan_with_nthreads — pass a thread count to admiral::plan instead.
//   Nothing here silently degrades: an impossible call does not compile.
//
// Where the behaviour differs from FFTW
//   The planning flags are honoured, the rest are accepted and ignored.
//   FFTW_ESTIMATE picks from the fitted cost model with no search; every other
//   flag combination (the default FFTW_MEASURE included) races the model's
//   candidates. The engine has one search budget, so FFTW_PATIENT and
//   FFTW_EXHAUSTIVE buy nothing extra. Planning never reads or writes the
//   arrays, so a measuring flag cannot clobber data already filled in, and
//   every plan behaves as FFTW_PRESERVE_INPUT.
//
//   fftw_execute() replays the pointers and direction the plan was created
//   with, as in FFTW. fftw_execute_dft() runs the same plan on other arrays;
//   their alignment and in/out-of-place character need not match the plan.
//   FFTW_WISDOM_ONLY is ignored; the call still returns a live plan. An
//   out-of-range `sign` computes a backward transform, the same as FFTW.
//   Rank-0 plans are rejected (NULL); FFTW defines rank 0 as a scalar copy.
//
//   Plans are single-threaded here. Calling one plan from two threads at once is
//   not supported; separate plans are independent.
//
//   in == out is in place. Arrays that partially overlap are undefined
//   behaviour, as in FFTW.
// ============================================================================

#include <stddef.h>

// The build hides every symbol by default, so these opt back in explicitly.
#if defined(_WIN32) || defined(__CYGWIN__)
#  define FFTW_C_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#  define FFTW_C_API __attribute__((visibility("default")))
#else
#  define FFTW_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interleaved [real, imag], as in FFTW and layout-compatible with std::complex.
typedef double fftw_complex[2];
typedef float  fftwf_complex[2];

// Distinct incomplete types, as in FFTW, so passing a float plan to the double
// API is a compile error instead of a silent failure at execute time.
typedef struct fftw_plan_s*  fftw_plan;
typedef struct fftwf_plan_s* fftwf_plan;

// Sign of the exponent: forward is exp(-2*pi*i*k*n/N).
#define FFTW_FORWARD  (-1)
#define FFTW_BACKWARD (+1)

// ESTIMATE / MEASURE / PATIENT / EXHAUSTIVE pick the planning effort; the rest are
// accepted and ignored. See the header note.
#define FFTW_MEASURE         (0U)
#define FFTW_DESTROY_INPUT   (1U << 0)
#define FFTW_UNALIGNED       (1U << 1)
#define FFTW_CONSERVE_MEMORY (1U << 2)
#define FFTW_EXHAUSTIVE      (1U << 3)
#define FFTW_PRESERVE_INPUT  (1U << 4)
#define FFTW_PATIENT         (1U << 5)
#define FFTW_ESTIMATE        (1U << 6)
#define FFTW_WISDOM_ONLY     (1U << 21)

// ---- double precision -------------------------------------------------------
// A plan call returns NULL on a bad rank, a non-positive extent, or allocation
// failure. Executing or destroying NULL is safe and does nothing.
// The alloc_* helpers return 64-byte aligned memory; free it with fftw_free.
// fftw_cleanup() is a no-op: there is no wisdom cache or other global state, so
// everything a plan owns is released by fftw_destroy_plan.
FFTW_C_API fftw_plan fftw_plan_dft_1d(int n0, fftw_complex* in, fftw_complex* out,
                                      int sign, unsigned flags);
FFTW_C_API fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out,
                                      int sign, unsigned flags);
FFTW_C_API fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex* in,
                                      fftw_complex* out, int sign, unsigned flags);
FFTW_C_API fftw_plan fftw_plan_dft(int rank, const int* n, fftw_complex* in,
                                   fftw_complex* out, int sign, unsigned flags);

FFTW_C_API void fftw_execute(const fftw_plan p);
FFTW_C_API void fftw_execute_dft(const fftw_plan p, fftw_complex* in, fftw_complex* out);
FFTW_C_API void fftw_destroy_plan(fftw_plan p);

FFTW_C_API void*         fftw_malloc(size_t n);
FFTW_C_API void          fftw_free(void* p);
FFTW_C_API double*       fftw_alloc_real(size_t n);
FFTW_C_API fftw_complex* fftw_alloc_complex(size_t n);
FFTW_C_API void          fftw_cleanup(void);

// ---- single precision (same contract as above) -------------------------------
FFTW_C_API fftwf_plan fftwf_plan_dft_1d(int n0, fftwf_complex* in, fftwf_complex* out,
                                        int sign, unsigned flags);
FFTW_C_API fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex* in, fftwf_complex* out,
                                        int sign, unsigned flags);
FFTW_C_API fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex* in,
                                        fftwf_complex* out, int sign, unsigned flags);
FFTW_C_API fftwf_plan fftwf_plan_dft(int rank, const int* n, fftwf_complex* in,
                                     fftwf_complex* out, int sign, unsigned flags);

FFTW_C_API void          fftwf_execute(const fftwf_plan p);
FFTW_C_API void          fftwf_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out);
FFTW_C_API void          fftwf_destroy_plan(fftwf_plan p);

FFTW_C_API void*          fftwf_malloc(size_t n);
FFTW_C_API void           fftwf_free(void* p);
FFTW_C_API float*         fftwf_alloc_real(size_t n);
FFTW_C_API fftwf_complex* fftwf_alloc_complex(size_t n);
FFTW_C_API void           fftwf_cleanup(void);

#ifdef __cplusplus
}
#endif
