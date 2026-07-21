#pragma once

// FFTW3-compatible surface over admiral::plan<T> (core complex-DFT plan API).
//
// Point at this header and link admiral::fftw to build FFTW code unchanged. Covers
// the new-array plan calls (fftw_plan_dft{,_1d,_2d,_3d}, execute/execute_dft,
// destroy, malloc/free/alloc_*, cleanup) plus the fftwf_ single-precision mirror.
// Both directions are unnormalized, matching FFTW. `flags` are ignored: planning
// never touches the arrays (always FFTW_ESTIMATE | FFTW_PRESERVE_INPUT).
// Out of scope: real (r2c/c2r) — use admiral::plan_r2c<T> — guru/split/wisdom.

#include <stddef.h>

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

// Interleaved complex, layout-compatible with std::complex<T> and FFTW.
typedef double fftw_complex[2];
typedef float  fftwf_complex[2];

typedef struct fftw_plan_s* fftw_plan;
typedef struct fftw_plan_s* fftwf_plan;

#define FFTW_FORWARD  (-1)
#define FFTW_BACKWARD (+1)

// Planner flags, accepted and ignored (see header note).
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

// ---- single precision -------------------------------------------------------
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
