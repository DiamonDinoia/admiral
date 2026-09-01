
#include "admiral/fftw3.h"
#include "admiral/admiral.hpp"
#include "admiral/detail/scratch.hpp"

#include <complex>
#include <cstdlib>
#if defined(_MSC_VER)
#include <malloc.h>
#endif
#include <memory>
#include <new>
#include <utility>
#include <vector>
#include "admiral/detail/cxx_compat.hpp"

template<typename T>
struct fftw_handle {
    using value_type = T;

    template<typename... Args>
    fftw_handle(int s, void* i, void* o, Args&&... args)
        : plan(std::forward<Args>(args)...), sign(s), in(i), out(o) {}

    admiral::plan<T> plan;
    int   sign;
    void* in;
    void* out;
};

struct fftw_plan_s  : fftw_handle<double> { using fftw_handle<double>::fftw_handle; };
struct fftwf_plan_s : fftw_handle<float>  { using fftw_handle<float>::fftw_handle; };

namespace {

constexpr size_t kAlign = admiral::detail::span_align<double>;
static_assert(kAlign == admiral::detail::span_align<float>);

void* aligned_malloc(size_t n) {
    if (n == 0) n = 1;
#if defined(_MSC_VER)
    return _aligned_malloc(n, kAlign);
#else
    void* p = nullptr;
    if (posix_memalign(&p, kAlign, n) != 0) return nullptr;
    return p;
#endif
}

// MSVC pairs `_aligned_malloc` with `_aligned_free`; handing that block to `free` is undefined.
void aligned_free(void* p) noexcept {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    free(p);
#endif
}

template<typename H>
void run(H* p, void* in, void* out) {
    using T = typename H::value_type;
    if (in == nullptr || out == nullptr) return;
    auto* ci = static_cast<const std::complex<T>*>(in);
    auto* co = static_cast<std::complex<T>*>(out);
    const bool inplace = (in == out);
    if (p->sign == FFTW_FORWARD) {
        if (inplace) p->plan.forward(co, T(1));
        else         p->plan.forward(ci, co, T(1));
    } else {
        if (inplace) p->plan.inverse(co, T(1));
        else         p->plan.inverse(ci, co, T(1));
    }
}

template<typename H>
H* make_plan(int rank, const int* n, void* in, void* out, int sign, unsigned flags) {
    if (rank < 1 || n == nullptr) return nullptr;
    std::vector<size_t> shape(static_cast<size_t>(rank));
    for (size_t i = 0; i < shape.size(); ++i) {
        if (n[i] < 1) return nullptr;
        shape[i] = static_cast<size_t>(n[i]);
    }
    const admiral::effort eff =
        (flags & FFTW_ESTIMATE) ? admiral::effort::estimate : admiral::effort::automatic;
    try {
        return std::make_unique<H>(sign, in, out, admiral::span<const size_t>(shape),
                                   admiral::options{1, eff})
            .release();
    } catch (...) {
        return nullptr;
    }
}

}

extern "C" {

#define ADM_FFTW_SHIM(PREFIX, T, COMPLEX_T) \
    PREFIX##_plan PREFIX##_plan_dft(int rank, const int* n, COMPLEX_T* in, COMPLEX_T* out, \
                                    int sign, unsigned flags) { \
        return make_plan<PREFIX##_plan_s>(rank, n, in, out, sign, flags); \
    } \
    PREFIX##_plan PREFIX##_plan_dft_1d(int n0, COMPLEX_T* in, COMPLEX_T* out, int sign, \
                                       unsigned flags) { \
        const int n[1] = {n0}; \
        return PREFIX##_plan_dft(1, n, in, out, sign, flags); \
    } \
    PREFIX##_plan PREFIX##_plan_dft_2d(int n0, int n1, COMPLEX_T* in, COMPLEX_T* out, \
                                       int sign, unsigned flags) { \
        const int n[2] = {n0, n1}; \
        return PREFIX##_plan_dft(2, n, in, out, sign, flags); \
    } \
    PREFIX##_plan PREFIX##_plan_dft_3d(int n0, int n1, int n2, COMPLEX_T* in, COMPLEX_T* out, \
                                       int sign, unsigned flags) { \
        const int n[3] = {n0, n1, n2}; \
        return PREFIX##_plan_dft(3, n, in, out, sign, flags); \
    } \
    void PREFIX##_execute(const PREFIX##_plan p) { \
        if (p != nullptr) run(p, p->in, p->out); \
    } \
    void PREFIX##_execute_dft(const PREFIX##_plan p, COMPLEX_T* in, COMPLEX_T* out) { \
        if (p != nullptr) run(p, in, out); \
    } \
    void     PREFIX##_destroy_plan(PREFIX##_plan p) { delete p; } \
    void*    PREFIX##_malloc(size_t n)                 { return aligned_malloc(n); } \
    void     PREFIX##_free(void* p)                    { aligned_free(p); } \
    T*       PREFIX##_alloc_real(size_t n)             { return static_cast<T*>(aligned_malloc(n * sizeof(T))); } \
    COMPLEX_T* PREFIX##_alloc_complex(size_t n)        { return static_cast<COMPLEX_T*>(aligned_malloc(n * sizeof(COMPLEX_T))); } \
    void     PREFIX##_cleanup(void)                    {}

ADM_FFTW_SHIM(fftw,  double, fftw_complex)
ADM_FFTW_SHIM(fftwf, float,  fftwf_complex)

}
