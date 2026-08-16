// FFTW3-compatible shim over admiral::plan<T>. See include/admiral/fftw3.h for the
// covered surface and conventions (unnormalized both directions; flags ignored).

#include "admiral/fftw3.h"
#include "admiral/admiral.hpp"
#include "admiral/detail/scratch.hpp"  // span_align: the alignment the kernels assume

#include <complex>
#include <cstdlib>   // posix_memalign, free
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

// One handle template, instantiated once per precision. The handle owns the admiral
// plan plus the arrays and direction captured at plan time, so fftw_execute(p) can
// replay them. Precision is the handle's own type, so there is no enum to keep in
// sync, no void* to cast back, and no destructor to write.
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

// The two names fftw3.h declares. Separate types are what makes fftw_execute()
// on an fftwf_plan a compile error rather than a silent no-op.
struct fftw_plan_s  : fftw_handle<double> { using fftw_handle<double>::fftw_handle; };
struct fftwf_plan_s : fftw_handle<float>  { using fftw_handle<float>::fftw_handle; };

namespace {

// FFTW guarantees SIMD-friendly alignment; the value is the library's own, not a
// literal big enough for one arch. span_align is arch-derived and
// precision-independent, so one number covers both entry points.
constexpr size_t kAlign = admiral::detail::span_align<double>;
static_assert(kAlign == admiral::detail::span_align<float>);

void* aligned_malloc(size_t n) {
    void* p = nullptr;
    if (n == 0) n = 1;
    if (posix_memalign(&p, kAlign, n) != 0) return nullptr;
    return p;
}

// fct=1: FFTW is unnormalized both ways. in == out runs in place.
template<typename H>
void run(const H* p, void* in, void* out) {
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
    // FFTW_MEASURE is zero, so estimate is the flag to detect: it opts out of the
    // candidate race. Everything else, PATIENT/EXHAUSTIVE included, takes
    // effort::automatic — the engine has one search budget (admiral.hpp).
    const admiral::effort eff =
        (flags & FFTW_ESTIMATE) ? admiral::effort::estimate : admiral::effort::automatic;
    try {
        return std::make_unique<H>(sign, in, out, std::span<const size_t>(shape),
                                   admiral::options{.eff = eff})
            .release();
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

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
    void     PREFIX##_free(void* p)                    { free(p); } \
    T*       PREFIX##_alloc_real(size_t n)             { return static_cast<T*>(aligned_malloc(n * sizeof(T))); } \
    COMPLEX_T* PREFIX##_alloc_complex(size_t n)        { return static_cast<COMPLEX_T*>(aligned_malloc(n * sizeof(COMPLEX_T))); } \
    void     PREFIX##_cleanup(void)                    {}

ADM_FFTW_SHIM(fftw,  double, fftw_complex)
ADM_FFTW_SHIM(fftwf, float,  fftwf_complex)

}  // extern "C"
