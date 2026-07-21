// FFTW3-compatible shim over admiral::plan<T>. See include/admiral/fftw3.h for the
// covered surface and conventions (unnormalized both directions; flags ignored).

#include "admiral/fftw3.h"
#include "admiral/admiral.hpp"

#include <complex>
#include <cstdlib>   // posix_memalign, free
#include <new>
#include <span>
#include <vector>

// One opaque handle for both precisions (as in FFTW), holding the yafft plan plus
// the arrays and direction captured at plan time so fftw_execute(p) can replay them.
struct fftw_plan_s {
    enum class Prec { F32, F64 };
    Prec prec;
    void* plan;   // admiral::plan<float>* or admiral::plan<double>*
    int   sign;
    void* in;
    void* out;

    ~fftw_plan_s() {
        if (prec == Prec::F64) delete static_cast<admiral::plan<double>*>(plan);
        else                   delete static_cast<admiral::plan<float>*>(plan);
    }
};

namespace {

constexpr size_t kAlign = 64;  // FFTW guarantees SIMD-friendly alignment.

void* aligned_malloc(size_t n) {
    void* p = nullptr;
    if (n == 0) n = 1;
    if (posix_memalign(&p, kAlign, n) != 0) return nullptr;
    return p;
}

// fct=1: FFTW is unnormalized both ways. in == out runs in place.
template<typename T>
void run(admiral::plan<T>* p, int sign, void* in, void* out) {
    auto* ci = static_cast<const std::complex<T>*>(in);
    auto* co = static_cast<std::complex<T>*>(out);
    const bool inplace = (in == out);
    if (sign == FFTW_FORWARD) {
        if (inplace) p->forward(co, T(1));
        else         p->forward(ci, co, T(1));
    } else {
        if (inplace) p->inverse(co, T(1));
        else         p->inverse(ci, co, T(1));
    }
}

template<typename T>
fftw_plan make_plan(fftw_plan_s::Prec prec, int rank, const int* n, void* in, void* out, int sign) {
    if (rank < 1 || n == nullptr) return nullptr;
    std::vector<size_t> shape(static_cast<size_t>(rank));
    for (int i = 0; i < rank; ++i) {
        if (n[i] < 1) return nullptr;
        shape[static_cast<size_t>(i)] = static_cast<size_t>(n[i]);
    }
    try {
        auto* cpp = new admiral::plan<T>(std::span<const size_t>(shape.data(), shape.size()));
        return new fftw_plan_s{prec, cpp, sign, in, out};
    } catch (...) {
        return nullptr;
    }
}

}  // namespace

// ---- double precision -------------------------------------------------------
extern "C" {

fftw_plan fftw_plan_dft(int rank, const int* n, fftw_complex* in, fftw_complex* out,
                        int sign, unsigned /*flags*/) {
    return make_plan<double>(fftw_plan_s::Prec::F64, rank, n, in, out, sign);
}
fftw_plan fftw_plan_dft_1d(int n0, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
    const int n[1] = {n0};
    return fftw_plan_dft(1, n, in, out, sign, flags);
}
fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
    const int n[2] = {n0, n1};
    return fftw_plan_dft(2, n, in, out, sign, flags);
}
fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex* in, fftw_complex* out,
                           int sign, unsigned flags) {
    const int n[3] = {n0, n1, n2};
    return fftw_plan_dft(3, n, in, out, sign, flags);
}

void fftw_execute(const fftw_plan p) {
    if (p == nullptr) return;
    run<double>(static_cast<admiral::plan<double>*>(p->plan), p->sign, p->in, p->out);
}
void fftw_execute_dft(const fftw_plan p, fftw_complex* in, fftw_complex* out) {
    if (p == nullptr) return;
    run<double>(static_cast<admiral::plan<double>*>(p->plan), p->sign, in, out);
}
void fftw_destroy_plan(fftw_plan p) { delete p; }

void*         fftw_malloc(size_t n)        { return aligned_malloc(n); }
void          fftw_free(void* p)           { free(p); }
double*       fftw_alloc_real(size_t n)    { return static_cast<double*>(aligned_malloc(n * sizeof(double))); }
fftw_complex* fftw_alloc_complex(size_t n) { return static_cast<fftw_complex*>(aligned_malloc(n * sizeof(fftw_complex))); }
void          fftw_cleanup(void)           {}

// ---- single precision -------------------------------------------------------
fftwf_plan fftwf_plan_dft(int rank, const int* n, fftwf_complex* in, fftwf_complex* out,
                          int sign, unsigned /*flags*/) {
    return make_plan<float>(fftw_plan_s::Prec::F32, rank, n, in, out, sign);
}
fftwf_plan fftwf_plan_dft_1d(int n0, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags) {
    const int n[1] = {n0};
    return fftwf_plan_dft(1, n, in, out, sign, flags);
}
fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags) {
    const int n[2] = {n0, n1};
    return fftwf_plan_dft(2, n, in, out, sign, flags);
}
fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out,
                             int sign, unsigned flags) {
    const int n[3] = {n0, n1, n2};
    return fftwf_plan_dft(3, n, in, out, sign, flags);
}

void fftwf_execute(const fftwf_plan p) {
    if (p == nullptr) return;
    run<float>(static_cast<admiral::plan<float>*>(p->plan), p->sign, p->in, p->out);
}
void fftwf_execute_dft(const fftwf_plan p, fftwf_complex* in, fftwf_complex* out) {
    if (p == nullptr) return;
    run<float>(static_cast<admiral::plan<float>*>(p->plan), p->sign, in, out);
}
void fftwf_destroy_plan(fftwf_plan p) { delete p; }

void*          fftwf_malloc(size_t n)        { return aligned_malloc(n); }
void           fftwf_free(void* p)           { free(p); }
float*         fftwf_alloc_real(size_t n)    { return static_cast<float*>(aligned_malloc(n * sizeof(float))); }
fftwf_complex* fftwf_alloc_complex(size_t n) { return static_cast<fftwf_complex*>(aligned_malloc(n * sizeof(fftwf_complex))); }
void           fftwf_cleanup(void)           {}

}  // extern "C"
