
#include "admiral/fftw3.h"
#include "admiral/admiral.hpp"
#include "admiral/detail/scratch.hpp"

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if !defined(_WIN32)
#include <dlfcn.h>
#endif
#if defined(__linux__)
#include <link.h>
#include <string>
#endif
#if defined(_MSC_VER)
#include <malloc.h>
#endif
#include <memory>
#include <utility>
#include <vector>
#include "admiral/detail/cxx_compat.hpp"

template<typename T>
struct fftw_handle {
    using value_type = T;

    // A plan from another FFTW starts with a pointer, never this value. One value for each
    // precision, so a float plan given to a double call is foreign too.
    static constexpr std::uint64_t kTag =
        sizeof(T) == sizeof(double) ? 0x41444d4646545744ull : 0x41444d4646545746ull;  // ADMFFTWD/F

    template<typename... Args>
    fftw_handle(int s, void* i, void* o, Args&&... args)
        : plan(std::forward<Args>(args)...), sign(s), in(i), out(o) {}

    std::uint64_t tag = kTag;
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

#if defined(__linux__)
struct later_objects {
    bool after_shim = false;
    std::vector<std::string> names;
};

// dl_iterate_phdr callback: the names of the objects loaded after the one that holds this code.
// It must not call dlopen, because dlopen takes the two loader locks in the other order.
int add_later_object(dl_phdr_info* info, size_t, void* arg) {
    auto& w = *static_cast<later_objects*>(arg);
    if (w.after_shim) {
        w.names.emplace_back(info->dlpi_name);
        return 0;
    }
    const auto self = reinterpret_cast<ElfW(Addr)>(&add_later_object);
    for (ElfW(Half) i = 0; i < info->dlpi_phnum; ++i) {
        const ElfW(Phdr)& ph = info->dlpi_phdr[i];
        if (ph.p_type == PT_LOAD && self - (info->dlpi_addr + ph.p_vaddr) < ph.p_memsz)
            w.after_shim = true;
    }
    return 0;
}

// The first object loaded after this one that defines `name` itself, whatever its soname and
// scope: RTLD_NEXT cannot see an FFTW that a plugin opened with dlopen(RTLD_LOCAL).
// The walk skips the objects loaded before, so a second copy of the shim cannot find the first.
// The handle of the hit stays open and keeps the pointer valid. noexcept: a failed allocation
// terminates here and does not go through the C ABI of fftw_destroy_plan.
void* later_definition(const char* name) noexcept {
    later_objects w;
    dl_iterate_phdr(add_later_object, &w);
    for (const std::string& lib : w.names) {
        void* h = dlopen(lib.c_str(), RTLD_LAZY | RTLD_NOLOAD);
        if (h == nullptr) continue;
        void* f = dlsym(h, name);
        Dl_info d;
        // dlsym also searches the dependencies of `lib`. Each of them comes up in the walk.
        if (f != nullptr && dladdr(f, &d) != 0 && lib == d.dli_fname) return f;
        dlclose(h);
    }
    return nullptr;
}
#endif

// What next_fftw searched, for the abort message.
#if defined(__linux__)
constexpr const char* kSearched = "no library loaded after this one defines";
#elif !defined(_WIN32)
constexpr const char* kSearched = "dlsym(RTLD_NEXT) does not find";
#else
constexpr const char* kSearched = "this platform does not look for";
#endif

// The next FFTW's `name`: dlsym(RTLD_NEXT), then on Linux later_definition.
// With none found: null, or a message and abort when `required`.
template<typename F>
F* next_fftw(const char* name, bool required = true) {
#if !defined(_WIN32)
    void* f = dlsym(RTLD_NEXT, name);
#if defined(__linux__)
    if (f == nullptr) f = later_definition(name);
#endif
    if (f != nullptr) return reinterpret_cast<F*>(f);
#endif
    // Windows has no RTLD_NEXT. A program can import planners from libfftw3-3.dll and execute
    // from admiral_fftw.dll, so a foreign plan can arrive there too and it aborts.
    if (!required) return nullptr;
    std::fprintf(stderr, "admiral_fftw: %s got a plan this library did not create, and %s %s\n",
                 name, kSearched, name);
    std::abort();
}

thread_local bool forwarding = false;

// Calls the next FFTW. That FFTW can call this library again: a second copy of the shim does
// when its own search finds this copy. The second visit on one thread aborts instead of looping.
template<typename F, typename... A>
void forward(F* next, const char* name, A... args) {
    if (forwarding) {
        std::fprintf(stderr, "admiral_fftw: %s came back to this library from the next FFTW\n",
                     name);
        std::abort();
    }
    forwarding = true;
    next(args...);
    forwarding = false;
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

// The planner's failure channel is a NULL plan, so the whole body — the shape vector's
// allocation included — runs inside the try: nothing crosses the extern "C" boundary.
template<typename H>
H* make_plan(int rank, const int* n, void* in, void* out, int sign, unsigned flags) {
    try {
        if (rank < 1 || n == nullptr) return nullptr;
        std::vector<size_t> shape(static_cast<size_t>(rank));
        for (size_t i = 0; i < shape.size(); ++i) {
            if (n[i] < 1) return nullptr;
            shape[i] = static_cast<size_t>(n[i]);
        }
        const admiral::effort eff =
            (flags & FFTW_ESTIMATE) ? admiral::effort::estimate : admiral::effort::automatic;
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
    void PREFIX##_execute(const PREFIX##_plan p) noexcept { \
        if (p == nullptr) return; \
        if (p->tag == PREFIX##_plan_s::kTag) return run(p, p->in, p->out); \
        static const auto next = next_fftw<void(PREFIX##_plan)>(#PREFIX "_execute"); \
        forward(next, #PREFIX "_execute", p); \
    } \
    void PREFIX##_execute_dft(const PREFIX##_plan p, COMPLEX_T* in, \
                              COMPLEX_T* out) noexcept { \
        if (p == nullptr) return; \
        if (p->tag == PREFIX##_plan_s::kTag) return run(p, in, out); \
        static const auto next = \
            next_fftw<void(PREFIX##_plan, COMPLEX_T*, COMPLEX_T*)>(#PREFIX "_execute_dft"); \
        forward(next, #PREFIX "_execute_dft", p, in, out); \
    } \
    void PREFIX##_destroy_plan(PREFIX##_plan p) { \
        if (p == nullptr) return; \
        if (p->tag == PREFIX##_plan_s::kTag) return delete p; \
        static const auto next = next_fftw<void(PREFIX##_plan)>(#PREFIX "_destroy_plan"); \
        forward(next, #PREFIX "_destroy_plan", p); \
    } \
    void*    PREFIX##_malloc(size_t n)                 { return aligned_malloc(n); } \
    void     PREFIX##_free(void* p)                    { aligned_free(p); } \
    T*       PREFIX##_alloc_real(size_t n)             { return static_cast<T*>(aligned_malloc(n * sizeof(T))); } \
    COMPLEX_T* PREFIX##_alloc_complex(size_t n)        { return static_cast<COMPLEX_T*>(aligned_malloc(n * sizeof(COMPLEX_T))); } \
    void PREFIX##_cleanup(void) { \
        static const auto next = next_fftw<void()>(#PREFIX "_cleanup", false); \
        if (next != nullptr) forward(next, #PREFIX "_cleanup"); \
    }

ADM_FFTW_SHIM(fftw,  double, fftw_complex)
ADM_FFTW_SHIM(fftwf, float,  fftwf_complex)

}
