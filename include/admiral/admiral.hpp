#pragma once

// ============================================================================
// Admiral C++ API: complex and real FFTs, 1-D and N-D, float or double.
//
//   #include <admiral/admiral.hpp>
//
//   std::vector<std::complex<double>> x(1024);
//   admiral::plan<double> p(x.size());   // reusable
//   p.forward(x);                        // in place
//   p.inverse(x);                        // divides by 1024
//
// Link admiral::admiral (shared) or admiral::admiral_static. T is float,
// double or long double. long double runs a scalar backend, because no ISA has
// 80-bit SIMD lanes, and only plan, plan_r2c and the one-shots offer it;
// axis_plan and plan_r2r stay float and double. The engine is opaque here, so
// this header needs only the standard library.
//
// Everything is in namespace admiral; namespace admiral::detail is internal and
// has no stability guarantee.
//
// Layout. Contiguous row-major, last axis fastest, the same layout as FFTW.
// strides_plan is the exception: it takes a batch of strided lines. General mdspan
// views are not supported.
//
// Sign and scale. Forward uses exp(-2*pi*i*k*n/N) and is unscaled. Inverse uses
// exp(+2*pi*i*k*n/N) and divides by the element count, so forward then inverse
// gives back the input. Every call takes an optional `fct` that overrides the
// output scale: pass fct = 1 for an unscaled inverse, matching FFTW.
//
// Aliasing. The (src, dst) overloads are out of place when src != dst and in
// place when src == dst. Buffers that partially overlap are undefined
// behaviour. The engine never stages a copy to hide it.
//
// Errors. Span overloads check the element count and throw admiral::size_error
// on a mismatch. Pointer overloads trust the caller. Construction throws
// admiral::size_error for a bad shape and std::bad_alloc if scratch cannot be
// allocated. Every caller-caused failure derives from admiral::error and
// carries an error_code; see <admiral/errors.hpp>.
//
// Options. Plans and one-shots alike take one optional `options` aggregate (see
// its declaration below) rather than a parameter per knob.
//
// Threads. options::nthreads = 0 (default) is auto: a size-aware heuristic picks
// the worker count, serial for small transforms, capped at one per physical
// core; 1 forces serial, n forces n. A plan owns its workers and reuses them
// across calls; execute is const but not reentrant, so do not run one plan from
// two threads at once. A threaded plan must not cross fork(): the child inherits
// no workers; replan there.
//
// Plan or one-shot. A plan holds the twiddle tables and the worker threads, so
// reuse one whenever the same shape transforms more than once. The free
// forward()/inverse() functions build and discard a plan per call, so they
// always route with effort::estimate. A measuring effort cannot repay on a discarded
// plan, so opts.eff is ignored there.
// ============================================================================

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <admiral/errors.hpp>     // error, error_code, size_error

// A C++20 toolchain takes std::span and the other std spellings from here by
// alias, so the API below carries one spelling of each.
#include "detail/cxx_compat.hpp"  // ADM_UNLIKELY, span, detail::type_identity_t,
                                  // detail::is_precision_v

// The library is built with hidden visibility; public symbols opt back in.
#ifndef ADM_API
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define ADM_API __declspec(dllexport)
#  elif defined(__GNUC__)
#    define ADM_API __attribute__((visibility("default")))
#  else
#    define ADM_API
#  endif
#endif

namespace admiral {

namespace detail {

// The element types the C++ API compiles for. The free functions, `plan` and
// `plan_r2c` take all three; `axis_plan` and `plan_r2r` take the two the SIMD engine
// has kernels for. long double runs the scalar backend (detail/scalar_fft.hpp),
// because no ISA has 80-bit SIMD lanes.
//
// The free functions constrain T so a wrong T leaves overload resolution: the concept
// at C++20, and at C++17 `precision_void_t` as the return type, which is `void` for a
// supported T and ill-formed for any other. The class templates use a static_assert in
// both standards, which names the offending T.
#if ADM_CXX20
template<typename T>
concept precision = is_precision_v<T>;
#else
template<typename T>
using precision_void_t = std::enable_if_t<is_precision_v<T>>;
#endif

// The one message every plan's span-extent check throws.
inline constexpr char kSizeMismatch[] = "Data size doesn't match plan size";

// Engine state, defined in the library. Declaring it is all this header needs.
template<typename T> struct plan_state;
template<typename T> struct axis_state;
template<typename T> struct strides_state;
template<typename T> struct real_state;
template<typename T> struct r2r_state;

}  // namespace detail

/// Which real-to-real basis plan_r2r uses. Named for the transform, not for FFTW's
/// REDFT/RODFT spelling: dct2 is REDFT10, dct3 is REDFT01, dst2 is RODFT10, dst3 is
/// RODFT01. Each kind's inverse() is its own exact inverse, so the type-2/type-3
/// pairing FFTW expresses by swapping kinds is here just forward vs inverse.
enum class r2r_kind { dct2, dct3, dst2, dst3 };

/// How hard plan construction works to pick the execution path, like FFTW's
/// ESTIMATE/MEASURE:
///   estimate   cost-model pick only. Fast and bitwise reproducible. Default.
///   automatic  also time the model's top candidates and keep the fastest. Worth it
///              when one plan serves many transforms (~10 us planning at N=64,
///              ~0.1 s at N=200000). The elected plan depends on the machine.
///   measure    identical to automatic; kept for the FFTW flag mapping. The engine
///              has one search budget, so PATIENT/EXHAUSTIVE buy nothing extra.
/// With -DADM_MEASURE=OFF the measuring efforts are accepted and inert.
enum class effort { estimate, automatic, measure };

/// Everything a plan is built with, in one aggregate so a call site names what it
/// sets and nothing else:
///   admiral::plan<float> p(4096, {.nthreads = 0, .eff = admiral::effort::automatic});
///
/// debug is a verbosity, not a flag: 0 prints nothing, 1 prints one line per execute
/// saying what ran, 2 adds the shape the route chose, 3 adds the cost-model ranking that
/// chose it (each form's modelled cycles and whether it was buildable). Tracing goes to
/// stderr and is fixed at construction, so a traced run and a silent one differ in no
/// other way.
struct options {
    std::size_t nthreads = 0;   ///< 0 = auto (size-aware, capped at one per physical core)
    effort eff = effort::estimate;
    unsigned debug = 0;
};

// ============================================================================
// One-shot 1-D transforms
// ============================================================================

/// Forward 1-D FFT, unscaled. `input` and `output` must have the same size; pass
/// the same span twice to transform in place.
/// T is deduced from `output` alone, so one overload serves every caller.
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
forward(detail::type_identity_t<span<const std::complex<T>>> input, span<std::complex<T>> output,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

/// Inverse 1-D FFT, scaled by 1/N. Otherwise identical to forward().
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
inverse(detail::type_identity_t<span<const std::complex<T>>> input, span<std::complex<T>> output,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

// ============================================================================
// Plans
// ============================================================================

/// Complex plan, forward and inverse, any rank.
///
/// The rank is a runtime property, so plan(1024) and plan({64, 64}) are the same
/// type. One plan serves both directions: build it once, then call forward() and
/// inverse() any number of times.
template<typename T>
class ADM_API plan {
    static_assert(detail::is_precision_v<T>,
                  "admiral: T must be float, double or long double");
public:
    /// 1-D over `size` complex elements.
    [[nodiscard]] explicit plan(std::size_t size, const options& opts = {})
        : plan(span<const std::size_t>(&size, 1), opts) {}

    /// N-D over `shape`, last axis fastest.
    [[nodiscard]] explicit plan(span<const std::size_t> shape, const options& opts = {});

    [[nodiscard]] plan(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan(span<const std::size_t>(shape.begin(), shape.size()), opts) {}

    ~plan();
    plan(const plan&) = delete;
    plan& operator=(const plan&) = delete;
    plan(plan&&) noexcept;
    plan& operator=(plan&&) noexcept;

    /// Forward FFT, unscaled. Three ways to call it: a span or a pointer to
    /// transform in place, or (src, dst) which is out of place when src != dst
    /// and leaves src untouched.
    ///
    /// Prefer (src, dst) over a manual copy. The engine reads the source during
    /// its first pass rather than in a separate sweep, which saves one pass over the data.
    void forward(span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
        check_size(data.size());
        run(true, data.data(), scale(fct));
    }
    void forward(std::complex<T>* data, std::optional<T> fct = std::nullopt) const {
        run(true, data, scale(fct));
    }
    void forward(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const {
        run(true, src, dst, scale(fct));
    }

    /// Inverse FFT, scaled by 1/Ntot. Same three overloads as forward().
    void inverse(span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
        check_size(data.size());
        run(false, data.data(), scale(fct));
    }
    void inverse(std::complex<T>* data, std::optional<T> fct = std::nullopt) const {
        run(false, data, scale(fct));
    }
    void inverse(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const {
        run(false, src, dst, scale(fct));
    }

    /// Total complex element count, i.e. the product of the extents.
    [[nodiscard]] std::size_t size() const noexcept;

private:
    void check_size(std::size_t n) const {
        if (n != size()) ADM_UNLIKELY throw size_error(detail::kSizeMismatch);
    }
    // In place and out of place are different passes in the engine, not one path
    // with src == dst.
    // The scale crosses as a pointer, not an optional: gcc reassembles an
    // optional<T> argument through the stack at every frame. Null means the
    // direction's default.
    [[nodiscard]] static const T* scale(const std::optional<T>& fct) noexcept {
        return fct ? &*fct : nullptr;
    }
    void run(bool is_forward, std::complex<T>* data, const T* fct) const;
    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             const T* fct) const;

    std::unique_ptr<detail::plan_state<T>> m;
};

/// Transform ONE axis of a fixed-shape tensor, in place, over a rectangular
/// sub-box. The transformed axis must be whole; the others may be bands. The
/// direction is fixed at construction. This is ducc0's
/// c2c(subarray(data, box), {axis}, forward, fct).
template<typename T>
class ADM_API axis_plan {
    static_assert(detail::is_simd_precision_v<T>, "admiral: T must be float or double");
public:
    [[nodiscard]] axis_plan(span<const std::size_t> shape, std::size_t axis, bool forward,
                            const options& opts = {});
    [[nodiscard]] axis_plan(std::initializer_list<std::size_t> shape, std::size_t axis,
                            bool forward, const options& opts = {})
        : axis_plan(span<const std::size_t>(shape.begin(), shape.size()), axis, forward,
                    opts) {}

    ~axis_plan();
    axis_plan(const axis_plan&) = delete;
    axis_plan& operator=(const axis_plan&) = delete;
    axis_plan(axis_plan&&) noexcept;
    axis_plan& operator=(axis_plan&&) noexcept;

    /// Half-open box: dimension d runs over lo[d]..hi[d]. An empty span means the
    /// full extent, and an empty box is a no-op. The transformed axis must be
    /// whole, so lo[axis] = 0 and hi[axis] = shape[axis]. fct defaults to 1 for
    /// forward and 1/len for inverse.
    void execute(std::complex<T>* data, span<const std::size_t> lo,
                 span<const std::size_t> hi, std::optional<T> fct = std::nullopt) const;

    /// Two disjoint bands of the LAST dimension in one call: the box as given, plus
    /// [lo2_last, hi2_last) on the last dimension with every other dimension shared.
    /// Equivalent to two execute() calls, but cheaper for narrow bands, which the
    /// planner can pack into one pass chain. lo2_last == hi2_last is execute().
    void execute_bands(std::complex<T>* data, span<const std::size_t> lo,
                       span<const std::size_t> hi, std::size_t lo2_last,
                       std::size_t hi2_last, std::optional<T> fct = std::nullopt) const;

private:
    std::unique_ptr<detail::axis_state<T>> m;
};

/// Batched 1-D FFT over `nbatch` strided lines of `len` complex elements, out of
/// place. The geometry is FFTW's plan_many(rank = 1), under the parameter names
/// FFTW and cuFFT both use: a stride separates the elements of one transform, a
/// dist separates consecutive transforms. The two layouts are independent:
///
///   transform l, element p:
///       src[p * in_stride + l * in_dist] -> dst[p * out_stride + l * out_dist]
///
/// Where it sits: plan transforms a whole contiguous tensor, axis_plan one axis of a
/// contiguous tensor in place, strides_plan one batch of strided lines out of place.
/// It builds both directions and picks per call, the way plan does. axis_plan is the
/// one that fixes its direction at construction.
///
/// The INPUT geometry alone picks the numbers: a given (len, in_stride, in_dist)
/// returns bit-identical results into every output layout, so results stay
/// comparable across layouts. Unit input dist runs the batched SIMD column pass
/// straight out of the source; other input strides move cache-resident groups of
/// columns through contiguous scratch. Unit input stride is a batch of contiguous
/// rows, one contiguous transform per line.
/// src == dst transforms in place, and then requires the two layouts to match.
/// Buffers that overlap without being equal are undefined behaviour, as everywhere
/// else in this header. T is float or double, because the strided kernels are SIMD.
///
/// A plan owns scratch that every call overwrites, so one plan serves one call at a
/// time. Concurrent calls need one plan each; opts.nthreads threads inside a call.
template<typename T>
class ADM_API strides_plan {
    static_assert(detail::is_simd_precision_v<T>, "admiral: T must be float or double");
public:
    /// Strides count std::complex<T> elements. Zero len, zero nbatch, zero stride and
    /// (with nbatch > 1) zero dist throw size_error, as does a len * nbatch that does
    /// not fit size_t.
    [[nodiscard]] strides_plan(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                               std::size_t in_dist, std::size_t out_stride,
                               std::size_t out_dist, const options& opts = {});

    ~strides_plan();
    strides_plan(const strides_plan&) = delete;
    strides_plan& operator=(const strides_plan&) = delete;
    strides_plan(strides_plan&&) noexcept;
    strides_plan& operator=(strides_plan&&) noexcept;

    /// Forward FFT, unscaled. fct overrides the output scale.
    void forward(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const;
    /// Inverse FFT, scaled by 1/len. Otherwise identical to forward().
    void inverse(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const;

    /// Elements touched per buffer: len * nbatch.
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::unique_ptr<detail::strides_state<T>> m;
};

// ============================================================================
// One-shot N-D transforms, in place. For repeated transforms use plan<T>(shape).
// ============================================================================

/// Forward N-D FFT, unscaled.
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
forward(std::complex<T>* data, span<const std::size_t> shape, const options& opts = {},
        std::optional<T> fct = std::nullopt);

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(std::complex<T>* data, std::initializer_list<std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt) {
    forward(data, span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

/// Inverse N-D FFT, scaled by 1/Ntot.
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
inverse(std::complex<T>* data, span<const std::size_t> shape, const options& opts = {},
        std::optional<T> fct = std::nullopt);

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
inverse(std::complex<T>* data, std::initializer_list<std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt) {
    inverse(data, span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

// ============================================================================
// Real transforms (r2c / c2r), N-D, out of place
//
// `shape` gives the extents of the REAL tensor. A real signal has a conjugate-
// symmetric spectrum, so a buffer holds only half of it. The complex tensor keeps
// every extent except the innermost, which becomes shape[Dim-1]/2 + 1. That is
// FFTW's and ducc0's half-spectrum layout, so buffers port directly. Use
// cplx_size() rather than computing it yourself.
//
// r2c is unscaled, c2r divides by the real element count, and c2r overwrites
// its input spectrum.
// ============================================================================

/// Real plan, r2c and c2r, any rank.
template<typename T>
class ADM_API plan_r2c {
    static_assert(detail::is_precision_v<T>,
                  "admiral: T must be float, double or long double");
public:
    /// 1-D over `size` REAL elements.
    [[nodiscard]] explicit plan_r2c(std::size_t size, const options& opts = {})
        : plan_r2c(span<const std::size_t>(&size, 1), opts) {}

    /// N-D over `shape`, last axis fastest.
    [[nodiscard]] explicit plan_r2c(span<const std::size_t> shape, const options& opts = {});
    [[nodiscard]] plan_r2c(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan_r2c(span<const std::size_t>(shape.begin(), shape.size()), opts) {}

    ~plan_r2c();
    plan_r2c(const plan_r2c&) = delete;
    plan_r2c& operator=(const plan_r2c&) = delete;
    plan_r2c(plan_r2c&&) noexcept;
    plan_r2c& operator=(plan_r2c&&) noexcept;

    /// `in` holds real_size() reals, `out` holds cplx_size() complex elements.
    void forward(const T* in, std::complex<T>* out, std::optional<T> fct = std::nullopt) const;
    /// `spec` is overwritten.
    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct = std::nullopt) const;

    /// Span overloads, as on plan<T>: they exist for the size check, since the two
    /// buffers here have DIFFERENT lengths (real_size() vs cplx_size() == n/2+1 on the
    /// last axis) and swapping them is the mistake the pointer form cannot catch.
    void forward(span<const T> in, span<std::complex<T>> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(span<std::complex<T>> spec, span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(out.size(), spec.size());
        inverse(spec.data(), out.data(), fct);
    }

    /// Buffer sizes in elements: the real tensor and the complex half-spectrum.
    [[nodiscard]] std::size_t real_size() const noexcept;
    [[nodiscard]] std::size_t cplx_size() const noexcept;

private:
    void check_sizes(std::size_t nreal, std::size_t ncplx) const {
        if (nreal != real_size() || ncplx != cplx_size()) ADM_UNLIKELY
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::real_state<T>> m;
};

/// One-shot r2c, unscaled. The real `in` pointer is what picks this over the
/// complex forward() above.
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
forward(const T* in, std::complex<T>* out, span<const std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
forward(const T* in, std::complex<T>* out, std::initializer_list<std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt) {
    forward(in, out, span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

/// One-shot c2r, scaled by 1/Ntot. `spec` is overwritten.
#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
inverse(std::complex<T>* spec, T* out, span<const std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

#if ADM_CXX20
template<detail::precision T>
void
#else
template<typename T>
detail::precision_void_t<T>
#endif
inverse(std::complex<T>* spec, T* out, std::initializer_list<std::size_t> shape,
        const options& opts = {}, std::optional<T> fct = std::nullopt) {
    inverse(spec, out, span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

// ============================================================================
// Real-to-real transforms (DCT / DST), 1-D, out of place
//
// Costs one real FFT of the SAME length N, not of the 2N even extension, plus
// two O(N) passes, via Makhoul's shuffle.
//
// Normalization is the library's, not FFTW's. forward() is FFTW's unnormalized
// kind and inverse() is its exact inverse, so forward -> inverse round-trips to
// the input. FFTW's own type-2/type-3 pair differs by 2N in that position; pass
// fct for FFTW's convention.
// ============================================================================

template<typename T>
class ADM_API plan_r2r {
    static_assert(detail::is_simd_precision_v<T>, "admiral: T must be float or double");
public:
    /// `rows` contiguous lines of `size` reals each, one kind for all of them. Only
    /// the innermost axis of a tensor is contiguous, so N-D needs a transpose the
    /// caller does. There is no strided r2r here yet.
    [[nodiscard]] plan_r2r(std::size_t size, r2r_kind kind, std::size_t rows = 1,
                           const options& opts = {});

    ~plan_r2r();
    plan_r2r(const plan_r2r&) = delete;
    plan_r2r& operator=(const plan_r2r&) = delete;
    plan_r2r(plan_r2r&&) noexcept;
    plan_r2r& operator=(plan_r2r&&) noexcept;

    /// Both buffers hold size() reals. `out == in` is allowed: both directions stage
    /// the entire input into scratch before storing anything.
    void forward(const T* in, T* out, std::optional<T> fct = std::nullopt) const;
    void inverse(const T* in, T* out, std::optional<T> fct = std::nullopt) const;

    void forward(span<const T> in, span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(span<const T> in, span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        inverse(in.data(), out.data(), fct);
    }

    /// Elements per buffer: size * rows.
    [[nodiscard]] std::size_t size() const noexcept;

private:
    void check_sizes(std::size_t nin, std::size_t nout) const {
        if (nin != size() || nout != size()) ADM_UNLIKELY
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::r2r_state<T>> m;
};

}  // namespace admiral
