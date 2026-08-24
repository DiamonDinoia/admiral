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
// Link admiral::admiral (shared) or admiral::admiral_static. T is float or
// double, and nothing else compiles. The engine is opaque here, so this header
// needs only the standard library.
//
// Everything is in namespace admiral; namespace admiral::detail is internal and
// has no stability guarantee.
//
// Layout. Contiguous row-major, last axis fastest, the same layout as FFTW.
// Strided data, mdspan and non-contiguous views are not supported.
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
#include <span>
#include <stdexcept>
#include <type_traits>

#include <admiral/errors.hpp>  // error, error_code, size_error

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

template<typename T>
concept precision = std::is_same_v<T, float> || std::is_same_v<T, double>;

// The one message every plan's span-extent check throws.
inline constexpr char kSizeMismatch[] = "Data size doesn't match plan size";

// Engine state, defined in the library. Declaring it is all this header needs.
template<typename T> struct plan_state;
template<typename T> struct axis_state;
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
template<detail::precision T>
ADM_API void forward(std::type_identity_t<std::span<const std::complex<T>>> input,
                     std::span<std::complex<T>> output, const options& opts = {},
                     std::optional<T> fct = std::nullopt);

/// Inverse 1-D FFT, scaled by 1/N. Otherwise identical to forward().
template<detail::precision T>
ADM_API void inverse(std::type_identity_t<std::span<const std::complex<T>>> input,
                     std::span<std::complex<T>> output, const options& opts = {},
                     std::optional<T> fct = std::nullopt);

// ============================================================================
// Plans
// ============================================================================

/// Complex plan, forward and inverse, any rank.
///
/// The rank is a runtime property, so plan(1024) and plan({64, 64}) are the same
/// type. One plan serves both directions: build it once, then call forward() and
/// inverse() any number of times.
template<detail::precision T>
class ADM_API plan {
public:
    /// 1-D over `size` complex elements.
    [[nodiscard]] explicit plan(std::size_t size, const options& opts = {})
        : plan(std::span<const std::size_t>(&size, 1), opts) {}

    /// N-D over `shape`, last axis fastest.
    [[nodiscard]] explicit plan(std::span<const std::size_t> shape, const options& opts = {});

    [[nodiscard]] plan(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan(std::span<const std::size_t>(shape.begin(), shape.size()), opts) {}

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
    void forward(std::span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
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
    void inverse(std::span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
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
        if (n != size()) [[unlikely]] throw size_error(detail::kSizeMismatch);
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
template<detail::precision T>
class ADM_API axis_plan {
public:
    [[nodiscard]] axis_plan(std::span<const std::size_t> shape, std::size_t axis, bool forward,
                            const options& opts = {});
    [[nodiscard]] axis_plan(std::initializer_list<std::size_t> shape, std::size_t axis,
                            bool forward, const options& opts = {})
        : axis_plan(std::span<const std::size_t>(shape.begin(), shape.size()), axis, forward,
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
    void execute(std::complex<T>* data, std::span<const std::size_t> lo,
                 std::span<const std::size_t> hi, std::optional<T> fct = std::nullopt) const;

    /// Two disjoint bands of the LAST dimension in one call: the box as given, plus
    /// [lo2_last, hi2_last) on the last dimension with every other dimension shared.
    /// Equivalent to two execute() calls, but cheaper for narrow bands, which the
    /// planner can pack into one pass chain. lo2_last == hi2_last is execute().
    void execute_bands(std::complex<T>* data, std::span<const std::size_t> lo,
                       std::span<const std::size_t> hi, std::size_t lo2_last,
                       std::size_t hi2_last, std::optional<T> fct = std::nullopt) const;

private:
    std::unique_ptr<detail::axis_state<T>> m;
};

// ============================================================================
// One-shot N-D transforms, in place. For repeated transforms use plan<T>(shape).
// ============================================================================

/// Forward N-D FFT, unscaled.
template<detail::precision T>
ADM_API void forward(std::complex<T>* data, std::span<const std::size_t> shape,
                     const options& opts = {}, std::optional<T> fct = std::nullopt);

template<detail::precision T>
void forward(std::complex<T>* data, std::initializer_list<std::size_t> shape,
             const options& opts = {}, std::optional<T> fct = std::nullopt) {
    forward(data, std::span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

/// Inverse N-D FFT, scaled by 1/Ntot.
template<detail::precision T>
ADM_API void inverse(std::complex<T>* data, std::span<const std::size_t> shape,
                     const options& opts = {}, std::optional<T> fct = std::nullopt);

template<detail::precision T>
void inverse(std::complex<T>* data, std::initializer_list<std::size_t> shape,
             const options& opts = {}, std::optional<T> fct = std::nullopt) {
    inverse(data, std::span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
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
template<detail::precision T>
class ADM_API plan_r2c {
public:
    /// 1-D over `size` REAL elements.
    [[nodiscard]] explicit plan_r2c(std::size_t size, const options& opts = {})
        : plan_r2c(std::span<const std::size_t>(&size, 1), opts) {}

    /// N-D over `shape`, last axis fastest.
    [[nodiscard]] explicit plan_r2c(std::span<const std::size_t> shape, const options& opts = {});
    [[nodiscard]] plan_r2c(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan_r2c(std::span<const std::size_t>(shape.begin(), shape.size()), opts) {}

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
    void forward(std::span<const T> in, std::span<std::complex<T>> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(std::span<std::complex<T>> spec, std::span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(out.size(), spec.size());
        inverse(spec.data(), out.data(), fct);
    }

    /// Buffer sizes in elements: the real tensor and the complex half-spectrum.
    [[nodiscard]] std::size_t real_size() const noexcept;
    [[nodiscard]] std::size_t cplx_size() const noexcept;

private:
    void check_sizes(std::size_t nreal, std::size_t ncplx) const {
        if (nreal != real_size() || ncplx != cplx_size()) [[unlikely]]
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::real_state<T>> m;
};

/// One-shot r2c, unscaled. The real `in` pointer is what picks this over the
/// complex forward() above.
template<detail::precision T>
ADM_API void forward(const T* in, std::complex<T>* out, std::span<const std::size_t> shape,
                     const options& opts = {}, std::optional<T> fct = std::nullopt);

template<detail::precision T>
void forward(const T* in, std::complex<T>* out, std::initializer_list<std::size_t> shape,
             const options& opts = {}, std::optional<T> fct = std::nullopt) {
    forward(in, out, std::span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
}

/// One-shot c2r, scaled by 1/Ntot. `spec` is overwritten.
template<detail::precision T>
ADM_API void inverse(std::complex<T>* spec, T* out, std::span<const std::size_t> shape,
                     const options& opts = {}, std::optional<T> fct = std::nullopt);

template<detail::precision T>
void inverse(std::complex<T>* spec, T* out, std::initializer_list<std::size_t> shape,
             const options& opts = {}, std::optional<T> fct = std::nullopt) {
    inverse(spec, out, std::span<const std::size_t>(shape.begin(), shape.size()), opts, fct);
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

template<detail::precision T>
class ADM_API plan_r2r {
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

    void forward(std::span<const T> in, std::span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(std::span<const T> in, std::span<T> out,
                 std::optional<T> fct = std::nullopt) const {
        check_sizes(in.size(), out.size());
        inverse(in.data(), out.data(), fct);
    }

    /// Elements per buffer: size * rows.
    [[nodiscard]] std::size_t size() const noexcept;

private:
    void check_sizes(std::size_t nin, std::size_t nout) const {
        if (nin != size() || nout != size()) [[unlikely]]
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::r2r_state<T>> m;
};

}  // namespace admiral
