#pragma once

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include <admiral/errors.hpp>

#include "detail/cxx_compat.hpp"

#ifndef ADM_API
#  if defined(_WIN32) || defined(__CYGWIN__)
#    define ADM_API __declspec(dllexport)
#  elif defined(__GNUC__)
#    define ADM_API __attribute__((visibility("default")))
#  else
#    define ADM_API
#  endif
#endif

// Every plan type below is safe to execute concurrently from several threads: an execute holds
// no shared mutable state. Pool dispatches serialize, so a shared plan adds no throughput.
namespace admiral {

namespace detail {

#if ADM_CXX20
template<typename T>
concept precision = is_precision_v<T>;
#else
template<typename T>
using precision_void_t = std::enable_if_t<is_precision_v<T>>;
#endif

inline constexpr char kSizeMismatch[] = "Data size doesn't match plan size";

template<typename T> struct plan_state;
template<typename T> struct axis_state;
template<typename T> struct strides_state;
template<typename T> struct real_state;
template<typename T> struct r2r_state;

}

// `dct2` is FFTW's REDFT10, `dct3` REDFT01, `dst2` RODFT10, `dst3` RODFT01, unnormalized.
// `inverse` applies the reciprocal kind scaled by 1/(2*N), so forward then inverse round-trips.
enum class r2r_kind { dct2, dct3, dst2, dst3 };

// `estimate` routes from the fitted cost model. `automatic` and `measure` also time the top
// candidates at construction, so the route depends on the host. `ADM_MEASURE=OFF` makes both inert.
enum class effort { estimate, automatic, measure };

struct options {
    // `0` auto: serial below 2^15 elements, else the power-of-two width minimising modelled work
    // plus wake cost, capped at the allowed physical cores. `1` forces serial, `n` forces `n`.
    std::size_t nthreads = 0;
    effort eff = effort::estimate;
    unsigned debug = 0;
};

#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
forward(detail::type_identity_t<span<const std::complex<T>>> input, span<std::complex<T>> output,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
inverse(detail::type_identity_t<span<const std::complex<T>>> input, span<std::complex<T>> output,
        const options& opts = {}, std::optional<T> fct = std::nullopt);

// Forward applies exp(-2*pi*i*k*n/N) unscaled; inverse applies exp(+2*pi*i*k*n/N) and divides
// by `size()`. A supplied `fct` REPLACES that default scale instead of multiplying it.
template<typename T>
class ADM_API plan {
    static_assert(detail::is_precision_v<T>,
                  "admiral: T must be float, double or long double");
public:
    [[nodiscard]] explicit plan(std::size_t size, const options& opts = {})
        : plan(span<const std::size_t>(&size, 1), opts) {}

    [[nodiscard]] explicit plan(span<const std::size_t> shape, const options& opts = {});

    [[nodiscard]] plan(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan(span<const std::size_t>(shape.begin(), shape.size()), opts) {}

    ~plan();
    plan(const plan&) = delete;
    plan& operator=(const plan&) = delete;
    plan(plan&&) noexcept;
    plan& operator=(plan&&) noexcept;

    // The `span` overloads throw `size_error` when the size differs from `size()`. The pointer
    // overloads check nothing and read and write exactly `size()` elements.
    void forward(span<std::complex<T>> data, std::optional<T> fct = std::nullopt) {
        check_size(data.size());
        run(true, data.data(), scale(fct));
    }
    void forward(std::complex<T>* data, std::optional<T> fct = std::nullopt) {
        run(true, data, scale(fct));
    }
    void forward(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) {
        run(true, src, dst, scale(fct));
    }

    void inverse(span<std::complex<T>> data, std::optional<T> fct = std::nullopt) {
        check_size(data.size());
        run(false, data.data(), scale(fct));
    }
    void inverse(std::complex<T>* data, std::optional<T> fct = std::nullopt) {
        run(false, data, scale(fct));
    }
    void inverse(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) {
        run(false, src, dst, scale(fct));
    }

    [[nodiscard]] std::size_t size() const noexcept;

private:
    void check_size(std::size_t n) const {
        if (n != size()) ADM_UNLIKELY throw size_error(detail::kSizeMismatch);
    }
    [[nodiscard]] static const T* scale(const std::optional<T>& fct) noexcept {
        return fct ? &*fct : nullptr;
    }
    void run(bool is_forward, std::complex<T>* data, const T* fct);
    void run(bool is_forward, const std::complex<T>* src, std::complex<T>* dst,
             const T* fct);

    std::unique_ptr<detail::plan_state<T>> m;
};

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

    // `lo` and `hi` are half-open bounds of rank `shape.size()`, or empty for the full extent.
    // The transformed axis must be whole: `lo[axis] == 0` and `hi[axis] == shape[axis]`.
    void execute(std::complex<T>* data, span<const std::size_t> lo,
                 span<const std::size_t> hi, std::optional<T> fct = std::nullopt);

    // Runs two disjoint bands in one dispatch. `lo2_last`/`hi2_last` replace the LAST axis bound
    // for the second band; every other bound is shared. `axis` must not be the last axis.
    void execute_bands(std::complex<T>* data, span<const std::size_t> lo,
                       span<const std::size_t> hi, std::size_t lo2_last,
                       std::size_t hi2_last, std::optional<T> fct = std::nullopt);

private:
    std::unique_ptr<detail::axis_state<T>> m;
};

template<typename T>
class ADM_API strides_plan {
    static_assert(detail::is_simd_precision_v<T>, "admiral: T must be float or double");
public:
    [[nodiscard]] strides_plan(std::size_t len, std::size_t nbatch, std::size_t in_stride,
                               std::size_t in_dist, std::size_t out_stride,
                               std::size_t out_dist, const options& opts = {});

    ~strides_plan();
    strides_plan(const strides_plan&) = delete;
    strides_plan& operator=(const strides_plan&) = delete;
    strides_plan(strides_plan&&) noexcept;
    strides_plan& operator=(strides_plan&&) noexcept;

    // In place (`src == dst`) requires `in_stride == out_stride` and `in_dist == out_dist`;
    // a mismatch throws `size_error`.
    void forward(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt);
    void inverse(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt);

    // Total complex elements across every batch, `len * nbatch`, not the per-transform length.
    [[nodiscard]] std::size_t size() const noexcept;

private:
    std::unique_ptr<detail::strides_state<T>> m;
};

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

template<typename T>
class ADM_API plan_r2c {
    static_assert(detail::is_precision_v<T>,
                  "admiral: T must be float, double or long double");
public:
    [[nodiscard]] explicit plan_r2c(std::size_t size, const options& opts = {})
        : plan_r2c(span<const std::size_t>(&size, 1), opts) {}

    [[nodiscard]] explicit plan_r2c(span<const std::size_t> shape, const options& opts = {});
    [[nodiscard]] plan_r2c(std::initializer_list<std::size_t> shape, const options& opts = {})
        : plan_r2c(span<const std::size_t>(shape.begin(), shape.size()), opts) {}

    ~plan_r2c();
    plan_r2c(const plan_r2c&) = delete;
    plan_r2c& operator=(const plan_r2c&) = delete;
    plan_r2c(plan_r2c&&) noexcept;
    plan_r2c& operator=(plan_r2c&&) noexcept;

    // `out` holds `cplx_size()` elements: the LAST axis is halved to `shape.back()/2 + 1`.
    void forward(const T* in, std::complex<T>* out, std::optional<T> fct = std::nullopt);
    // At rank >= 2 `inverse` overwrites `spec` while it runs. Copy `spec` first to keep it.
    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct = std::nullopt);

    void forward(span<const T> in, span<std::complex<T>> out,
                 std::optional<T> fct = std::nullopt) {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(span<std::complex<T>> spec, span<T> out,
                 std::optional<T> fct = std::nullopt) {
        check_sizes(out.size(), spec.size());
        inverse(spec.data(), out.data(), fct);
    }

    [[nodiscard]] std::size_t real_size() const noexcept;
    [[nodiscard]] std::size_t cplx_size() const noexcept;

private:
    void check_sizes(std::size_t nreal, std::size_t ncplx) const {
        if (nreal != real_size() || ncplx != cplx_size()) ADM_UNLIKELY
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::real_state<T>> m;
};

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

#if ADM_CXX20
template<detail::precision T>
ADM_API void
#else
template<typename T>
ADM_API detail::precision_void_t<T>
#endif
// At rank >= 2 this overwrites `spec` while it runs, as `plan_r2c::inverse` does.
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

template<typename T>
class ADM_API plan_r2r {
    static_assert(detail::is_simd_precision_v<T>, "admiral: T must be float or double");
public:
    // Transforms `rows` contiguous lines of `size` reals each; `size()` is their product.
    [[nodiscard]] plan_r2r(std::size_t size, r2r_kind kind, std::size_t rows = 1,
                           const options& opts = {});

    ~plan_r2r();
    plan_r2r(const plan_r2r&) = delete;
    plan_r2r& operator=(const plan_r2r&) = delete;
    plan_r2r(plan_r2r&&) noexcept;
    plan_r2r& operator=(plan_r2r&&) noexcept;

    void forward(const T* in, T* out, std::optional<T> fct = std::nullopt);
    void inverse(const T* in, T* out, std::optional<T> fct = std::nullopt);

    void forward(span<const T> in, span<T> out,
                 std::optional<T> fct = std::nullopt) {
        check_sizes(in.size(), out.size());
        forward(in.data(), out.data(), fct);
    }
    void inverse(span<const T> in, span<T> out,
                 std::optional<T> fct = std::nullopt) {
        check_sizes(in.size(), out.size());
        inverse(in.data(), out.data(), fct);
    }

    [[nodiscard]] std::size_t size() const noexcept;

private:
    void check_sizes(std::size_t nin, std::size_t nout) const {
        if (nin != size() || nout != size()) ADM_UNLIKELY
            throw size_error(detail::kSizeMismatch);
    }
    std::unique_ptr<detail::r2r_state<T>> m;
};

}
