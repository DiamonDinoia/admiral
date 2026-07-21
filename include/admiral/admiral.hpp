#pragma once

#include <algorithm>  // std::copy (out-of-place staging)
#include <complex>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <array>
#include <initializer_list>
#include <stdexcept>
#include "detail/plan.hpp"
#include "detail/nd_plan.hpp"      // detail::nd_runtime_plan (N-D row-column engine)
#include "detail/real_fft.hpp"     // detail::nd_real_plan (N-D r2c/c2r engine)
#include "detail/thread_pool.hpp"  // detail::thread_pool (opt-in multithreading)

// The heavy route-tree engines (plan_impl / nd_runtime_plan / nd_real_plan) are
// instantiated ONCE in c_api.cpp (linked into fft_c). Every other TU including
// this header references those symbols instead of re-instantiating the full
// tree: measured, an engine-instantiating TU peaks ~3.2 GiB, these decls drop it
// to the ~0.3 GiB header-parse floor. Runtime is unchanged — transforms dispatch
// at runtime and were never inlined into callers. The defining TU sets
// ADM_INSTANTIATE_ENGINE (see c_api.cpp) to emit the definitions.
namespace admiral {

// Exception types
class error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ============================================================================
// Core Transform Functions (user manages memory)
// ============================================================================

// Forward FFT. Input and output spans (may alias for in-place) must have equal
// size. `nthreads` (1 = serial, 0 = auto) builds a temporary worker pool. `fct`
// scales the output; std::nullopt is the direction default (forward = 1).
template<typename T>
void forward(std::span<const std::complex<T>> input, std::span<std::complex<T>> output,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    if (input.size() != output.size()) [[unlikely]] {
        throw error("Input and output sizes must match");
    }
    if (input.size() == 0) [[unlikely]] {
        return;
    }
    // Out-of-place: stage input into the output buffer first.
    if (input.data() != output.data()) {
        std::copy(input.begin(), input.end(), output.begin());
    }
    // Route through the plan machinery so the public one-shot API gets the same
    // codelet / iterative / four-step / Bluestein dispatch (and fast-sincos
    // twiddle generation) as an explicit plan.
    auto pool = detail::make_temp_pool(nthreads);
    detail::plan_impl<T>(output.size(), /*is_forward=*/true)
        .execute(output, {.pool = pool.get(), .fct = fct});
}

// Overload for non-const input span (for convenience)
template<typename T>
void forward(std::span<std::complex<T>> input, std::span<std::complex<T>> output,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    forward(std::span<const std::complex<T>>(input), output, nthreads, fct);
}

// Inverse FFT. See forward(); `fct` defaults to 1/N so forward then inverse is
// the identity.
template<typename T>
void inverse(std::span<const std::complex<T>> input, std::span<std::complex<T>> output,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    if (input.size() != output.size()) [[unlikely]] {
        throw error("Input and output sizes must match");
    }
    if (input.size() == 0) [[unlikely]] {
        return;
    }
    // Out-of-place: stage input into the output buffer first.
    if (input.data() != output.data()) {
        std::copy(input.begin(), input.end(), output.begin());
    }
    auto pool = detail::make_temp_pool(nthreads);
    detail::plan_impl<T>(output.size(), /*is_forward=*/false)
        .execute(output, {.pool = pool.get(), .fct = fct});
}

// Overload for non-const input span (for convenience)
template<typename T>
void inverse(std::span<std::complex<T>> input, std::span<std::complex<T>> output,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    inverse(std::span<const std::complex<T>>(input), output, nthreads, fct);
}

// ============================================================================
// Plan Classes (for repeated transforms with pre-computed twiddle factors)
// ============================================================================

// Bidirectional plan (both forward and inverse), for any rank.
//
// The rank is a runtime property of the shape: 1D `plan(size)` and N-D
// `plan({d0,d1,...})` are the same type. It holds two detail::nd_runtime_plan<T>
// by value (forward + inverse) — a plain per-axis loop, no Dim template, no type
// erasure, no heap. The per-axis loop is not the hot path (the 1D kernels are),
// so a runtime rank costs nothing measurable. A 1D plan is just a rank-1 shape,
// which reproduces the legacy 1D path.
//
// Data is contiguous row-major (last axis fastest); span overloads validate the
// total element count, the pointer overloads trust the caller.
template<typename T>
class plan {
public:
    // `nthreads` (default 1) is opt-in multithreading: 1 = today's serial path,
    // byte-identical, zero threads spawned. nthreads > 1 caches nthreads-1 worker
    // threads in the plan (reused across execute() calls; see detail::thread_pool)
    // and threads the N-D batch loops.
    [[nodiscard]] explicit plan(std::size_t size, std::size_t nthreads = 1)
        : plan(std::array<std::size_t, 1>{size}, nthreads) {}

    [[nodiscard]] explicit plan(std::span<const std::size_t> shape, std::size_t nthreads = 1)
        : m{.fwd{shape, /*is_forward=*/true}, .inv{shape, /*is_forward=*/false}, .pool{}} {
        nthreads = detail::resolve_nthreads(nthreads);   // 0 -> hardware_concurrency (capped)
        if (nthreads > 1) m.pool = std::make_unique<detail::thread_pool>(nthreads);
    }

    [[nodiscard]] plan(std::initializer_list<std::size_t> shape, std::size_t nthreads = 1)
        : plan(std::span<const std::size_t>(shape.begin(), shape.size()), nthreads) {}

    template<std::size_t Dim>
    [[nodiscard]] explicit plan(const std::array<std::size_t, Dim>& shape, std::size_t nthreads = 1)
        : plan(std::span<const std::size_t>(shape.data(), Dim), nthreads) {}

    // Non-copyable, movable
    plan(const plan&) = delete;
    plan& operator=(const plan&) = delete;
    plan(plan&&) noexcept = default;
    plan& operator=(plan&&) noexcept = default;

    // Forward FFT. In place for the span / single-pointer forms; out-of-place for
    // the (src, dst) form (src preserved, must not alias dst). The span overload
    // validates the total element count; the pointer overloads trust the caller
    // (contiguous row-major, last axis fastest). `fct` scales the output;
    // std::nullopt is the direction default (forward = 1). The worker pool is a
    // plan member — never a call argument.
    void forward(std::span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
        if (data.size() != m.fwd.size()) [[unlikely]] {
            throw std::invalid_argument("Data size doesn't match plan size");
        }
        m.fwd.execute(data.data(), {.pool = m.pool.get(), .fct = fct});
    }
    void forward(std::complex<T>* data, std::optional<T> fct = std::nullopt) const {
        m.fwd.execute(data, {.pool = m.pool.get(), .fct = fct});
    }
    void forward(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const {
        m.fwd.execute(src, dst, {.pool = m.pool.get(), .fct = fct});
    }

    // Inverse FFT. Same overload set as forward(); `fct` defaults to 1/Ntot, so
    // forward then inverse is the identity.
    void inverse(std::span<std::complex<T>> data, std::optional<T> fct = std::nullopt) const {
        if (data.size() != m.inv.size()) [[unlikely]] {
            throw std::invalid_argument("Data size doesn't match plan size");
        }
        m.inv.execute(data.data(), {.pool = m.pool.get(), .fct = fct});
    }
    void inverse(std::complex<T>* data, std::optional<T> fct = std::nullopt) const {
        m.inv.execute(data, {.pool = m.pool.get(), .fct = fct});
    }
    void inverse(const std::complex<T>* src, std::complex<T>* dst,
                 std::optional<T> fct = std::nullopt) const {
        m.inv.execute(src, dst, {.pool = m.pool.get(), .fct = fct});
    }

    [[nodiscard]] std::size_t size() const noexcept { return m.fwd.size(); }

private:
    struct M {
        detail::nd_runtime_plan<T> fwd;
        detail::nd_runtime_plan<T> inv;
        // Opt-in worker pool (empty == serial). unique_ptr, NOT optional, because:
        //  (1) the workers capture the pool's `this` — a stable heap address keeps
        //      them valid across a plan move (inline optional storage would dangle);
        //  (2) thread_pool holds a mutex/cv (non-movable), so optional<thread_pool>
        //      would delete plan's defaulted move ctor;
        //  (3) unique_ptr::get() const yields a non-const thread_pool*, so the const
        //      execute() can hand the workers a mutable pool with no `mutable` hack.
        std::unique_ptr<detail::thread_pool> pool;
    } m;
};

// ============================================================================
// One-shot N-D Transforms (row-column algorithm; contiguous row-major, in place)
//
// An N-D complex FFT is a sequence of batched 1D transforms along each axis.
// `data` points to a contiguous row-major tensor of shape[0]*...*shape[Dim-1]
// complex<T> elements, last axis fastest. Strided / mdspan inputs are not (yet)
// supported. Normalization matches the 1D API: forward is unscaled, inverse
// divides by the total element count, so forward then inverse is the identity.
//
// For repeated transforms, build a reusable admiral::plan<T>(shape) instead.
// ============================================================================

// One-shot forward N-D FFT, in place. nthreads (default 1 = serial; 0 = auto)
// builds a temporary worker pool for this call only. `fct` scales the output
// (std::nullopt = direction default: forward = 1).
template<typename T, std::size_t Dim>
void forward(std::complex<T>* data, const std::array<std::size_t, Dim>& shape,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    auto pool = detail::make_temp_pool(nthreads);
    detail::nd_runtime_plan<T>(std::span<const std::size_t>(shape.data(), Dim),
                               /*is_forward=*/true).execute(data, {.pool = pool.get(), .fct = fct});
}

// One-shot inverse N-D FFT, in place. `fct` defaults to 1/Ntot. nthreads: see forward.
template<typename T, std::size_t Dim>
void inverse(std::complex<T>* data, const std::array<std::size_t, Dim>& shape,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    auto pool = detail::make_temp_pool(nthreads);
    detail::nd_runtime_plan<T>(std::span<const std::size_t>(shape.data(), Dim),
                               /*is_forward=*/false).execute(data, {.pool = pool.get(), .fct = fct});
}

// ============================================================================
// Real-to-complex N-D Transforms (r2c / c2r), for real input data.
//
// The real tensor is `shape` contiguous row-major (last axis fastest, and real).
// The complex half-spectrum tensor replaces the innermost extent shape[Dim-1]
// with shape[Dim-1]/2 + 1, exploiting the conjugate symmetry of a real
// transform (layout matches FFTW/ducc0). Normalization matches the c2c API:
// forward (r2c) is unscaled, inverse (c2r) divides so r2c then c2r is identity.
//
// For repeated transforms build a reusable admiral::plan_r2c<T>(shape) instead.
// ============================================================================

// Bidirectional N-D real-FFT plan. forward = r2c, inverse = c2r. Reusable
// (twiddles and inner c2c sub-plans built once), any rank from a runtime shape.
template<typename T>
class plan_r2c {
public:
    // `nthreads` (default 1 = serial, byte-identical) opt-in multithreading; the
    // worker pool is cached in the plan and reused across every forward/inverse.
    [[nodiscard]] explicit plan_r2c(std::span<const std::size_t> shape, std::size_t nthreads = 1)
        : m{shape} {
        nthreads = detail::resolve_nthreads(nthreads);   // 0 -> hardware_concurrency (capped)
        if (nthreads > 1) pool_ = std::make_unique<detail::thread_pool>(nthreads);
    }
    [[nodiscard]] plan_r2c(std::initializer_list<std::size_t> shape, std::size_t nthreads = 1)
        : plan_r2c(std::span<const std::size_t>(shape.begin(), shape.size()), nthreads) {}
    template<std::size_t Dim>
    [[nodiscard]] explicit plan_r2c(const std::array<std::size_t, Dim>& shape, std::size_t nthreads = 1)
        : plan_r2c(std::span<const std::size_t>(shape.data(), Dim), nthreads) {}

    plan_r2c(const plan_r2c&) = delete;
    plan_r2c& operator=(const plan_r2c&) = delete;
    plan_r2c(plan_r2c&&) noexcept = default;
    plan_r2c& operator=(plan_r2c&&) noexcept = default;

    // Forward r2c: real tensor `in` -> complex half-spectrum `out` (cplx_size()).
    // `fct` scales the output (std::nullopt = unscaled).
    void forward(const T* in, std::complex<T>* out, std::optional<T> fct = std::nullopt) const {
        m.forward(in, out, {.pool = pool_.get(), .fct = fct});
    }
    // Inverse c2r: complex half-spectrum `spec` (consumed) -> real tensor `out`.
    // `fct` scales the output (std::nullopt = 1/Ntot, so r2c then c2r is identity).
    void inverse(std::complex<T>* spec, T* out, std::optional<T> fct = std::nullopt) const {
        m.inverse(spec, out, {.pool = pool_.get(), .fct = fct});
    }

    // Element counts: real tensor (product of extents) and complex half-spectrum.
    [[nodiscard]] std::size_t real_size() const noexcept { return m.real_size(); }
    [[nodiscard]] std::size_t cplx_size() const noexcept { return m.cplx_size(); }

private:
    detail::nd_real_plan<T> m;
    std::unique_ptr<detail::thread_pool> pool_;  // opt-in worker pool (empty == serial)
};

// One-shot forward N-D r2c (unscaled): real `in` -> complex half-spectrum `out`.
// Overloads the c2c forward() — the real `in` pointer selects r2c by type.
// nthreads (default 1 = serial; 0 = auto) builds a temporary worker pool.
template<typename T, std::size_t Dim>
void forward(const T* in, std::complex<T>* out, const std::array<std::size_t, Dim>& shape,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    auto pool = detail::make_temp_pool(nthreads);
    detail::nd_real_plan<T>(std::span<const std::size_t>(shape.data(), Dim))
        .forward(in, out, {.pool = pool.get(), .fct = fct});
}

// One-shot inverse N-D c2r (scaled): complex half-spectrum `spec` (consumed) ->
// real `out`. Overloads the c2c inverse() — the real `out` pointer selects c2r by
// type. forward(r2c) then inverse(c2r) is the identity. nthreads: see forward.
template<typename T, std::size_t Dim>
void inverse(std::complex<T>* spec, T* out, const std::array<std::size_t, Dim>& shape,
             std::size_t nthreads = 1, std::optional<T> fct = std::nullopt) {
    auto pool = detail::make_temp_pool(nthreads);
    detail::nd_real_plan<T>(std::span<const std::size_t>(shape.data(), Dim))
        .inverse(spec, out, {.pool = pool.get(), .fct = fct});
}

} // namespace admiral
