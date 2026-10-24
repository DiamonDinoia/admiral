# C++ API

All types are in `admiral/admiral.hpp`, except the error types, which are in `admiral/errors.hpp`.

## Types

| Name | Purpose |
|------|---------|
| `plan<T>` | Complex transform of any rank. `T` is `float`, `double` or `long double`. |
| `plan_r2c<T>` | Real-to-complex transform in FFTW's packed layout. The last axis shrinks to `n/2 + 1`. |
| `plan_r2r<T>` | DCT and DST of one kind, one line or `rows` contiguous lines. `float` and `double` only. |
| `axis_plan<T>` | Transform of one axis, optionally inside a sub-box. `float` and `double` only. |
| `strides_plan<T>` | Out-of-place transform of `nbatch` strided lines. `float` and `double` only. |
| `options` | Settings struct. Every plan and every one-shot takes it. |
| `effort` | Route selection: `estimate`, `automatic`, `measure`. |
| `r2r_kind` | Transform kind: `dct2`, `dct3`, `dst2`, `dst3`. |

## plan

| Name | Purpose |
|------|---------|
| `plan(size_t n, const options& = {})` | 1-D plan of `n` elements. |
| `plan(span<const size_t> shape, const options& = {})` | N-D plan. The last axis is the fastest. A braced list also works. |
| `forward(span<complex<T>> x, optional<T> fct = {})` | In-place forward transform, unscaled. Throws `size_error` on a size mismatch. |
| `inverse(span<complex<T>> x, optional<T> fct = {})` | In-place inverse transform. Divides by `size()`. |
| `forward(complex<T>* x, optional<T> fct = {})` | Pointer overload. Reads and writes `size()` elements without a check. |
| `inverse(complex<T>* x, optional<T> fct = {})` | Pointer overload of the inverse. |
| `forward(const complex<T>* src, complex<T>* dst, optional<T> fct = {})` | Out-of-place forward transform. |
| `inverse(const complex<T>* src, complex<T>* dst, optional<T> fct = {})` | Out-of-place inverse transform. |
| `size()` | Element count the plan expects. |

## One-shots

| Name | Purpose |
|------|---------|
| `forward(span<const complex<T>> in, span<complex<T>> out, const options& = {}, optional<T> fct = {})` | 1-D complex transform without a plan. |
| `inverse(span<const complex<T>> in, span<complex<T>> out, const options& = {}, optional<T> fct = {})` | 1-D complex inverse without a plan. |
| `forward(complex<T>* x, span<const size_t> shape, const options& = {}, optional<T> fct = {})` | N-D in-place transform without a plan. |
| `inverse(complex<T>* x, span<const size_t> shape, const options& = {}, optional<T> fct = {})` | N-D in-place inverse without a plan. |
| `forward(const T* in, complex<T>* out, span<const size_t> shape, const options& = {}, optional<T> fct = {})` | Real-to-complex transform without a plan. |
| `inverse(complex<T>* spec, T* out, span<const size_t> shape, const options& = {}, optional<T> fct = {})` | Complex-to-real inverse without a plan. |

## plan_r2c

| Name | Purpose |
|------|---------|
| `plan_r2c(span<const size_t> shape, const options& = {})` | Plan for a real array of the given shape. A size or a braced list also works. |
| `forward(span<const T> in, span<complex<T>> out, optional<T> fct = {})` | Real to half spectrum. |
| `inverse(span<complex<T>> spec, span<T> out, optional<T> fct = {})` | Half spectrum to real. Divides by the real element count. |
| `forward(const T* in, complex<T>* out, optional<T> fct = {})` | Pointer overload. No size check. |
| `inverse(complex<T>* spec, T* out, optional<T> fct = {})` | Pointer overload. No size check. |
| `real_size()` | Length of the real buffer. |
| `cplx_size()` | Length of the complex buffer. |

## plan_r2r

| Name | Purpose |
|------|---------|
| `plan_r2r(size_t n, r2r_kind kind, size_t rows = 1, const options& = {})` | Plan for `rows` contiguous lines of length `n`. |
| `forward(span<const T> in, span<T> out, optional<T> fct = {})` | Applies the unnormalized kind. |
| `inverse(span<const T> in, span<T> out, optional<T> fct = {})` | Applies the exact inverse, scaled by 1/(2N). |
| `forward(const T* in, T* out, optional<T> fct = {})` | Pointer overload. No size check. |
| `inverse(const T* in, T* out, optional<T> fct = {})` | Pointer overload. No size check. |
| `size()` | Element count the plan expects. |

## axis_plan

| Name | Purpose |
|------|---------|
| `axis_plan(span<const size_t> shape, size_t axis, bool forward, const options& = {})` | Plan for one axis of a tensor of the given shape. |
| `execute(complex<T>* data, span<const size_t> lo, span<const size_t> hi, optional<T> fct = {})` | Transforms the sub-box `[lo, hi)`. Empty bounds mean the full extent. |
| `execute_bands(complex<T>* data, span<const size_t> lo, span<const size_t> hi, size_t lo2_last, size_t hi2_last, optional<T> fct = {})` | Runs the plan on two disjoint bands of the last dimension in one call. |

## strides_plan

| Name | Purpose |
|------|---------|
| `strides_plan(size_t len, size_t nbatch, size_t in_stride, size_t in_dist, size_t out_stride, size_t out_dist, const options& = {})` | Plan for `nbatch` lines of `len` elements with FFTW's `plan_many` geometry. |
| `forward(const complex<T>* src, complex<T>* dst, optional<T> fct = {})` | Forward transform. `src == dst` needs equal layouts. |
| `inverse(const complex<T>* src, complex<T>* dst, optional<T> fct = {})` | Inverse transform. |
| `size()` | Element count the plan expects. |

## options

| Field | Default | Purpose |
|-------|---------|---------|
| `nthreads` | `0` | Worker threads. `0` selects automatically, `1` forces serial, `n` forces `n`. |
| `eff` | `effort::estimate` | Route selection of the plan classes. `automatic` and `measure` time the top candidates at construction. One-shots always use `estimate`. |
| `debug` | `0` | Stderr trace per execute, level `0` to `3`. |
| `pin_threads` | `false` | Binds each pool worker to one physical core. Linux only. |

## Errors

| Name | Purpose |
|------|---------|
| `error` | Base class. Derives from `std::runtime_error`. `code()` returns an `error_code`. |
| `size_error` | `invalid_size`: zero extent, size mismatch, axis out of range. |
| `unsupported_error` | `unsupported`: a forced route that the size does not allow. |
| `internal_error` | `internal`: a broken invariant, not a bad argument. |
| `error_code` | Enum: `invalid_size`, `unsupported`, `internal`. |
