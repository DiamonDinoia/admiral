# The C++ API

Everything lives in `<admiral/admiral.hpp>`. Complex plans and one-shots come
in `float` and `double`; keep a plan when a size repeats, one-shots when it
does not. Every snippet below has a runnable counterpart under
[examples/](https://github.com/DiamonDinoia/admiral/tree/master/examples/), built and run by ctest.

## Complex plans

```cpp
admiral::plan<double> p(1024);          // 1-D
admiral::plan<double> q({64, 64, 8});   // any rank, last axis fastest
p.forward(x);                           // in place
p.inverse(x);                           // divides by the element count
```

Forward is unscaled; inverse divides by the element count, so a round trip
returns the input. Pass a third argument (`fct`) or use the `(src, dst)`
overloads for out-of-place work. `p.size()` returns the element count the plan
expects. [example](https://github.com/DiamonDinoia/admiral/blob/master/examples/quickstart.cpp)

One-shots do the same without a plan, always route with `effort::estimate`:

```cpp
admiral::forward<double>(x, x);
admiral::forward<double>(y.data(), {64, 64});   // N-D, in place
```

## Real transforms

`plan_r2c` transforms real input to half the spectrum, FFTW's packed layout
(last axis shrinks to `n/2 + 1`): [example](https://github.com/DiamonDinoia/admiral/blob/master/examples/real_transforms.cpp)

```cpp
admiral::plan_r2c<double> p(512);
std::vector<double> in(p.real_size());
std::vector<std::complex<double>> spec(p.cplx_size());
p.forward(in, spec);
p.inverse(spec, in);                    // divides by the real element count
```

`real_size()` and `cplx_size()` return the two buffer lengths; use them rather
than computing `n/2 + 1` yourself. One-shot forms exist too: the free
`forward(in, out, shape)` / `inverse(spec, out, shape)` overloads on a real
pointer plan, transform, and discard.

## DCT / DST

`plan_r2r` with `r2r_kind::{dct2, dct3, dst2, dst3}` (FFTW's REDFT10/01,
RODFT10/01). `forward()` is FFTW's unnormalized kind, `inverse()` its exact
inverse, so forward then inverse round-trips; pass `fct` for FFTW's 2N
convention. The batched form transforms `rows` contiguous lines at once; there
is no strided r2r, so N-D DCT means transposing to the innermost axis yourself.
[example](https://github.com/DiamonDinoia/admiral/blob/master/examples/r2r_dct.cpp)

```cpp
admiral::plan_r2r<double> dct2(n, admiral::r2r_kind::dct2);
dct2.forward(in, out);
dct2.inverse(out, in);

admiral::plan_r2r<double> batched(n, admiral::r2r_kind::dct2, rows);
```

## One axis of a tensor

`axis_plan` transforms a single axis, optionally over a rectangular sub-box;
chaining every axis equals `plan<T>(shape)`. [example](https://github.com/DiamonDinoia/admiral/blob/master/examples/axis_transform.cpp)

```cpp
admiral::axis_plan<double> ax0({rows, cols}, 0, /*forward=*/true);
ax0.execute(data.data(), {}, {});        // empty box = full extent
```

`execute_bands()` runs the same plan on two disjoint bands of the last
dimension in one call; the header documents when that beats two `execute()`
calls.

## Errors

Every failure the caller can cause derives from `admiral::error` and carries an
`error_code`, so handling never parses `what()`:

```cpp
try {
    admiral::plan<double> p(0);            // zero size
} catch (const admiral::error& e) {
    assert(e.code() == admiral::error_code::invalid_size);
}
```

`size_error` (`invalid_size`) covers zero extents, span/plan size mismatches,
and out-of-range axes; `unsupported_error` (`unsupported`) fires only on a
forced route the size cannot take; `internal_error` (`internal`) marks an
invariant break, never a bad argument. Allocation failure stays
`std::bad_alloc`. Span overloads validate; pointer overloads trust the caller.

## Options

`{.nthreads = N, .eff = .., .debug = ..}` on any plan or one-shot; see
[usage.md](usage.md#options) for the field table, and
[build-options.md](build-options.md) for the CMake and benchmark knobs.

## The other interfaces

C: [usage.md](usage.md#c) with [examples/c_api.c](https://github.com/DiamonDinoia/admiral/blob/master/examples/c_api.c).
FFTW: [usage.md](usage.md#fftw) with [examples/fftw_dropin.c](https://github.com/DiamonDinoia/admiral/blob/master/examples/fftw_dropin.c).
