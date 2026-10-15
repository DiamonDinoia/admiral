# The C++ API
The whole C++ API lives in `<admiral/admiral.hpp>`. Keep a plan when a size repeats. Use a one-shot when it
does not. Every snippet has a runnable counterpart under [../examples/](../examples/).

## Complex plans
```cpp
admiral::plan<double> p(1024);          // 1-D
admiral::plan<double> q({64, 64, 8});   // any rank, last axis fastest
p.forward(x);                           // in place
p.inverse(x);                           // divides by the element count
```

Forward is unscaled and inverse divides by the element count, so a round trip returns the input. Pass `fct` or
the `(src, dst)` overloads for out-of-place work. `p.size()` returns the element count the plan expects. Span
overloads throw on a size mismatch. Pointer overloads trust the caller
([example](../examples/quickstart.cpp)).

One-shots do the same without a plan and always route with `effort::estimate`:

```cpp
admiral::forward<double>(x, x);
admiral::forward<double>(y.data(), {64, 64});   // N-D, in place
```

## Real transforms
`plan_r2c` transforms real input to half the spectrum, in FFTW's packed layout: the last axis shrinks to `n/2
+ 1`. `real_size()` and `cplx_size()` return the two buffer lengths
([example](../examples/real_transforms.cpp)).

```cpp
admiral::plan_r2c<double> p(512);
std::vector<double> in(p.real_size());
std::vector<std::complex<double>> spec(p.cplx_size());
p.forward(in, spec);
p.inverse(spec, in);                    // divides by the real element count
```

The free `forward(in, out, shape)` and `inverse(spec, out, shape)` overloads are the one-shot forms: they
plan, transform, and discard.

## DCT / DST
`plan_r2r` takes one of `r2r_kind::{dct2, dct3, dst2, dst3}` (FFTW's REDFT10, REDFT01, RODFT10, RODFT01).
`forward()` applies the unnormalized kind and `inverse()` its exact inverse scaled by 1/(2N), so a round trip
returns the input. Pass `fct` for FFTW's 2N scaling. The `rows` form transforms `rows` contiguous lines at
once. There is no strided r2r, so an N-D DCT requires a transpose to the innermost axis first
([example](../examples/r2r_dct.cpp)).

```cpp
admiral::plan_r2r<double> dct2(n, admiral::r2r_kind::dct2);
dct2.forward(in, out);
dct2.inverse(out, in);

admiral::plan_r2r<double> batched(n, admiral::r2r_kind::dct2, rows);
```

## One axis of a tensor
`axis_plan` transforms a single axis, optionally inside a rectangular sub-box. Chaining a plan per axis equals
`plan<T>(shape)`. `execute_bands()` runs the same plan on two disjoint bands of the last dimension in one call.
The header states when that beats two `execute()` calls ([example](../examples/axis_transform.cpp)).

```cpp
admiral::axis_plan<double> ax0({rows, cols}, 0, /*forward=*/true);
ax0.execute(data.data(), {}, {});        // empty bounds = full extent
```

## Strided lines
`strides_plan` transforms `nbatch` lines of `len` complex elements out of place, with FFTW's `plan_many(rank =
1)` geometry: a stride separates the elements of one transform, and a dist separates consecutive transforms.
Source and destination carry independent pairs, so one plan can gather, scatter, or transpose.

```cpp
// Transforms every column of a row-major (rows x cols) matrix and writes column-major.
admiral::strides_plan<double> columns(/*len=*/rows, /*nbatch=*/cols,
                                      /*in_stride=*/cols, /*in_dist=*/1,
                                      /*out_stride=*/1,   /*out_dist=*/rows);
columns.forward(src.data(), dst.data());
```

The input geometry alone picks the kernel, so a given source produces bit-identical results for every output
layout. A unit input dist runs the batched SIMD column pass, a unit input stride runs one contiguous transform
per line, and any other pair moves groups of columns through scratch. At `src == dst` the transform runs in
place and the two layouts must match. `float` and `double` only, because the strided kernels are SIMD
([example](../examples/strided_lines.cpp)).

## Errors
Every failure the caller can cause derives from `admiral::error` and carries an `error_code`:

```cpp
try {
    admiral::plan<double> p(0);            // zero size
} catch (const admiral::error& e) {
    assert(e.code() == admiral::error_code::invalid_size);
}
```

- `size_error` (`invalid_size`): zero extents, span/plan size mismatches, out-of-range axes.
- `unsupported_error` (`unsupported`): a forced route the size cannot take.
- `internal_error` (`internal`): an invariant break, never a bad argument.
- Allocation failure stays `std::bad_alloc`.

Any plan or one-shot takes `{.nthreads = N, .eff = .., .debug = ..}`. [usage.md](usage.md#options) has the
field table and the [C](usage.md#c) and [FFTW](usage.md#fftw) interfaces, and
[build-options.md](build-options.md) has the CMake options.
