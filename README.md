# Admiral

A C++20 FFT library. Complex and real transforms, any size, 1-D and N-D, with
optional multithreading. One engine behind three interfaces: C++, C, and a
drop-in FFTW subset.

## Performance

Measured on a Xeon w5-3435X (16 physical cores, AVX-512, `-march=native`).
Speedup is their time over ours, so above 1 means Admiral is faster.

Against **ducc0** (`c4dda23`), over 61 1-D lengths and 27 N-D shapes per precision:

| | 1 thread | 16 threads |
|-|----------|------------|
| 1-D | 1.99× (f64), 2.78× (f32) | — |
| 2-D | 1.24× (f64), 1.40× (f32) | 1.68× / 1.82× |
| 3-D | 1.36× (f64), 1.61× (f32) | 5.96× / 5.09× |

Against **FFTW** with `FFTW_MEASURE`, single thread, 19 1-D lengths and 9 N-D
shapes, both precisions pooled: 1.04× at 1-D, 1.35× at N-D. (Comparisons that use
`FFTW_ESTIMATE` leave FFTW on an untuned plan and would read 1.42× and 2.15× —
flattering, but not what an FFTW user gets.)

**Threaded scaling is bimodal.** From 1024² up, and 3-D from 64³ up, you get 8–16×
on 16 cores. 512² and 60³ land near 7×. At 256² and below the pool costs more than
it saves and 16 threads are *slower* than one. For rectangles the innermost extent
decides, not the aspect ratio: 64×4096 scales 11×, its transpose 4096×64 only
1.0–1.3×. Use one thread for small shapes.

## Interfaces

| Interface | Header | Link |
|-----------|--------|------|
| C++ | `<admiral/admiral.hpp>` | `admiral::admiral` or `admiral::admiral_static` |
| C | `<admiral/admiral.h>` | `admiral::admiral_c` or `admiral::admiral_c_static` |
| FFTW | `<admiral/fftw3.h>` | `admiral::fftw` or `admiral::fftw_static` |

Each header opens with its own contract — layout, sign, scaling, threads. Read the
one you use; you do not need the other two.

## Requirements

- CMake 3.25+, and Ninja or Make
- A C++20 compiler. **GCC 14+ or Clang 19+.** Older ones that still do C++20 build
  and pass the tests but generate worse codelets; configure warns below the floor.
- xsimd and poet, fetched automatically. Header-only, and the only dependencies of
  the library itself.

## Build

```bash
cmake --preset dev            # optimized + tests
cmake --build --preset dev
ctest --preset dev
```

| Preset | What it is for |
|--------|----------------|
| `dev` | Release + tests. The everyday edit-build-test loop. |
| `release` | Release + benchmarks. Ship or measure with this. |
| `debug` | Unoptimized, with tests. |
| `asan` | Debug + AddressSanitizer + UBSan (pins `clang++`). |
| `tsan` | Debug + ThreadSanitizer. |
| `coverage` | Clang source-based coverage, fast-math off. |
| `relwithdebinfo` | Optimized with symbols, for profiling. |

Or configure by hand:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DADM_BUILD_TESTS=ON
cmake --build build && ctest --test-dir build --output-on-failure
```

Two notes that save time. If ninja aborts with `fatal: unknown placeholder`
before doing anything, a site-set `NINJA_STATUS` is incompatible with your ninja:
`unset NINJA_STATUS`. And compile memory tracks SIMD width, not optimization level
— a single engine TU peaks near 1.8 GB at `x86-64-v2` and near 12 GB at `native`,
so cut `ADM_TARGET_ARCH` before anything else if a build will not fit. The
instrumented presets pin `none` for that reason.

## Usage

Forward is unscaled. Inverse divides by the element count, so forward then inverse
returns the input. Pass an explicit `fct` to set the output scale instead —
`fct = 1` gives FFTW's unnormalized inverse.

### C++

```cpp
#include <admiral/admiral.hpp>
#include <complex>
#include <vector>

std::vector<std::complex<double>> x(1024);

admiral::plan<double> p(x.size());        // second arg is admiral::options, see below
p.forward(x);                  // in place
p.inverse(x);                  // divides by 1024
p.inverse(x, 1.0);             // unnormalized

admiral::plan<double> p8(x.size(), {.nthreads = 8});   // 0 = one per physical core

admiral::plan<double> p2d({64, 64});
p2d.forward(data);

admiral::plan_r2c<double> r(256);        // 256 REAL elements; {64, 64} for 2-D
r.forward(real_in, spectrum);  // spectrum holds r.cplx_size() elements
r.inverse(spectrum, real_out);

admiral::plan_r2r<double> c(256, admiral::r2r_kind::dct2);   // DCT-II, 1-D
c.forward(x, y);               // FFTW's unnormalized REDFT10
c.inverse(y, x);               // its exact inverse, so this round-trips
```

`admiral::forward(in, out)` / `admiral::inverse(in, out)` do the same thing without
a plan, for a size you transform once (pass one span twice to go in place). With
`std::span` arguments the precision deduces; with plain containers, name it:
`admiral::forward<double>(x, x)`.

`admiral::plan_r2r<T>` does DCT-II/III and DST-II/III (`r2r_kind::dct2`, `dct3`,
`dst2`, `dst3` = FFTW's REDFT10, REDFT01, RODFT10, RODFT01) over `rows` contiguous
lines of length N, out of place. It costs one real FFT of the same length N, not of
the 2N even extension. `forward()` is FFTW's unnormalized kind and `inverse()` is its
exact inverse — FFTW's own type-2/type-3 pair differs by 2N in that position, so pass
`fct` if you want its convention. 1-D only: an N-D separable DCT needs the caller to
transpose between axes.

`admiral::axis_plan<double>` transforms one axis of a fixed-shape tensor in place over
a rectangular sub-box, for data that arrives in strips rather than whole:

```cpp
admiral::axis_plan<double> ax({64, 64, 64}, /*axis=*/1, /*forward=*/true);
ax.execute(data, lo, hi);      // half-open box; empty spans mean the full extent
```

The transformed axis must be whole; the others may be bands. `execute_bands` does two
disjoint bands of the last dimension in one call, which is cheaper than two calls when
both fit one SIMD batch.

Span overloads check the element count and throw `std::invalid_argument` if it does not
match; construction throws `std::invalid_argument` for a bad shape and `std::bad_alloc`
if scratch cannot be allocated. Pointer overloads trust the caller.

### Options

Every plan and every one-shot takes the same optional `admiral::options`, so a call
site names what it sets and nothing else. The C API mirrors it as `adm_options`,
passed by pointer, with `NULL` meaning the defaults.

```cpp
admiral::plan<double> p(1024, {.nthreads = 8, .eff = admiral::effort::automatic});
admiral::forward<double>(x, y, {.debug = 1});
```

| field | default | meaning |
|---|---|---|
| `nthreads` | `1` | `1` serial, `0` one per physical core |
| `eff` | `estimate` | how hard construction works to pick a route (below) |
| `debug` | `0` | trace verbosity to stderr: `1` what ran, `2` also the shape |

`debug` is off by default and costs one compare per execute when off — the printers are
cold and never inlined, so nothing formats on the fast path.

### Choosing the effort

`eff` is a hint, like FFTW's `ESTIMATE`/`MEASURE`:

| `admiral::effort` | How it routes | Plan build, f64 N=1080 / 122220 |
|---|---|---|
| `estimate` (default) | fitted cost model only — fast, and the only reproducible one | 10 µs / 0.31 ms |
| `automatic` | races routes, factorizations and pass orders at plan time | 0.41 ms / 0.12 s |
| `measure` | the same race; exists for the FFTW flag mapping | same as `automatic` |

**For the best execution speed, use `automatic`.** The cost model ranks
factorizations by an additive per-pass cost that barely separates pass *order*, yet
order is worth up to 1.25× (`8-16-16-8` beats `16-8-8-16` at N=16384). Only
measurement sees that.

The price is a fixed *count* of trial executions, not a wall-clock deadline: over
26 sizes × both precisions the whole plan build costs 33–379 executions of the plan
it builds. So it repays a 1% win after ~10⁴ transforms of the same
plan, ~10³ at 10%. For a plan used once, stay on `estimate` — it is discarded
before it can repay. (`FFTW_MEASURE` costs 3500–47000 executions, two to
three orders of magnitude more.)

A measuring effort is not bitwise reproducible: it elects from timings, so the route
depends on the machine and on what else is running. That is why `estimate` is the
default. `-DADM_MEASURE=OFF` makes both measuring efforts inert.

### C

Double precision keeps the plain name; single precision adds an `f` to the prefix
(`adm_` → `admf_`).

```c
#include <admiral/admiral.h>

adm_plan plan;
adm_plan_1d(&plan, 1024, NULL);           // 1024 points, default options
adm_plan_execute_forward(plan, data);     // data: adm_complex[1024]
adm_plan_execute_inverse(plan, data);
adm_plan_destroy(plan);
```

Every call returns an `adm_status`; `ADM_SUCCESS` is 0 and the result is nodiscard.
N-D is `adm_plan_nd(&plan, shape, ndim, opts)`. To set any option, pass an
`adm_options` instead of `NULL` — note a zeroed struct is not the default, since
`nthreads = 0` asks for one worker per core:

```c
const adm_options opts = {.nthreads = 8, .eff = ADM_EFFORT_AUTOMATIC, .debug = 0};
adm_plan_1d(&plan, 1024, &opts);
```

### FFTW

Existing FFTW code compiles unchanged against `<admiral/fftw3.h>`:

```c
#include <admiral/fftw3.h>

fftw_complex* in  = fftw_alloc_complex(N);
fftw_complex* out = fftw_alloc_complex(N);
fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
fftw_free(in); fftw_free(out);
```

Both directions stay unnormalized, as in FFTW. Covered: `fftw_plan_dft` and its
1d/2d/3d forms, `fftw_execute` and `fftw_execute_dft`, destroy, `fftw_cleanup`, the
alloc helpers, and the `fftwf_` mirror. Not covered: real and r2r transforms (use
`admiral::plan_r2c<T>` / `admiral::plan_r2r<T>`), the guru and split interfaces, wisdom,
`fftw_plan_with_nthreads`. Nothing degrades silently — a call that is not there does
not compile.

Migrating: shim plans are single-threaded and, unlike FFTW, one plan must not
execute from two threads at once — make one plan per executing thread (creating
plans concurrently is fine). `fftw_execute_dft` accepts any alignment and either
in/out-of-place character. `FFTW_MEASURE` never touches your arrays,
`FFTW_WISDOM_ONLY` is ignored (every plan is live), and an out-of-range `sign` means
backward, as in FFTW. Planning flags map to efforts: `FFTW_ESTIMATE` → `estimate`,
`FFTW_PATIENT`/`FFTW_EXHAUSTIVE` → `measure`, everything else — including
`FFTW_MEASURE`, which *is* zero in FFTW — → `automatic`.

## Install and consume

```bash
cmake --install build --prefix /your/prefix   # preset users: build/<preset>
```

```cmake
find_package(admiral REQUIRED)   # needs CMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(app PRIVATE admiral::fftw)   # or admiral::admiral_c
```

The prefix gets three headers and six libraries — `admiral::admiral`,
`admiral::admiral_c`, `admiral::fftw` and their `_static` archives — and nothing
else. All six are machine code; the runtime footprint is the C++ runtime, libm and
the system threads library.

The C++ header declares the engine instead of defining it, so it includes only the
standard library: no xsimd, no poet, no Admiral internals. That keeps it cheap to
include. Compiled means the C++ API ships for `std::complex<float>` and
`std::complex<double>` only, and needs C++20 at the call site, which the exported
target requires for you.

The archives are C++ behind a C API, so a C-only project must enable C++:

```cmake
project(app C CXX)       # or enable_language(CXX) before find_package
target_link_libraries(app PRIVATE admiral::admiral_c_static)
```

Without that the C driver links the archive and the C++ runtime symbols come out
undefined. The shared libraries need nothing — they record their own dependency on
libstdc++.

## Options

| Option | What it does | Default |
|--------|--------------|---------|
| `ADM_BUILD_TESTS` | Build the unit tests | `ON` as top-level project |
| `ADM_BUILD_BENCHMARKS` | Build the benchmarks | `ON` as top-level project |
| `ADM_ENABLE_THREADS` | Multithreaded execution; `OFF` makes every plan serial and drops the Threads dependency | `ON` |
| `ADM_MEASURE` | Compile in plan-time route measurement (`effort::automatic`/`measure`, `FFTW_MEASURE`/`FFTW_PATIENT`). `OFF` makes both inert | `ON` |
| `ADM_ENABLE_WARNINGS_AS_ERRORS` | `-Werror` | `ON` |
| `ADM_SANITIZER` | `none`, `address`, `undefined`, `address+undefined`, `thread`. Anything else is a configure error, so a typo cannot leave you unsanitized | `none` |
| `ADM_TARGET_ARCH` | The one `-march` the build uses: `native`, `none`, or any value the compiler takes. Do not also pass `-march` in `CMAKE_CXX_FLAGS` | `native` |
| `ADM_USE_FAST_MATH` | `-ffast-math`. Turn it off if you need strict IEEE semantics | `ON` |
| `ADM_ENABLE_LTO` | Link-time optimization. No measured speedup, and GCC then writes GIMPLE instead of machine code into the installed `.a` | `OFF` |
| `ADM_ENABLE_PCH` | Precompiled headers for the library targets. Cuts the codelet TUs ~9×, but a `.gch` stale against its flags fails the build until you delete the build directory | `OFF` |
| `ADM_CODELET_MIN_N` / `ADM_CODELET_MAX_N` | Range of straight-line codelet sizes compiled in. Raising the ceiling trades compile time and code size for speed at those sizes; any sanitizer build caps it at 16 | `2` / `64` |
| `ADM_CODELET_EXTRA_SIZES` / `ADM_CODELET_EXCLUDE_SIZES` | Add or drop individual codelet sizes. Extras bypass `ADM_CODELET_MAX_N` | `120` / empty |
| `ADM_BENCH_FFTW` / `ADM_BENCH_THREADS` | Race FFTW in the benchmark (needs system fftw3/fftw3f) / build its multithreaded comparisons | `OFF` |
| `ADM_FIT_COST_MODEL` | Expose the targets that re-measure and refit the routing cost model | `OFF` |
| `ADM_EXTRA_C_FLAGS` / `ADM_EXTRA_CXX_FLAGS` | Extra flags appended to the library targets | empty |

## Tuning the routing cost model

Which algorithm a size routes to comes from a small fitted model: 35 coefficients,
one sparse fit per algorithm form, pooled over every machine that has been swept.
There is no per-machine correction table, so an unmeasured build is not a
second-class one — it routes off the same coefficients as a measured one, a few
percent off the best route on average, and a size those coefficients misprice is
recovered at plan time by `effort::automatic`.

Sweeping this machine folds it into the pooled fit (~2 min):

```bash
cmake -B build -DADM_FIT_COST_MODEL=ON
cmake --build build --target admiral_cost_sweep   # writes a receipt
cmake --build build --target admiral_cost_model   # refits the header
```

Receipts land in `ADM_COST_MODEL_DATA` (default `<src>/bench-results`) and are
self-describing, so several compilers and targets accumulate into one header. The
refit writes the tracked header — it is a code generator, so it dirties the
checkout; point `ADM_COST_MODEL_OUT` elsewhere to dry-run. It needs nothing beyond
the C++ compiler: the fitter is standard-library-only, so this also runs on machines
without Python.

## Dependencies

All fetched by CPM. Set `CPM_SOURCE_CACHE=$HOME/.cache/CPM` to share downloads
between build directories.

- **xsimd**, **poet** — SIMD and compile-time unrolling. Header-only, and the only
  dependencies of the library itself.
- **Catch2** — tests, with `ADM_BUILD_TESTS`.
- **ducc0**, **nanobench** — benchmark reference and harness, with
  `ADM_BUILD_BENCHMARKS`. Not dependencies of the library.

## License

BSD 3-Clause Attribution License (SPDX: `BSD-3-Clause-Attribution`) — see
[LICENSE](LICENSE). Permissive, with an attribution clause: redistributions must
retain the acknowledgment naming the Admiral FFT library.
