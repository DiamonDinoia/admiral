# Admiral

A C++20 FFT library: complex and real transforms, any size, 1-D and N-D, float
or double, with optional multithreading. One compiled engine behind three
interfaces: C++, C, and a drop-in FFTW subset.

## Performance

Measured on a Xeon w5-3435X (16 physical cores, AVX-512, `-march=native`).
Speedup is their time over Admiral's, so above 1 Admiral is faster.
[benchmark/](benchmark/README.md) covers how these numbers are measured and how to
rerun them.

Against ducc0 (`c4dda23`), over 61 1-D lengths and 27 N-D shapes per precision:

| | 1 thread | 16 threads |
|-|----------|------------|
| 1-D | 1.99× (f64), 2.78× (f32) | n/a |
| 2-D | 1.24× (f64), 1.40× (f32) | 1.68× / 1.82× |
| 3-D | 1.36× (f64), 1.61× (f32) | 5.96× / 5.09× |

Against FFTW (`FFTW_MEASURE`, single thread, both precisions): 1.04× at 1-D,
1.35× at N-D.

Threading pays from about 32k elements upward: 2-4× at 192²/32³, 5-7× at
512²/64³, and 9-11× from 1024²/128³ up, on 16 cores. Below ~25k elements there
is no gain, so the default `nthreads = 0` heuristic keeps those transforms
serial and ramps the pool with size. For rectangles the innermost extent
matters: 64×4096 scales 6.9×, its transpose 4096×64 only 2.6-5.3×.

## Requirements

- CMake 3.25+, and Ninja or Make
- A C++20 compiler; GCC 14+ or Clang 19+ recommended. Older C++20 compilers
  build and pass the tests but generate worse code
- xsimd and poet, fetched automatically; header-only, and the only dependencies
  of the library itself

## Build

```bash
cmake --preset release        # or `dev` to include the tests
cmake --build --preset release
ctest --preset dev            # under the dev preset
```

Or configure by hand:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

If ninja aborts with `fatal: unknown placeholder`, a site-set `NINJA_STATUS` is
incompatible with the local ninja: `unset NINJA_STATUS`.

## Install and consume

```bash
cmake --install build/release --prefix /your/prefix
```

```cmake
find_package(admiral REQUIRED)   # needs CMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(app PRIVATE admiral::fftw)   # or admiral::admiral_c
```

| Interface | Header | Link |
|-----------|--------|------|
| C++ | `<admiral/admiral.hpp>` | `admiral::admiral` or `admiral::admiral_static` |
| C | `<admiral/admiral.h>` | `admiral::admiral_c` or `admiral::admiral_c_static` |
| FFTW | `<admiral/fftw3.h>` | `admiral::fftw` or `admiral::fftw_static` |

The C++ header declares the engine instead of defining it, so it includes only
the standard library. The engine is compiled, so the C++ API ships for
`std::complex<float>` and `std::complex<double>` only. The exported target
requires C++20 at the call site.

The archives are C++ behind a C API, so a C-only project must enable C++:

```cmake
project(app C CXX)       # or enable_language(CXX) before find_package
```

The shared libraries need nothing. They record their own dependency on
libstdc++.

## Usage

Common to all three interfaces: data is contiguous row-major, last axis fastest.
Forward uses exp(-2πikn/N) and is unscaled; inverse divides by the element
count, so a round trip returns the input (`fct` overrides the scale on the C++
interface). Each header above opens with its full contract: layout, scaling,
errors, threading. The examples below are enough to start.

### C++

```cpp
#include <admiral/admiral.hpp>

std::vector<std::complex<double>> x(1024);

admiral::plan<double> p(x.size());
p.forward(x);                  // in place
p.inverse(x);                  // divides by 1024

admiral::plan<double> p8(x.size(), {.nthreads = 8});   // default 0 = auto
admiral::plan<double> p2d({64, 64});                   // N-D; r2c/DCT too, see header

admiral::forward<double>(x, x);   // one-shot, no plan. Plain containers need the
                                  // precision named; std::span arguments deduce it.
```

### Options

Every plan and one-shot takes an optional `admiral::options` aggregate, so a
call site names what it sets and nothing else. The C API mirrors it as
`adm_options`, passed by pointer, where `NULL` means the defaults.

| field | default | what it does |
|-------|---------|--------------|
| `nthreads` | `0` | worker threads owned by the plan. `0` is auto: a size-aware heuristic, serial for small transforms, ramping to one per physical core at 1024²/64³ scale. `1` forces serial, `n` forces `n` |
| `eff` | `estimate` | how hard construction works to pick a route. `estimate` uses the fitted cost model, so it is fast and reproducible. `automatic` also times the model's top candidates; pick it when one plan serves many transforms. `measure` is the same race, kept for the FFTW flag mapping |
| `debug` | `0` | stderr trace per execute: `0` silent, `1` what ran, `2` adds the shape, `3` adds the cost-model ranking |

`automatic` and `measure` elect from timings, so the picked route depends on the
machine and the load; both are inert under `-DADM_MEASURE=OFF`. The one-shot
functions ignore `eff` and always route with `estimate`, since a discarded plan
cannot repay a plan-time race.

### C

```c
#include <admiral/admiral.h>

adm_plan plan;
adm_plan_1d(&plan, 1024, NULL);           // default options; N-D: adm_plan_nd
adm_plan_execute_forward(plan, data);     // data: adm_complex[1024], in place
adm_plan_execute_inverse(plan, data);
adm_plan_destroy(plan);
```

Double precision keeps the plain name, single precision prefixes `admf_`. Every
call returns an `adm_status` (`ADM_SUCCESS` is 0, nodiscard). Pass a
`const adm_options*` instead of `NULL` to set options. A zeroed struct means
the defaults, so partial initializers are safe.

### FFTW

Existing FFTW code compiles unchanged against `<admiral/fftw3.h>`:

```c
fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```

Covered: `fftw_plan_dft` and its 1d/2d/3d forms, `fftw_execute` and
`fftw_execute_dft`, destroy, the alloc helpers, `fftw_cleanup`, and the `fftwf_`
mirror. Not covered: real/r2r transforms, guru and split interfaces, wisdom,
`fftw_plan_with_nthreads`. An uncovered call does not compile rather than
degrade. `FFTW_ESTIMATE` maps to `estimate`; every other flag, `FFTW_MEASURE`
included, maps to the plan-time race. Shim plans are single-threaded, so make
one plan per executing thread.

## Build options

| Option | What it does, and when to change it | Default |
|--------|-------------------------------------|---------|
| `ADM_BUILD_TESTS` | Builds the Catch2 test suite (see `test/`). On only when Admiral is the top-level project; turn off for an install-only build | `ON` as top-level project |
| `ADM_BUILD_BENCHMARKS` | Builds `admiral_benchmark`, which fetches ducc0 and nanobench. Only needed to measure performance; see `benchmark/` | `ON` as top-level project |
| `ADM_ENABLE_THREADS` | `OFF` makes every plan serial regardless of `nthreads` and drops the system threads library from the link. Turn off for a single-threaded environment | `ON` |
| `ADM_MEASURE` | Compiles in plan-time route measurement (`effort::automatic`/`measure`, and FFTW's non-`ESTIMATE` flags). `OFF` leaves them accepted but inert: every plan routes by the cost model, so plans become bitwise reproducible across runs | `ON` |
| `ADM_TARGET_ARCH` | The one `-march` the whole build uses. `native` is fastest on the build machine but the binaries may not run elsewhere; use `x86-64-v3` for portable binaries, `none` for the compiler baseline. Also the build-memory knob: one engine TU peaks near 12 GB at AVX-512 `native` versus ~2 GB at `x86-64-v2`. Do not also pass `-march` in `CMAKE_CXX_FLAGS` | `native` |
| `ADM_USE_FAST_MATH` | Adds `-ffast-math`: faster transforms, but relaxed IEEE (reassociation, flushed denormals, no errno). Turn off when strict IEEE semantics matter | `ON` |

The CMake sources document the remaining knobs (sanitizers, codelet catalog, LTO,
PCH, the cost-model fit targets) where they declare them. Those knobs face
development, not use.

## Dependencies

All fetched by CPM.

- xsimd and poet do the SIMD and the compile-time unrolling. They are the only
  dependencies of the library itself.
- Catch2 builds the tests, with `ADM_BUILD_TESTS`.
- ducc0 and nanobench are the benchmark reference and harness, with
  `ADM_BUILD_BENCHMARKS`.

## License

BSD 3-Clause Attribution License (SPDX `BSD-3-Clause-Attribution`), see
[LICENSE](LICENSE). Permissive, with an attribution clause: redistributions must
retain the acknowledgment naming the Admiral FFT library.
