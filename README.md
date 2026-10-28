# Admiral

[![CI](https://github.com/DiamonDinoia/admiral/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/DiamonDinoia/admiral/actions/workflows/ci.yml)
[![Docs](https://github.com/DiamonDinoia/admiral/actions/workflows/docs.yml/badge.svg?branch=master)](https://diamondinoia.com/admiral/)
[![codecov](https://codecov.io/gh/DiamonDinoia/admiral/graph/badge.svg)](https://codecov.io/gh/DiamonDinoia/admiral)

A C++20 FFT library, buildable at C++17: complex and real transforms, any size,
1-D and N-D, float, double or long double, with optional multithreading. One
compiled engine behind three interfaces: C++, C, and a drop-in FFTW subset.

## Why Admiral

- Fast on stock hardware. At 1 thread it beats FFTW_MEASURE by 1.34× (1-D) and
  1.43× (N-D), and ducc0 by 2.2-2.7× at 1-D, on a Xeon w5-3435X at
  `-march=native`.
- Any size, any rank. Compiled codelets up to radix 64, so primes and awkward
  composites route through the same engine as powers of two.
- Plans pick their own route. A fitted cost model at `effort::estimate`, a
  plan-time race at `automatic`. No wisdom files, no tuning step.
- Existing FFTW code drops in. Recompile against `<admiral/fftw3.h>` and keep
  the same calls, flags included.
- Threads stay serial below ~32k elements, where the pool would cost more than
  it saves. The `nthreads = 0` default ramps the pool with transform size.

## Quick start

One transform is one line:

```cpp
#include <admiral/admiral.hpp>

#include <complex>
#include <vector>

int main() {
    std::vector<std::complex<double>> x(1024, {1, 0});
    admiral::forward<double>(x, x);   // in place; the inverse divides by 1024

    admiral::plan<double> p(x.size());
    p.forward(x);   // same call, but a plan reuses its route
}
```

The one-shot plans and frees on every call. A `plan` prices the route once,
so it wins when a size repeats.

Build the library:

```bash
cmake --preset release           # benchmarks on, tests off; `dev` swaps that
cmake --build --preset release
ctest --test-dir build/release   # runs the examples; `ctest --preset dev` runs the test suite
```

[examples/quickstart.cpp](examples/quickstart.cpp) is this snippet plus a
round-trip check, built and run by ctest; [examples/](examples/) also covers
real transforms, DCT/DST, per-axis and strided transforms, C, and the FFTW
drop-in. [docs/cpp-api.md](docs/cpp-api.md) walks the C++ API;
<https://diamondinoia.com/admiral/> renders the docs.

The `release` preset also builds the benchmarks, which fetches ducc0 and
nanobench; `-DADM_BUILD_BENCHMARKS=OFF` skips them. Configure warns that the
default `-march=native` binaries may not run on other machines;
`ADM_TARGET_ARCH` in [docs/build-options.md](docs/build-options.md) changes it.

If ninja aborts with `fatal: unknown placeholder`, a site-set `NINJA_STATUS` is
incompatible with the local ninja: `unset NINJA_STATUS`.

Requirements: CMake 3.25+, Ninja (or Make), and a C++20 compiler. GCC 14+ or
Clang 19+ recommended; older C++20 compilers build and pass the tests but
generate worse code. `-DADM_CXX_STANDARD=17` builds the same library at C++17;
the `{.nthreads = 8}` form is C++20, which gcc and clang also accept at
C++17 as an extension, and other compilers need the fields in order. xsimd,
poet and snmalloc are fetched automatically and are the only dependencies of
the library itself.

## Warning

This project was not originally intended to be a library or something that others
would use. This was originally a collection of homeworks and kernels written
in asm, c and c++ with the sole purpose of learning how to write performing code.
Why FFT? Because FFTW and MKL are hard to beat, so the baseline to compare
against is meaningful. In the end, I asked Claude to see if by using `poet` and
`xsimd` it is possible to match the inline assembly and instantiate codelets up
to size 64. Claude also tuned my handcrafted cost model to decompose large
transforms and route the best kernel. I also asked Claude to refine the
interface into something usable, though for this I gave specifications so that
it resembles something I like. Is this the best FFT? Probably not. Do I claim I
have the fastest FFT out there? I don't think so. But it was a fun exercise and
since I found it useful I decided to allow others to see it. Am I proud of
everything you see in here? No. Any constructive feedback is welcome. I tried to
remove all AI slop from this project, but I do not like writing good
documentation, and sometimes AI parsed some rough notes in code comments, and
the code is likely more verbose than it should. If you have any good
suggestions, feel free to improve the project.

If it takes a long time to compile, just wait, it will compile eventually.
Use a recent compiler if you don't want to leave performance on the table.

## Performance

Measured on a Xeon w5-3435X (16 physical cores, AVX-512, `-march=native`).
Speedup is the reference's time over Admiral's, so above 1 Admiral is faster.
[benchmark/](benchmark/README.md) covers how to rerun these numbers.

Against ducc0 (`c4dda23`), over 44 1-D lengths and 11 N-D shapes per precision:

| | 1 thread | 16 threads |
|-|----------|------------|
| 1-D | 2.18× (f64), 2.70× (f32) | n/a* |
| 2-D | 1.12× (f64), 1.41× (f32) | 1.27× / 1.58× |
| 3-D | 1.60× (f64), 1.67× (f32) | 2.67× / 3.53× |

*1-D threading only engages past the byte floor (≥ ~32k elements); the 44-size
1-D grid tops at 32768, so a 16-thread cell there would measure nothing. Big
1-D transforms do thread and split wins with ducc0's threaded 1-D.

Against FFTW (`FFTW_MEASURE`, single thread, both precisions): 1.34× at 1-D,
1.43× at N-D.

Threading pays from about 32k elements up and reaches 9-12× on large shapes on
16 threads; below ~16k elements one thread beats sixteen. For rectangles the
innermost extent decides.

## Install and consume

```bash
cmake --install build/release --prefix /your/prefix
```

```cmake
find_package(admiral REQUIRED)   # needs CMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(app PRIVATE admiral::admiral)   # C: admiral::admiral_c; FFTW shim: admiral::fftw
```

To consume the checkout without installing, use `FetchContent` or
`add_subdirectory`; [docs/install.md](docs/install.md) shows both.

| Interface | Header | Link |
|-----------|--------|------|
| C++ | `<admiral/admiral.hpp>` | `admiral::admiral` or `admiral::admiral_static` |
| C | `<admiral/admiral.h>` | `admiral::admiral_c` or `admiral::admiral_c_static` |
| FFTW | `<admiral/fftw3.h>` | `admiral::fftw` or `admiral::fftw_static` |

The C++ header declares the engine instead of defining it, so it pulls no SIMD
dependency into your code, only small Admiral support headers and the standard
library. The engine is compiled, so the C++ API ships for
`std::complex<float>`, `std::complex<double>` and `std::complex<long double>`.
The long-double transform covers `plan`, `plan_r2c` and the one-shots;
`axis_plan`, `strides_plan` and `plan_r2r` stay float/double. Where the host
long double is the x87 80-bit format with FMA the lines run a double-double
engine on SIMD doubles; other targets run a scalar backend, because no SIMD ISA
has 80-bit lanes. The exported target compiles at whatever `ADM_CXX_STANDARD`
was configured with. The two builds are not link-compatible: every span in the
API is `admiral::span`, which is `std::span` at C++20 and a polyfill of the
same shape at C++17.

The shared libraries need nothing extra; they record their own dependency on
libstdc++, so a plain C project links `admiral::admiral_c` and runs. The static
archives are C++ behind a C API, so a C-only project must enable C++ to link
them:

```cmake
project(app C CXX)       # or enable_language(CXX) before find_package
```

## Usage

The C++ snippet in Quick start covers most needs. Data is contiguous row-major,
last axis fastest; forward is unscaled and inverse divides by the element count.
[docs/cpp-api.md](docs/cpp-api.md) covers the full C++ API and the options
aggregate (`nthreads`, `eff`, `debug`, `pin_threads`);
[docs/fftw.md](docs/fftw.md) covers what the FFTW shim does and does not
implement. Each interface header opens with the same contract in full.

## Build options

Every CMake option and the benchmark environment variables live in
[docs/build-options.md](docs/build-options.md). The defaults
ship a fast library; reach for the docs when packaging, cross-compiling, or
developing Admiral itself.

Run-time ISA dispatch is under development and is not on master yet:
`-DADM_DISPATCH=ON` compiles the engine once per ISA tier of the target
architecture, and each plan runs the best tier the CPU has, so one installed
library serves older and newer machines.

## Dependencies

All fetched by CPM.

- xsimd, poet and snmalloc do the SIMD, the compile-time unrolling and the
  engine's scratch allocator. They are the only dependencies of the library
  itself.
- Catch2 builds the tests, with `ADM_BUILD_TESTS`.
- ducc0 and nanobench are the benchmark reference and harness, with
  `ADM_BUILD_BENCHMARKS`.

## License

BSD 3-Clause Attribution License (SPDX `BSD-3-Clause-Attribution`), see
[LICENSE](LICENSE). Permissive, with an attribution clause: redistributions must
retain the acknowledgment naming the Admiral FFT library. See also
[NOTICE](NOTICE).

## Why the name

An admiral is a butterfly. The FFT decomposes one large transform into repeated
two-term combines, and the signal-flow diagram of each combine is called a
butterfly: two inputs enter, two scaled sums leave, and the crossing edges draw
a pair of wings. The small kernels in this library carry the same name. One
butterfly for the algorithm, one admiral for the library.
