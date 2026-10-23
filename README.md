# Admiral
[![CI](https://github.com/DiamonDinoia/admiral/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/DiamonDinoia/admiral/actions/workflows/ci.yml)
[![Docs](https://github.com/DiamonDinoia/admiral/actions/workflows/docs.yml/badge.svg?branch=master)](https://diamondinoia.com/admiral/)
[![codecov](https://codecov.io/gh/DiamonDinoia/admiral/graph/badge.svg)](https://codecov.io/gh/DiamonDinoia/admiral)

Admiral is a C++20 FFT library: complex and real transforms of any size, in 1-D and N-D, in float, double and
long double, with optional multithreading. One compiled engine serves three interfaces: C++, C, and a drop-in
FFTW subset. `-DADM_CXX_STANDARD=17` builds the same library at C++17.

## Quick start
```bash
cmake --preset release           # benchmarks on, tests off; `dev` swaps that
cmake --build --preset release
ctest --test-dir build/release   # runs the examples; `ctest --preset dev` runs the test suite
```

```cpp
#include <admiral/admiral.hpp>

std::vector<std::complex<double>> x(1024);

admiral::plan<double> p(x.size());
p.forward(x);                  // in place
p.inverse(x);                  // divides by 1024

admiral::plan<double> p2d({64, 64});              // N-D; r2c and DCT/DST plans too
admiral::plan<double> p8(1 << 20, {.nthreads = 8});

admiral::forward<double>(x, x);   // one-shot, no plan
```

This snippet is [examples/quickstart.cpp](examples/quickstart.cpp), built and run by ctest.
[examples/](examples/) also covers real transforms, DCT/DST, per-axis and strided transforms, C, and the FFTW
drop-in. [docs/cpp-api.md](docs/cpp-api.md) describes the C++ API.

Requirements: CMake 3.25+, Ninja (or Make), and a C++20 compiler. GCC 14+ or Clang 19+ is recommended. Older
C++20 compilers build and pass the tests but generate slower code. CMake fetches xsimd and poet automatically.
They are the library's only dependencies (Catch2, ducc0 and nanobench serve tests and benchmarks). The
`release` preset builds no tests; `-DADM_BUILD_TESTS=ON` enables the full suite. gcc and
clang accept the `{.nthreads = 8}` form at C++17 as an extension. Other compilers need the fields in order.

**Warning**: the default `ADM_TARGET_ARCH=native` passes `-march=native`, so the binaries can fail to run on other
machines. Set `-DADM_TARGET_ARCH=x86-64-v3` for portable binaries. **Note**: if ninja aborts with `fatal:
unknown placeholder`, run `unset NINJA_STATUS`, because a site-set value is incompatible with the local ninja.

## Performance
Measured on a Xeon w5-3435X (16 physical cores, AVX-512, `-march=native`). The speedup is the reference time
over Admiral's, so above 1 Admiral is faster. [benchmark/README.md](benchmark/README.md) documents the method
and the rerun commands. Against ducc0 (`c4dda23`), over 61 1-D lengths and 27 N-D shapes per precision:

| | 1 thread | 16 threads |
|-|----------|------------|
| 1-D | 2.18× (f64), 2.70× (f32) | n/a* |
| 2-D | 1.12× (f64), 1.41× (f32) | 1.27× / 1.58× |
| 3-D | 1.60× (f64), 1.67× (f32) | 2.67× / 3.53× |

\* The 44-size 1-D grid ends at 32768 elements, below the 1-D threading threshold. Against FFTW
(`FFTW_MEASURE`, single thread, both precisions): 1.34× at 1-D and 1.43× at N-D. Threading pays from about 32k
elements up and reaches 9-12× on large shapes at 16 threads; below ~16k elements one thread beats sixteen, and
for rectangles the innermost extent decides. Plans pick their own route, a cost model at `effort::estimate` and
a plan-time race at `automatic`: no wisdom files, no tuning step.

## Install and use
```bash
cmake --install build/release --prefix /your/prefix
```

```cmake
find_package(admiral REQUIRED)   # needs CMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(app PRIVATE admiral::admiral)
```

The CMake targets are in [docs/install.md](docs/install.md), which also covers `FetchContent`, `add_subdirectory`
and the static-link rules. [docs/usage.md](docs/usage.md) covers the data layout, the scaling, the runtime
options, the C API and the FFTW shim. [docs/](docs/README.md) is the docs index.

The engine is compiled, so the C++ API creates plans for `std::complex<float>`, `std::complex<double>` and
`std::complex<long double>` only. The long-double transform covers `plan`, `plan_r2c` and the one-shots. In an
x86-64 build with FMA (`x86-64-v3` and newer), a line of length 8 or more (n/2 for an even-length r2c line)
runs on SIMD double-double lanes and converts from and to long double only at its ends. The tests compare it
with a float128 DFT: c2c, r2c and c2r at lengths 8 to 1024, and 12 bins at length 65537. Forward and at the
default inverse scale, every 1-D output above 2^-30 of the largest output is within 0.5 + 2^-7 ulp. N-D
transforms round to long double between axes. An input more than about 2^1000 below the largest input of its
line reads as zero. A non-finite input makes the whole line NaN. Shorter lines, other targets and other
`long double` formats run a scalar backend. `axis_plan`, `strides_plan` and `plan_r2r` stay float/double. The
C++20 and C++17 builds are not link-compatible: `admiral::span` is `std::span` at C++20 and a polyfill at
C++17. The exported target compiles at the configured `ADM_CXX_STANDARD`.

## License
BSD 3-Clause Attribution License (SPDX `BSD-3-Clause-Attribution`), see [LICENSE](LICENSE). Redistributions
must keep the acknowledgment that names the Admiral FFT library.
