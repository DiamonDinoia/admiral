# Admiral

A high-performance C++20 FFT library. Complex and real transforms of any size
(mixed-radix codelets, iterative DIF, four-step, Rader, Bluestein, Good-Thomas),
1-D and N-D, with optional multithreading. Measured on AVX2: **1.8× (f64) / 2.3×
(f32) vs ducc0** and **~1.3× vs FFTW** across the standard size sweep.

Named for the red admiral — the fast, migratory butterfly — because the FFT's
core operation *is* the butterfly.

Three interfaces over one engine:

- **C++** — `admiral::plan<T>` / `admiral::plan_r2c<T>` (`include/admiral/admiral.hpp`)
- **C** — `admiral/admiral.h`, linked as `admiral::admiral_c`
- **FFTW** — drop-in `fftw3.h` subset, linked as `admiral::fftw` (`include/admiral/fftw3.h`)

## Requirements

- CMake 3.20+
- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022)
- xsimd and poet (header-only, fetched automatically via CPM)
- Ninja (recommended) or Make

## Building

### Quick Start with Presets

```bash
# Configure
cmake --preset debug

# Build
cmake --build --preset debug

# Test
ctest --preset debug
```

### Manual Configuration

```bash
# Configure
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=ON

# Build
cmake --build .

# Test
ctest --output-on-failure
```

## Usage

Normalization is explicit. Forward is unscaled and inverse divides by N by
default (so forward∘inverse is the identity). Pass an optional `fct` to set the
output scale directly — `fct = 1` gives an unnormalized inverse.

### C++

```cpp
#include <admiral/admiral.hpp>

std::vector<std::complex<double>> x(1024);
admiral::plan<double> p(x.size());    // reusable; N-D via {n0, n1, ...}
p.forward(x);                 // in place
p.inverse(x);                 // divides by N

admiral::plan<double> pnd({64, 64});  // 2-D
pnd.forward(data);

p.inverse(x, 1.0);            // unnormalized (fct = 1)

// Real transforms:
admiral::plan_r2c<double> r({256});
r.forward(real_in, spectrum);         // spectrum has r.cplx_size() elements
r.inverse(spectrum, real_out);
```

### C

The C API follows the FINUFFT convention: double precision is the base name,
single precision gets an `f` on the library token (`adm_` → `admf_`).

```c
#include <admiral/admiral.h>

adm_plan plan;                            // admf_plan usage is identical
adm_plan_both(&plan, 1024);               // admf_plan_both for float
adm_plan_execute_forward(plan, data);     // data: adm_complex*
adm_plan_execute_inverse(plan, data);
adm_plan_destroy(plan);
```

### FFTW compatibility

Existing FFTW code builds unchanged against `admiral/fftw3.h` + `admiral::fftw`:

```c
#include <admiral/fftw3.h>

fftw_complex* in  = fftw_alloc_complex(N);
fftw_complex* out = fftw_alloc_complex(N);
fftw_plan p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
fftw_free(in); fftw_free(out);
```

Both directions are unnormalized, matching FFTW. Real transforms and the
guru/wisdom interfaces are out of scope — use the native `admiral::plan_r2c<T>`.

## Installing and consuming

```bash
cmake --install build --prefix /your/prefix
```

Then from another project:

```cmake
find_package(admiral REQUIRED)
target_link_libraries(app PRIVATE admiral::fftw)   # or admiral::admiral_c
```

`find_package` exports the two self-contained compiled libraries
`admiral::admiral_c` and `admiral::fftw` (no runtime dependencies beyond
threads). The header-only C++ `admiral::admiral` target additionally needs xsimd
and poet on the include path, so it is available via `add_subdirectory`, not the
installed package.

> **Note:** on a CPU with AVX-512 the engine is currently built and tuned for
> AVX2 (W=8/4). Configure with `-DADM_USE_NATIVE_ARCH=OFF
> -DADM_EXTRA_CXX_FLAGS=-march=alderlake` on such hosts.

## Available CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `ADM_BUILD_TESTS` | Build unit tests | `ON` when main project |
| `ADM_BUILD_BENCHMARKS` | Build benchmarks | `ON` when main project |
| `ADM_ENABLE_WARNINGS_AS_ERRORS` | Treat warnings as errors | `ON` |
| `ADM_ENABLE_ASAN` | Enable AddressSanitizer | `OFF` |
| `BUILD_SHARED_LIBS` | Build shared libraries | `OFF` |

## Build Configurations

- **debug**: Debug build with tests and warnings
- **debug-asan**: Debug build with AddressSanitizer enabled
- **release**: Optimized release build
- **relwithdebinfo**: Release with debug symbols

## Dependencies

Managed automatically via CPM.cmake:

- **xsimd**, **poet**: SIMD and compile-time unroll — the only dependencies of
  the library itself (header-only)
- **Catch2**: unit tests (only when `ADM_BUILD_TESTS`)
- **ducc0**, **nanobench**: benchmark reference and harness (only when
  `ADM_BUILD_BENCHMARKS`); not dependencies of the library

## CPM Cache

To speed up rebuilds, set the CPM source cache:

```bash
export CPM_SOURCE_CACHE=$HOME/.cache/CPM
```

## Running Tests

```bash
# After building with tests enabled
ctest --test-dir build --output-on-failure
```

## Running Benchmarks

```bash
# After building with benchmarks enabled
./benchmark/admiral_benchmark
```

## Project Structure

```
admiral/
├── include/admiral/  # Public headers
├── src/              # Implementation
├── test/             # Unit tests
├── benchmark/        # Performance benchmarks
└── cmake/            # CMake modules
```

## Optimization archive

The full record of the WS4–WS8 performance workstream — every per-candidate
A/B receipt, factor sweep, profile, and the accumulated findings/traps — lives
on the [`archive`](../../tree/archive) branch (`ARCHIVE.md` there is the index).
`master` keeps only the durable summaries: the loss ledger, current ratio
tables, acceptance receipts, and the WS8 P0 census under `bench-results/`.

## License

BSD 3-Clause Attribution License (SPDX: `BSD-3-Clause-Attribution`) — see [LICENSE](LICENSE).
Permissive, with an attribution clause: redistributions must retain the acknowledgment
naming the Admiral FFT library.
