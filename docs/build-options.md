# Build and runtime options
The defaults ship a fast library: `Release`, `-march=native`, fast math, threads, and plan-time measurement
all on. Change the defaults when packaging, cross-compiling, or debugging. Every option is a CMake cache
variable: pass `-DNAME=value` on the configure line.

## User-facing build options
| Option | What it does, and when to change it | Default |
|--------|-------------------------------------|---------|
| `ADM_BUILD_TESTS` | Builds the Catch2 test suite (see `test/`). Turn off for an install-only build | `ON` as top-level project |
| `ADM_BUILD_BENCHMARKS` | Builds `admiral_benchmark`, which fetches ducc0 and nanobench. Only needed to measure performance; see `benchmark/` | `ON` as top-level project |
| `ADM_BUILD_EXAMPLES` | Builds the runnable examples in `examples/` and registers them with ctest | `ON` as top-level project |
| `ADM_CXX_STANDARD` | The standard the library and its public headers compile at: `17`, `20`, or `23`. The two builds are not link-compatible: `admiral::span` is `std::span` at C++20 and a polyfill at C++17 | `20` |
| `ADM_ENABLE_THREADS` | `OFF` makes every plan serial regardless of `nthreads` and drops the threads library from the link. Turn off for a single-threaded environment | `ON` |
| `ADM_MEASURE` | Compiles in plan-time route measurement (`effort::automatic`/`measure`, and FFTW's non-`ESTIMATE` flags). `OFF` keeps the flags accepted but inert: every plan routes by the cost model, so plans become bitwise reproducible across runs | `ON` |
| `ADM_TARGET_ARCH` | The one `-march` the whole build uses. `native` is fastest on the build machine but the binaries may not run elsewhere; use `x86-64-v3` for portable binaries, `none` for the compiler baseline. Also the build-memory knob: one engine TU peaks near 12 GB at AVX-512 `native` versus ~2 GB at `x86-64-v2`. Do not also pass `-march` in `CMAKE_CXX_FLAGS` | `native` |
| `ADM_USE_FAST_MATH` | Adds `-ffast-math`: faster transforms, but relaxed IEEE semantics (reassociation, flushed denormals, no errno). Turn off when strict IEEE semantics matter | `ON` |
| `ADM_GRANULE_ADMIT` | Admits the AoS-granule leaf kernels into the codelet routes. `OFF` removes the granule arms at compile time (`if constexpr`), so the SoA fallback runs; the default build is byte-identical to a pre-knob tree | `ON` |
| `ADM_BENCH_FFTW` | Adds an FFTW reference arm to `admiral_benchmark`. Needs the system `fftw3` and `fftw3f`, found through pkg-config | `OFF` |
| `ADM_BENCH_THREADS` | Threads the benchmark reference libraries too (ducc0's pool, FFTW's `fftw3_threads` companions) | `OFF` |

## Development options
These options face maintainers. The seven presets in `CMakePresets.json` cover the common combinations
(`debug`, `asan`, `tsan`, `coverage`, `dev`, `release`, `relwithdebinfo`), and `scripts/validate.sh` sweeps
them. Every preset except `release` has a matching test preset; `release` builds no tests
(`ADM_BUILD_TESTS=OFF`, and `-DADM_BUILD_TESTS=ON` enables the full suite), so `ctest
--preset dev` is the test entry point. `dev` is the same optimized build with tests on and benchmarks off.

| Option | What it does | Default |
|--------|--------------|---------|
| `ADM_ENABLE_WARNINGS_AS_ERRORS` | `-Werror` profile on the library targets | `ON` |
| `ADM_SANITIZER` | One of `none`, `address`, `undefined`, `address+undefined`, `thread`, checked at configure time. Needs GCC or Clang; the `asan` preset pins Clang | `none` |
| `ADM_ENABLE_LTO` | Link-time optimization across the engine TUs | `OFF` |
| `ADM_ENABLE_PCH` | Precompiled headers for the library targets | `OFF` |
| `ADM_FIT_COST_MODEL` | Exposes the `admiral_cost_sweep` and `admiral_cost_model` targets that re-measure and regenerate the routing cost-model header | `OFF` |
| `ADM_COST_MODEL_DATA` | Directory of BASECOST receipts the `admiral_cost_model` refit reads | `${source_dir}/bench-results` |
| `ADM_COST_MODEL_TAG` | Name of the receipt the `admiral_cost_sweep` target writes. The default encodes compiler, arch, and flags | derived from the build |
| `ADM_COST_MODEL_OUT` | Header the refit writes. Point elsewhere to dry-run without dirtying the checkout | the tracked header in `include/` |

## Runtime options
Every plan and one-shot takes an optional `admiral::options` aggregate (`adm_options` by pointer in C). The
fields are `nthreads`, `eff`, `debug`, and `pin_threads`. [usage.md](usage.md) documents each. The library
reads no environment variables.
