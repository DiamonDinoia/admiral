# Build options

This page lists the CMake options of the library.
Pass each option as `-DNAME=value` on the CMake configure line.
An option takes `ON` or `OFF` unless its description lists other values.

| Option | Default | Effect |
|--------|---------|--------|
| `ADM_BUILD_TESTS` | `ON` as top-level project | Builds the Catch2 test suite. |
| `ADM_BUILD_BENCHMARKS` | `ON` as top-level project | Builds `admiral_benchmark`. Downloads ducc0 and nanobench. |
| `ADM_BUILD_EXAMPLES` | `ON` as top-level project | Builds the programs in `examples/` and registers them with ctest. |
| `ADM_CXX_STANDARD` | `20` | C++ standard of the library and its public headers. The builds are not link-compatible. Values: `17`, `20`, `23`. |
| `ADM_ENABLE_THREADS` | `ON` | `OFF` makes every plan serial and drops the threads library from the link. |
| `ADM_MEASURE` | `ON` | Compiles in plan-time route measurement. `OFF` makes `effort::automatic` and `effort::measure` inert. |
| `ADM_TARGET_ARCH` | `native` | Sets the target CPU. Values: `native`, `none`, an architecture or CPU name. See the rules below the table. |
| `ADM_ENABLE_WARNINGS_AS_ERRORS` | `ON`, also as a subproject | Adds `-Werror` (`/WX` on MSVC) to the admiral targets. Set `OFF` before `add_subdirectory`. Consumer targets keep their own flags. |
| `ADM_ENABLE_LTO` | `OFF` | Sets `CMAKE_INTERPROCEDURAL_OPTIMIZATION` for Release, RelWithDebInfo and MinSizeRel when `check_ipo_supported` succeeds. With `OFF`, this project sets it for no build type. GCC writes GIMPLE into the installed archives, so only the same compiler can link them. Use `OFF` for a shipped install. |
| `ADM_ENABLE_PCH` | `OFF` | Precompiles `xsimd/xsimd.hpp` and `poet/poet.hpp` for the engine and codelet targets. |
| `ADM_SANITIZER` | `none` | Adds `-fsanitize=` to compile and link. Requires GCC or Clang. Values: `none`, `address`, `undefined`, `address+undefined`, `thread`. Any other value or compiler stops the configure. See the section below the table. |
| `ADM_USE_FAST_MATH` | `ON` | Adds `-ffast-math`. `OFF` keeps strict IEEE semantics. |
| `ADM_GRANULE_ADMIT` | `ON` | `OFF` removes the AoS-granule leaf kernels at compile time. |
| `ADM_CODELET_MIN_N` | `2` | Smallest size that gets a compiled straight-line codelet. Values: integer, at least 2. |
| `ADM_CODELET_MAX_N` | `64` | Largest size that gets a compiled straight-line codelet. Values: integer, at least `ADM_CODELET_MIN_N`. |
| `ADM_CODELET_EXCLUDE_SIZES` | empty | Sizes that the generated codelet catalog leaves out. Values: list separated by `;`. |
| `ADM_CODELET_EXTRA_SIZES` | `120;65;85;143;100;360` | Sizes that the catalog adds beyond `ADM_CODELET_MIN_N` to `ADM_CODELET_MAX_N`. Values: list separated by `;`. |
| `ADM_EXTRA_C_FLAGS` | not set | Adds these flags to every C compile of the admiral targets. Values: flag string. |
| `ADM_EXTRA_CXX_FLAGS` | not set | Adds these flags to every C++ compile of the admiral targets. Values: flag string. |
| `ADM_BENCH_FFTW` | `OFF` | Adds an FFTW reference to `admiral_benchmark`. Requires system `fftw3` and `fftw3f`. |
| `ADM_BENCH_THREADS` | `OFF` | Threads the benchmark reference libraries (ducc0, FFTW). |

## Rules for `ADM_TARGET_ARCH`

- `none` adds no flag on any compiler. `native` on a cross build adds no flag.
- GCC and Clang read `CMAKE_SYSTEM_PROCESSOR` to choose the flag. The flag gets the option value unchanged.
- The flag is `-mcpu=` when `CMAKE_SYSTEM_PROCESSOR` starts with `arm`, `powerpc` or `ppc`, and for `aarch64` on Apple.
- The flag is `-march=` for all other processors, including `aarch64` on Linux. With `native`, `-mtune=native` is also added.
- MSVC gets `/arch:AVX2` when `CMAKE_SYSTEM_PROCESSOR` is x86 or AMD64. Other MSVC combinations, such as ARM64, get no flag.

## Sanitizer builds

A sanitizer build trims the codelet catalog to sizes up to 16, plus 45. It ignores `ADM_CODELET_MAX_N` and `ADM_CODELET_EXTRA_SIZES`.

## Development

These options are for work on the library itself.

| Option | Default | Effect |
|--------|---------|--------|
| `ADM_PIN_1D` | `OFF` | Pins floating-point contraction and reassociation in the 1-D engine instantiation TUs. |
| `ADM_PIN_PLAN_ESTIMATE` | `OFF` | Resolves every plan route as `effort::estimate`, even at `measure` and `automatic`. |
| `ADM_ENABLE_CLANG_TIDY` | `OFF` | Runs clang-tidy on the admiral sources during the build. |
| `ADM_ENABLE_CPPCHECK` | `OFF` | Runs cppcheck on the admiral sources during the build. |
| `ADM_FIT_COST_MODEL` | `OFF` | Adds the targets that re-measure and refit the routing cost model. The next three options require it. |
| `ADM_COST_MODEL_DATA` | `bench-results/` in the source tree | Directory of receipts that the fit reads. Values: path. |
| `ADM_COST_MODEL_OUT` | `include/admiral/detail/base_cost_model.hpp` | Header that `admiral_cost_model` regenerates. Values: path. |
| `ADM_COST_MODEL_TAG` | `<compiler><major>_<processor>_<hash>` | Name of the receipt that this build writes. The hash covers the C++ flags and the probed CPU model. Values: string. |
