# Build options

Pass each option as `-DNAME=value` on the CMake configure line.

| Option | Default | Values | Effect |
|--------|---------|--------|--------|
| `ADM_BUILD_TESTS` | `ON` as top-level project | `ON`, `OFF` | Builds the Catch2 test suite. |
| `ADM_BUILD_BENCHMARKS` | `ON` as top-level project | `ON`, `OFF` | Builds `admiral_benchmark`. Downloads ducc0 and nanobench. |
| `ADM_BUILD_EXAMPLES` | `ON` as top-level project | `ON`, `OFF` | Builds the programs in `examples/` and registers them with ctest. |
| `ADM_CXX_STANDARD` | `20` | `17`, `20`, `23` | C++ standard of the library and its public headers. The builds are not link-compatible. |
| `ADM_ENABLE_THREADS` | `ON` | `ON`, `OFF` | `OFF` makes every plan serial and drops the threads library from the link. |
| `ADM_MEASURE` | `ON` | `ON`, `OFF` | Compiles in plan-time route measurement. `OFF` makes `effort::automatic` and `effort::measure` inert. |
| `ADM_TARGET_ARCH` | `native` | `native`, `none`, an architecture or CPU name | Sets the target CPU. `none` adds no flag on every compiler. `native` on a cross build adds no flag. GCC and Clang get `-mcpu=` in two cases. One case is a processor name that starts with `arm`, `powerpc` or `ppc`. The other is `arm64` or `aarch64` on Apple. They get `-march=` otherwise, including Linux aarch64. MSVC on x86 and AMD64 gets `/arch:AVX2` for any other value. Other combinations, such as MSVC on ARM64, get no flag. Use `x86-64-v3` for portable binaries. |
| `ADM_ENABLE_WARNINGS_AS_ERRORS` | `ON` | `ON`, `OFF` | Adds `-Werror` (`/WX` on MSVC) to the compile of the admiral targets. Applies as a subproject too, because the default is `ON` there. Set `OFF` before `add_subdirectory` to disable it. Consumer targets are not affected. |
| `ADM_ENABLE_LTO` | `OFF` | `ON`, `OFF` | Turns on interprocedural optimization for Release, RelWithDebInfo and MinSizeRel builds when the compiler supports it. Not for shipping: the installed archives link only with the exact compiler. |
| `ADM_ENABLE_PCH` | `OFF` | `ON`, `OFF` | Precompiles `xsimd/xsimd.hpp` and `poet/poet.hpp` for the engine and codelet targets. |
| `ADM_SANITIZER` | `none` | `none`, `address`, `undefined`, `address+undefined`, `thread` | Adds `-fsanitize=` to compile and link. Needs GCC or Clang. Other values and compilers stop the configure. A sanitizer build also trims the codelet catalog to sizes up to 16, plus 45, and ignores `ADM_CODELET_MAX_N` and `ADM_CODELET_EXTRA_SIZES`. |
| `ADM_USE_FAST_MATH` | `ON` | `ON`, `OFF` | Adds `-ffast-math`. `OFF` keeps strict IEEE semantics. |
| `ADM_GRANULE_ADMIT` | `ON` | `ON`, `OFF` | `OFF` removes the AoS-granule leaf kernels at compile time. |
| `ADM_CODELET_MIN_N` | `2` | integer, at least 2 | Smallest size that gets a compiled straight-line codelet. |
| `ADM_CODELET_MAX_N` | `64` | integer, at least `ADM_CODELET_MIN_N` | Largest size that gets a compiled straight-line codelet. |
| `ADM_CODELET_EXCLUDE_SIZES` | empty | list separated by `;` | Sizes that the generated codelet catalog leaves out. |
| `ADM_CODELET_EXTRA_SIZES` | `120;65;85;143;100;360` | list separated by `;` | Sizes that the catalog adds beyond `ADM_CODELET_MIN_N` to `ADM_CODELET_MAX_N`. |
| `ADM_EXTRA_C_FLAGS` | not set | flag string | Adds these flags to every C compile of the admiral targets. |
| `ADM_EXTRA_CXX_FLAGS` | not set | flag string | Adds these flags to every C++ compile of the admiral targets. |
| `ADM_BENCH_FFTW` | `OFF` | `ON`, `OFF` | Adds an FFTW reference to `admiral_benchmark`. Requires system `fftw3` and `fftw3f`. |
| `ADM_BENCH_THREADS` | `OFF` | `ON`, `OFF` | Threads the benchmark reference libraries (ducc0, FFTW). |

## Development

| Option | Default | Values | Effect |
|--------|---------|--------|--------|
| `ADM_PIN_1D` | `OFF` | `ON`, `OFF` | Pins floating-point contraction and reassociation in the 1-D engine instantiation TUs. |
| `ADM_PIN_PLAN_ESTIMATE` | `OFF` | `ON`, `OFF` | Resolves every plan route as `effort::estimate`, even at `measure` and `automatic`. |
| `ADM_ENABLE_CLANG_TIDY` | `OFF` | `ON`, `OFF` | Runs clang-tidy on the admiral sources during the build. |
| `ADM_ENABLE_CPPCHECK` | `OFF` | `ON`, `OFF` | Runs cppcheck on the admiral sources during the build. |
| `ADM_FIT_COST_MODEL` | `OFF` | `ON`, `OFF` | Adds the targets that re-measure and refit the routing cost model. The next three options need it. |
| `ADM_COST_MODEL_DATA` | `bench-results/` in the source tree | path | Directory of receipts that the fit reads. |
| `ADM_COST_MODEL_OUT` | `include/admiral/detail/base_cost_model.hpp` | path | Header that `admiral_cost_model` regenerates. |
| `ADM_COST_MODEL_TAG` | `<compiler><major>_<processor>_<hash>`, the hash of the C++ flags and the probed CPU model | string | Name of the receipt that this build writes. |
