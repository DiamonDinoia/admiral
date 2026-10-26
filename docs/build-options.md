# Build options

Give each option as `-DNAME=value` on the CMake configure line.
An option takes `ON` or `OFF` unless its description lists other values.
CI runs each example from `scripts/options.sh`. Each example configures the project and checks the effect.

## Library options

### `ADM_BUILD_TESTS`

Default: `ON` as top-level project. Builds the Catch2 test suite.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_BUILD_TESTS]
:end-before: [doc:opt-ADM_BUILD_TESTS-end]
```

### `ADM_BUILD_BENCHMARKS`

Default: `ON` as top-level project. Builds `admiral_benchmark`. Downloads ducc0 and nanobench.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_BUILD_BENCHMARKS]
:end-before: [doc:opt-ADM_BUILD_BENCHMARKS-end]
```

### `ADM_BUILD_EXAMPLES`

Default: `ON` as top-level project. Builds the programs in `examples/` and registers them with ctest.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_BUILD_EXAMPLES]
:end-before: [doc:opt-ADM_BUILD_EXAMPLES-end]
```

### `ADM_CXX_STANDARD`

Default: `20`. C++ standard of the library and its public headers. The builds are not link-compatible. Values: `17`, `20`, `23`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_CXX_STANDARD]
:end-before: [doc:opt-ADM_CXX_STANDARD-end]
```

### `ADM_ENABLE_THREADS`

Default: `ON`. `OFF` makes every plan serial and drops the threads library from the link.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_THREADS]
:end-before: [doc:opt-ADM_ENABLE_THREADS-end]
```

### `ADM_MEASURE`

Default: `ON`. Compiles in plan-time route measurement. `OFF` makes `effort::automatic` and `effort::measure` inert.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_MEASURE]
:end-before: [doc:opt-ADM_MEASURE-end]
```

### `ADM_TARGET_ARCH`

Default: `native`. Sets the target CPU. Values: `native`, `none`, an architecture or CPU name. The example uses an x86-64 name. See the rules below.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_TARGET_ARCH]
:end-before: [doc:opt-ADM_TARGET_ARCH-end]
```

The flag follows these rules.

- `none` adds no flag on any compiler. `native` on a cross build adds no flag.
- GCC and Clang read `CMAKE_SYSTEM_PROCESSOR` to choose the flag. The flag gets the option value unchanged.
- The flag is `-mcpu=` when `CMAKE_SYSTEM_PROCESSOR` starts with `arm`, `powerpc` or `ppc`, and for `aarch64` on Apple.
- The flag is `-march=` for all other processors, including `aarch64` on Linux. With `native`, `-mtune=native` is also added.
- MSVC gets `/arch:AVX2` when `CMAKE_SYSTEM_PROCESSOR` is x86 or AMD64. Other MSVC combinations, such as ARM64, get no flag.

### `ADM_ENABLE_WARNINGS_AS_ERRORS`

Default: `ON`, also as a subproject. Adds `-Werror` (`/WX` on MSVC) to the admiral targets. Set `OFF` before `add_subdirectory`. Consumer targets keep their own flags.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_WARNINGS_AS_ERRORS]
:end-before: [doc:opt-ADM_ENABLE_WARNINGS_AS_ERRORS-end]
```

### `ADM_ENABLE_LTO`

Default: `OFF`. Sets `CMAKE_INTERPROCEDURAL_OPTIMIZATION` for Release, RelWithDebInfo and MinSizeRel when `check_ipo_supported` succeeds. With `OFF`, this project sets it for no build type. GCC writes GIMPLE into the installed archives, so only the same compiler can link them. Use `OFF` for a shipped install.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_LTO]
:end-before: [doc:opt-ADM_ENABLE_LTO-end]
```

### `ADM_ENABLE_PCH`

Default: `OFF`. Precompiles `xsimd/xsimd.hpp` and `poet/poet.hpp` for the engine and codelet targets.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_PCH]
:end-before: [doc:opt-ADM_ENABLE_PCH-end]
```

### `ADM_SANITIZER`

Default: `none`. Adds `-fsanitize=` to compile and link. Requires GCC or Clang. Values: `none`, `address`, `undefined`, `address+undefined`, `thread`. Any other value or compiler stops the configure. A sanitizer build trims the codelet catalog to sizes up to 16, plus 45. It ignores `ADM_CODELET_MAX_N` and `ADM_CODELET_EXTRA_SIZES`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_SANITIZER]
:end-before: [doc:opt-ADM_SANITIZER-end]
```

### `ADM_USE_FAST_MATH`

Default: `ON`. Adds `-ffast-math`. `OFF` keeps strict IEEE semantics.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_USE_FAST_MATH]
:end-before: [doc:opt-ADM_USE_FAST_MATH-end]
```

### `ADM_GRANULE_ADMIT`

Default: `ON`. `OFF` removes the AoS-granule leaf kernels at compile time.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_GRANULE_ADMIT]
:end-before: [doc:opt-ADM_GRANULE_ADMIT-end]
```

### `ADM_CODELET_MIN_N`

Default: `2`. Smallest size that gets a compiled straight-line codelet. Values: integer, at least 2.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_CODELET_MIN_N]
:end-before: [doc:opt-ADM_CODELET_MIN_N-end]
```

### `ADM_CODELET_MAX_N`

Default: `64`. Largest size that gets a compiled straight-line codelet. Values: integer, at least `ADM_CODELET_MIN_N`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_CODELET_MAX_N]
:end-before: [doc:opt-ADM_CODELET_MAX_N-end]
```

### `ADM_CODELET_EXCLUDE_SIZES`

Default: empty. Sizes that the generated codelet catalog leaves out. Values: list separated by `;`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_CODELET_EXCLUDE_SIZES]
:end-before: [doc:opt-ADM_CODELET_EXCLUDE_SIZES-end]
```

### `ADM_CODELET_EXTRA_SIZES`

Default: `120;65;85;143;100;360`. Sizes that the catalog adds beyond `ADM_CODELET_MIN_N` to `ADM_CODELET_MAX_N`. Values: list separated by `;`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_CODELET_EXTRA_SIZES]
:end-before: [doc:opt-ADM_CODELET_EXTRA_SIZES-end]
```

### `ADM_EXTRA_C_FLAGS`

Default: not set. Adds these flags to the C compile options of the admiral targets. No library source is C code, so no library compile uses them. Values: flag string.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_EXTRA_C_FLAGS]
:end-before: [doc:opt-ADM_EXTRA_C_FLAGS-end]
```

### `ADM_EXTRA_CXX_FLAGS`

Default: not set. Adds these flags to every C++ compile of the admiral targets. Values: flag string.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_EXTRA_CXX_FLAGS]
:end-before: [doc:opt-ADM_EXTRA_CXX_FLAGS-end]
```

### `ADM_BENCH_FFTW`

Default: `OFF`. Adds an FFTW reference to `admiral_benchmark`. Requires system `fftw3` and `fftw3f`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_BENCH_FFTW]
:end-before: [doc:opt-ADM_BENCH_FFTW-end]
```

### `ADM_BENCH_THREADS`

Default: `OFF`. Threads the benchmark reference libraries (ducc0, FFTW).

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_BENCH_THREADS]
:end-before: [doc:opt-ADM_BENCH_THREADS-end]
```

## Development options

These options are for work on the library itself.

### `ADM_PIN_1D`

Default: `OFF`. Pins floating-point contraction and reassociation in the 1-D engine instantiation TUs.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_PIN_1D]
:end-before: [doc:opt-ADM_PIN_1D-end]
```

### `ADM_PIN_PLAN_ESTIMATE`

Default: `OFF`. Resolves every plan route as `effort::estimate`, even at `measure` and `automatic`.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_PIN_PLAN_ESTIMATE]
:end-before: [doc:opt-ADM_PIN_PLAN_ESTIMATE-end]
```

### `ADM_ENABLE_CLANG_TIDY`

Default: `OFF`. Runs clang-tidy on the admiral sources during the build. The configure stops if `clang-tidy` is missing.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_CLANG_TIDY]
:end-before: [doc:opt-ADM_ENABLE_CLANG_TIDY-end]
```

### `ADM_ENABLE_CPPCHECK`

Default: `OFF`. Runs cppcheck on the admiral sources during the build. The configure stops if `cppcheck` is missing.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_ENABLE_CPPCHECK]
:end-before: [doc:opt-ADM_ENABLE_CPPCHECK-end]
```

### `ADM_FIT_COST_MODEL`

Default: `OFF`. Adds the targets that re-measure and refit the routing cost model. The next three options require it.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_FIT_COST_MODEL]
:end-before: [doc:opt-ADM_FIT_COST_MODEL-end]
```

### `ADM_COST_MODEL_DATA`

Default: `bench-results/` in the source tree. Directory of receipts that the fit reads. Values: path.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_COST_MODEL_DATA]
:end-before: [doc:opt-ADM_COST_MODEL_DATA-end]
```

### `ADM_COST_MODEL_OUT`

Default: `include/admiral/detail/base_cost_model.hpp`. Header that `admiral_cost_model` regenerates. Values: path.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_COST_MODEL_OUT]
:end-before: [doc:opt-ADM_COST_MODEL_OUT-end]
```

### `ADM_COST_MODEL_TAG`

Default: `<compiler><major>_<processor>_<hash>`. Name of the receipt that this build writes. The hash covers the C++ flags and the probed CPU model. Values: string.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_COST_MODEL_TAG]
:end-before: [doc:opt-ADM_COST_MODEL_TAG-end]
```
