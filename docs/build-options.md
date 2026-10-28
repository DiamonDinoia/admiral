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

Default: `ON` as top-level project. Builds `admiral_benchmark`. Downloads ducc0 and nanobench. Where pkg-config finds system `fftw3` and `fftw3f`, the benchmark also builds the `--fftw-ab` FFTW reference, threaded like ducc0's.

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

### `ADM_USE_FAST_MATH`

Default: `ON`. Adds `-ffast-math`. `OFF` keeps strict IEEE semantics.

```{literalinclude} ../scripts/options.sh
:start-after: [doc:opt-ADM_USE_FAST_MATH]
:end-before: [doc:opt-ADM_USE_FAST_MATH-end]
```

## Advanced options

These options are for work on the library itself.

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
