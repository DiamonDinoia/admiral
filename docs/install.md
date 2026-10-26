# Install and use

Admiral needs CMake 3.25 or newer and a C++20 compiler: GCC 13, Clang 18, Apple Clang or MSVC.
The first configure downloads xsimd, poet and snmalloc, so it needs network access.

## Install to a prefix

Set `PREFIX` to the install directory.
The commands skip the tests, the benchmarks and the examples.
`-DCMAKE_INSTALL_LIBDIR=lib` puts the FFTW shim in `$PREFIX/lib`.

```{literalinclude} ../scripts/install_prefix.sh
:start-after: [doc:install-prefix]
:end-before: [doc:install-prefix-end]
```

`CMAKE_BUILD_PARALLEL_LEVEL` sets the number of parallel compiles.
One GCC 14 compile of `inst_col_f_inv.cpp` at `x86-64-v4` in a Release build peaked at 3.4 GiB (3497 MiB).
Keep the number of compiles x 3.4 GiB below 75% of the memory.

A native GCC or Clang build targets the build CPU.
For a binary that runs on other machines, set `ADM_TARGET_ARCH`.
See [Build options](build-options.md).

## Use from CMake

To use an installed prefix, put `$PREFIX` in `CMAKE_PREFIX_PATH`.

```{literalinclude} ../test/package/CMakeLists.txt
:start-after: [doc:find-package]
:end-before: [doc:find-package-end]
:dedent:
:language: cmake
```

To use a source tree at `./admiral`, add it as a subdirectory.

```{literalinclude} ../test/package/CMakeLists.txt
:start-after: [doc:add-subdirectory]
:end-before: [doc:add-subdirectory-end]
:dedent:
:language: cmake
```

To fetch the source with FetchContent, declare it.

```{literalinclude} ../test/package/CMakeLists.txt
:start-after: [doc:fetchcontent]
:end-before: [doc:fetchcontent-end]
:dedent:
:language: cmake
```

To fetch the source with CPM, add the package.

```{literalinclude} ../test/package/CMakeLists.txt
:start-after: [doc:cpm]
:end-before: [doc:cpm-end]
:dedent:
:language: cmake
```

All four routes give the same targets: `admiral::admiral` (C++), `admiral::admiral_c` (C) and `admiral::fftw` (FFTW shim).
Each target has a static twin with the suffix `_static`.
To link a static twin from C, use `project(x C CXX)`.

## C++ example

Include `<admiral/admiral.hpp>`.
Here `x` is a `std::vector<std::complex<double>>`.

```{literalinclude} ../examples/quickstart.cpp
:start-after: [doc:quickstart]
:end-before: [doc:quickstart-end]
:dedent:
:language: cpp
```

The C API is in `include/admiral/admiral.h`.
`examples/c_api.c` shows its use.
The [C++ API](cpp-api.md) page lists the types.
