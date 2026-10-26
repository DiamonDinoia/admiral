# admiral

Admiral is an FFT library with a C++20 API. It also compiles at C++17 (`-DADM_CXX_STANDARD=17`).
It has a C API. It also has a library that exports the FFTW3 complex-to-complex API. A program links
that library. On Linux, a program can also use it through `LD_PRELOAD`.

## Build and install

Use CMake 3.25 or newer, and a C++20 compiler: GCC 13, Clang 18, Apple Clang or MSVC. The first
configure downloads xsimd, poet and snmalloc, so it needs network access. The flags below skip the
tests and the benchmarks, which download more. The examples download nothing.
A native GCC or Clang build targets the build CPU by default. For a binary that runs on other
machines, add `-DADM_TARGET_ARCH=<cpu>`, for example `x86-64-v3`. For other compilers and cross
builds, read [docs/build-options.md](docs/build-options.md).

Set `PREFIX` to the install directory. `-DCMAKE_INSTALL_LIBDIR=lib` puts the FFTW shim in
`$PREFIX/lib`.

<!-- from: scripts/install_prefix.sh#install-prefix -->
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF -DADM_BUILD_EXAMPLES=OFF
cmake --build build --config Release
cmake --install build --prefix "$PREFIX" --config Release
```

`CMAKE_BUILD_PARALLEL_LEVEL` sets the number of parallel compiles. One GCC compile at the default
native arch can use 3.4 GiB. Keep the number of compiles x 3.4 GiB below 75% of the memory.

## Use from CMake

To use an installed prefix, put `$PREFIX` in `CMAKE_PREFIX_PATH`.

<!-- from: test/package/CMakeLists.txt#find-package -->
```cmake
find_package(admiral REQUIRED CONFIG)
```

To use a source tree at `./admiral`, add it as a subdirectory.

<!-- from: test/package/CMakeLists.txt#add-subdirectory -->
```cmake
add_subdirectory(admiral)
```

To fetch the source with FetchContent, declare it.

<!-- from: test/package/CMakeLists.txt#fetchcontent -->
```cmake
include(FetchContent)
FetchContent_Declare(admiral GIT_REPOSITORY https://github.com/DiamonDinoia/admiral.git GIT_TAG master)
FetchContent_MakeAvailable(admiral)
```

To fetch the source with CPM, add the package.

<!-- from: test/package/CMakeLists.txt#cpm -->
```cmake
CPMAddPackage("gh:DiamonDinoia/admiral#master")
```

All routes give the same targets: `admiral::admiral` (C++), `admiral::admiral_c` (C) and
`admiral::fftw` (FFTW shim). Each target has a static twin with the suffix `_static`. To link a static twin from C, use
`project(x C CXX)`.

## C++ example

Include `<admiral/admiral.hpp>`. Here `x` is a `std::vector<std::complex<double>>`.

<!-- from: examples/quickstart.cpp#quickstart -->
```cpp
admiral::plan<double> p(x.size());
p.forward(x);
p.inverse(x);
```

The C API is in `include/admiral/admiral.h`. `examples/c_api.c` shows its use.

## FFTW shim

Link `admiral::fftw`. On Linux, `LD_PRELOAD` can also inject the shim into an FFTW program.

<!-- from: test/preload/run.sh#ld-preload -->
```sh
LD_PRELOAD=$PREFIX/lib/libadmiral_fftw.so ./app
```

The shim has the complex-to-complex planners. A call with a plan that another FFTW made goes to the
first other FFTW that the shim finds. On Windows, or when the shim finds no other FFTW, such a call
prints a line to stderr and aborts. `include/admiral/fftw3.h` gives the search order and the other
limits.

## Reference

The hosted documentation is at <http://diamondinoia.com/admiral/>.

- [docs/cpp-api.md](docs/cpp-api.md): the C++ API.
- [docs/build-options.md](docs/build-options.md): the CMake options.

## License

BSD 3-Clause Attribution License, see [LICENSE](LICENSE).
