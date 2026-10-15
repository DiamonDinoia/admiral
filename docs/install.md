# Installation
Requirements: CMake 3.25+, Ninja (or Make), and a C++20 compiler. GCC 14+ or Clang 19+ is recommended. Older
C++20 compilers build and pass the tests but generate slower code. `-DADM_CXX_STANDARD=17` builds the same
library at C++17. CMake fetches xsimd and poet automatically. They are the only dependencies of the library
itself.

## Build and install
```bash
cmake --preset release
cmake --build --preset release
cmake --install build/release --prefix /your/prefix
```

For an install-only build (no tests, benchmarks, or examples) with portable binaries, configure without a
preset:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF -DADM_BUILD_EXAMPLES=OFF \
      -DADM_TARGET_ARCH=x86-64-v3
```

`ADM_TARGET_ARCH` defaults to `native`: the fastest binaries for the build machine, but they may not run
elsewhere ([build-options.md](build-options.md) lists every option).

## Consume the installed package
```cmake
find_package(admiral REQUIRED)   # needs CMAKE_PREFIX_PATH=/your/prefix
target_link_libraries(app PRIVATE admiral::admiral)
```

| Interface | Header | Link |
|-----------|--------|------|
| C++ | `<admiral/admiral.hpp>` | `admiral::admiral` or `admiral::admiral_static` |
| C | `<admiral/admiral.h>` | `admiral::admiral_c` or `admiral::admiral_c_static` |
| FFTW | `<admiral/fftw3.h>` | `admiral::fftw` or `admiral::fftw_static` |

The exported targets compile at the configured `ADM_CXX_STANDARD`. The C++20 and C++17 builds are not
link-compatible: `admiral::span` is `std::span` at C++20 and a polyfill at C++17. The shared libraries record
their own dependency on libstdc++, so a plain C project links `admiral::admiral_c` and runs. The static
archives are C++ behind a C API, so a C-only project must enable C++ to link them:

```cmake
project(app C CXX)       # or enable_language(CXX) before find_package
```

Code that calls `<math.h>` functions still links libm itself (`target_link_libraries(app PRIVATE m)`). Some
linkers (mold) do not pull libm in implicitly.

## Consume the checkout without installing
A vendored checkout builds as a subproject (`add_subdirectory`), or CMake can fetch the repository
(`FetchContent`):

```cmake
add_subdirectory(extern/admiral)
target_link_libraries(app PRIVATE admiral::admiral)
```

```cmake
include(FetchContent)
FetchContent_Declare(
  admiral
  GIT_REPOSITORY https://github.com/DiamonDinoia/admiral.git
  GIT_TAG master
)
FetchContent_MakeAvailable(admiral)
target_link_libraries(app PRIVATE admiral::admiral)
```

Either way, as a subproject Admiral turns its tests, benchmarks, and examples off by default.

## Build these docs locally
```bash
cd docs
uv venv .venv && . .venv/bin/activate      # or python -m venv
uv pip install -r requirements.txt         # or pip install
sphinx-build -b html . _build/html
```

Doxygen must be on `PATH`. Sphinx runs it itself. Open `_build/html/index.html`.
