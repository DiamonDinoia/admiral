# Settings that apply to every target.

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD ${ADM_CXX_STANDARD})   # `ADM_CXX_STANDARD`: CMakeLists.txt options block
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Hidden visibility project-wide. The codelet library exposes only
# `codelet_dispatch<T>`; the public C entry points opt back in with `ADM_C_API` /
# `FFTW_C_API` (`admiral.h`).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# The per-N codelet TUs and Catch2 are stable across rebuilds, so caching pays.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# A tarball or export without git gets the version alone, so the fallback never fails
# a configure.
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --always --dirty --tags
                    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                    OUTPUT_VARIABLE ADM_GIT_REVISION
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
endif()
if(NOT ADM_GIT_REVISION)
    set(ADM_GIT_REVISION "unknown")
endif()

# Recommended-compiler floor: older compilers build correctly but emit slower codelets
# (register spills, outlined kernel bodies). Warn, do not fail: performance, not correctness.
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
   OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19))
    message(WARNING
        "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} is below the recommended "
        "floor (GCC 14 / Clang 19). Expect slower codelets: register spills and outlined "
        "kernel bodies. Correctness is unaffected.")
endif()
