cmake_minimum_required(VERSION 3.20)

# Project declaration
project(
    admiral
    VERSION 0.1.0
    DESCRIPTION "Admiral: high-performance C++20 FFT library with C and C++ APIs"
    LANGUAGES C CXX
)

# Only do these if this is the main project, not if included via add_subdirectory
set(IS_MAIN_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(IS_MAIN_PROJECT ON)
endif()

# ============================================================================
# Project Options
# ============================================================================

option(ADM_BUILD_TESTS "Build unit tests" ${IS_MAIN_PROJECT})
option(ADM_BUILD_BENCHMARKS "Build benchmarks" ${IS_MAIN_PROJECT})
option(ADM_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(ADM_ENABLE_ASAN "Enable AddressSanitizer for testing" OFF)

# Optimization options
option(ADM_USE_FAST_MATH "Enable -ffast-math for relaxed floating-point math" ON)
option(ADM_USE_NATIVE_ARCH "Enable -march=native for CPU-specific optimizations" ON)
option(ADM_ENABLE_LTO "Enable link-time optimization" ON)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Set C and C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Position-independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use ccache when available — the per-N codelet TUs (and Catch2) are stable across
# rebuilds, so caching object files makes incremental reconfigures near-instant.
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

# Hidden symbol visibility project-wide: smaller export table / binary and better
# inlining & devirtualization. The codelet library keeps every kernel<N> body
# internal and exposes only codelet_dispatch<T>. The public C API is explicitly
# re-exported via the ADM_C_API attribute (see include/admiral/admiral.h).
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# Export compile commands for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# CMake Modules
# ============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Load CPM for dependency management
include(CPM)

# Configure compiler optimization flags
include(CompilerOptions)

# Fetch external dependencies
include(Dependencies)

# Configure compiler warnings
include(CompilerWarnings)

# Configure sanitizers (if enabled)
if(ADM_ENABLE_ASAN)
    include(Sanitizers)
endif()

# ============================================================================
# Subdirectories
# ============================================================================

add_subdirectory(src)

if(ADM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(ADM_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Admiral Configuration Summary:")
message(STATUS "  Version:      ${PROJECT_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  Tests:        ${ADM_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${ADM_BUILD_BENCHMARKS}")
message(STATUS "  ASAN:         ${ADM_ENABLE_ASAN}")
message(STATUS "  Fast math:    ${ADM_USE_FAST_MATH}")
message(STATUS "  Native arch:  ${ADM_USE_NATIVE_ARCH}")
message(STATUS "  LTO:          ${ADM_ENABLE_LTO}")
message(STATUS "")
