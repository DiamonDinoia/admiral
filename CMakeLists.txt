# 3.25 for add_subdirectory(SYSTEM), which CPMAddPackage(SYSTEM YES) forwards: it
# is what keeps dependency headers out of the -Werror profile. Deliberately below
# 3.28, where CMP0155 turns on a per-TU C++20 module scan this project has no use
# for (verified: zero cppdyndeps/.ddi edges in build.ninja).
cmake_minimum_required(VERSION 3.25)

project(
    admiral
    VERSION 0.0.1
    DESCRIPTION "Admiral: high-performance C++20 FFT library with C and C++ APIs"
    LANGUAGES C CXX
)

# CMake accepts an unknown -D name in silence, which is how -DFFT_ENABLE_ASAN=ON went
# unnoticed after the rename to ADM_ while every "sanitizer" build ran unsanitized.
get_cmake_property(_adm_vars VARIABLES)
foreach(_adm_var IN LISTS _adm_vars)
    if(_adm_var MATCHES "^FFT_")
        message(FATAL_ERROR
            "${_adm_var} has no effect: this project's options are the ADM_* "
            "block below.")
    endif()
endforeach()

# ============================================================================
# Options
# ============================================================================

# Tests and benchmarks default on standalone, off when consumed via
# add_subdirectory (PROJECT_IS_TOP_LEVEL, set by project() above).
option(ADM_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ADM_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(ADM_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(ADM_ENABLE_THREADS "Multi-threaded plan execution (links Threads); OFF = serial no-op pool" ON)
option(ADM_MEASURE "Compile in plan-time route measurement (admiral::effort::measure, FFTW_MEASURE)" ON)

option(ADM_USE_FAST_MATH "Enable -ffast-math for relaxed floating-point math" ON)

# ONE knob for the target ISA, not a bool plus a raw -march in CMAKE_CXX_FLAGS. Passing
# both put "-march=<level> -march=native" on every admiral TU with native last, so a
# validation sweep over four microarchitecture levels in fact compiled native four times.
set(ADM_TARGET_ARCH "native" CACHE STRING
    "Target microarchitecture: native, none, or an arch/CPU name (x86-64-v3, armv8-a, power8). CompilerOptions.cmake picks -march= or -mcpu= from CMAKE_SYSTEM_PROCESSOR")

# ONE knob for sanitizers, validated in Sanitizers.cmake so a typo is a configure error.
# Building unsanitized while believing otherwise is the failure this replaces.
set(ADM_SANITIZER "none" CACHE STRING "Sanitizers: none, address, undefined, address+undefined, thread")
set_property(CACHE ADM_SANITIZER PROPERTY STRINGS none address undefined address+undefined thread)

# Off in every preset, including release: LTO measured a mean speedup of 1.000 over
# 24 cells, it re-optimizes the whole library on each link, and it makes GCC put
# GIMPLE rather than machine code in the installed archives. Experiments only.
option(ADM_ENABLE_LTO "Enable link-time optimization" OFF)

# Measures as a build-time win (see the PCH block in src/CMakeLists.txt) but a stale
# .gch after a flag change can only be cleared by wiping the build directory, so
# this is opt-in rather than a default that can surprise a consumer.
option(ADM_ENABLE_PCH "Use precompiled headers for the library targets" OFF)

# No BUILD_SHARED_LIBS: adm_add_surface always emits both a SHARED library and a
# STATIC archive, so there is nothing for it to switch.

# ============================================================================
# Modules
# ============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectDefaults)    # standards, visibility, PIC, ccache, compiler floor
include(CPM)                # dependency fetcher
include(CompilerOptions)    # admiral_optimization_flags
include(Dependencies)       # xsimd, poet, and the test/benchmark deps
include(CompilerWarnings)   # project_warnings
include(AdmiralTargets)     # adm_apply_build_profile, adm_add_surface
include(Sanitizers)         # project_sanitizers, only when ADM_SANITIZER is not none

# ============================================================================
# Subdirectories
# ============================================================================

add_subdirectory(src)

# Benchmarks before tests: ninja starts edges in declaration order, so the
# benchmark TUs (the slowest in the tree) are enqueued first and hide behind
# everything else.
if(ADM_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(ADM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

include(CostModel)          # admiral_cost_sweep / admiral_cost_model targets

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Admiral ${PROJECT_VERSION} (${ADM_GIT_REVISION})")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} (C${CMAKE_C_STANDARD})")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (C++${CMAKE_CXX_STANDARD})")
# Both are always produced from one compile, so this is a fact, not an option.
message(STATUS "  Libraries:    shared + static")
message(STATUS "  Tests:        ${ADM_BUILD_TESTS}")
message(STATUS "  Benchmarks:   ${ADM_BUILD_BENCHMARKS}")
message(STATUS "  Threads:      ${ADM_ENABLE_THREADS}")
message(STATUS "  Sanitizer:    ${ADM_SANITIZER}")
message(STATUS "  Fast math:    ${ADM_USE_FAST_MATH}")
message(STATUS "  Measure:      ${ADM_MEASURE}")
message(STATUS "  Target arch:  ${ADM_TARGET_ARCH}")
message(STATUS "  LTO:          ${ADM_ENABLE_LTO}")
message(STATUS "  PCH:          ${ADM_ENABLE_PCH}")
message(STATUS "")
