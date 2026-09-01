cmake_minimum_required(VERSION 3.25)

project(
    admiral
    VERSION 0.0.1
    DESCRIPTION "Admiral: high-performance FFT library with C and C++ APIs"
    LANGUAGES C CXX
)

get_cmake_property(_adm_vars VARIABLES)
foreach(_adm_var IN LISTS _adm_vars)
    if(_adm_var MATCHES "^FFT_")
        message(FATAL_ERROR
            "${_adm_var} has no effect: this project's options are the ADM_* "
            "block below.")
    endif()
endforeach()

option(ADM_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(ADM_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(ADM_BUILD_EXAMPLES "Build the runnable examples" ${PROJECT_IS_TOP_LEVEL})
option(ADM_ENABLE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
option(ADM_ENABLE_THREADS "Multi-threaded plan execution (links Threads); OFF = serial no-op pool" ON)
option(ADM_MEASURE "Compile in plan-time route measurement (admiral::effort::measure, FFTW_MEASURE)" ON)

set(ADM_CXX_STANDARD 20 CACHE STRING "C++ standard for library and public headers: 17, 20 or 23")
set_property(CACHE ADM_CXX_STANDARD PROPERTY STRINGS 17 20 23)
if(NOT ADM_CXX_STANDARD MATCHES "^(17|20|23)$")
    message(FATAL_ERROR "ADM_CXX_STANDARD=${ADM_CXX_STANDARD}: only 17, 20 and 23 are supported")
endif()

option(ADM_USE_FAST_MATH "Enable -ffast-math for relaxed floating-point math" ON)

set(ADM_TARGET_ARCH "native" CACHE STRING
    "Target microarchitecture: native, none, or an arch/CPU name (x86-64-v3, armv8-a, power8). CompilerOptions.cmake picks -march= or -mcpu= from CMAKE_SYSTEM_PROCESSOR")

set(ADM_SANITIZER "none" CACHE STRING "Sanitizers: none, address, undefined, address+undefined, thread")
set_property(CACHE ADM_SANITIZER PROPERTY STRINGS none address undefined address+undefined thread)

option(ADM_ENABLE_LTO "Enable link-time optimization" OFF)

option(ADM_ENABLE_PCH "Use precompiled headers for the library targets" OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectDefaults)
include(CPM)
include(CompilerOptions)
include(Dependencies)
include(CompilerWarnings)
include(StaticAnalysis)
include(AdmiralTargets)
include(Sanitizers)

add_subdirectory(src)

if(ADM_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(ADM_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(ADM_BUILD_EXAMPLES)
    enable_testing()
    add_subdirectory(examples)
endif()

include(CostModel)

message(STATUS "")
message(STATUS "Admiral ${PROJECT_VERSION} (${ADM_GIT_REVISION})")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  C compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION} (C${CMAKE_C_STANDARD})")
message(STATUS "  C++ compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} (C++${CMAKE_CXX_STANDARD})")
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
