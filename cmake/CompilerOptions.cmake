# CompilerOptions.cmake
# Optimization flags and compiler-specific options
# Centralizes all performance-related compiler settings

# Guard against multiple inclusion
if(_FFT_COMPILER_OPTIONS_INCLUDED)
    return()
endif()
set(_FFT_COMPILER_OPTIONS_INCLUDED TRUE)

# ============================================================================
# Build Type Defaults
# ============================================================================

# Set default build type if none specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ============================================================================
# Optimization Flags Interface Library
# ============================================================================

add_library(admiral_optimization_flags INTERFACE)

# Fast math optimization (optional, user-controlled)
if(ADM_USE_FAST_MATH)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_optimization_flags INTERFACE
            -ffast-math
            -fno-math-errno
            -ffinite-math-only
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(admiral_optimization_flags INTERFACE /fp:fast)
    endif()
    message(STATUS "Fast math: ENABLED (-ffast-math)")
else()
    message(STATUS "Fast math: DISABLED (enable with -DFFT_USE_FAST_MATH=ON)")
endif()

# Native architecture optimization (optional, user-controlled)
if(ADM_USE_NATIVE_ARCH)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
            # Apple clang accepts -march=native on arm64 but degrades to generic
            # scheduling and drops FP16/crypto; -mcpu=native is the host-tuned
            # flag (Apple-pipeline scheduling + full ISA).
            target_compile_options(admiral_optimization_flags INTERFACE -mcpu=native)
        else()
            target_compile_options(admiral_optimization_flags INTERFACE
                -march=native
                -mtune=native
            )
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # MSVC: /arch:AVX2 or /arch:AVX512
        # Note: MSVC doesn't have direct equivalent to -march=native
        target_compile_options(admiral_optimization_flags INTERFACE /arch:AVX2)
    endif()
    message(STATUS "Native architecture: ENABLED (-march=native)")
    message(WARNING "Binary will be optimized for this CPU and may not run on others")
else()
    message(STATUS "Native architecture: DISABLED (enable with -DFFT_USE_NATIVE_ARCH=ON)")
endif()

# Link-time optimization (optional, user-controlled)
if(ADM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
        message(STATUS "Link-time optimization (LTO): ENABLED")
    else()
        message(STATUS "Link-time optimization (LTO): NOT SUPPORTED")
        message(STATUS "  Reason: ${ipo_error}")
    endif()
else()
    message(STATUS "Link-time optimization (LTO): DISABLED (enable with -DFFT_ENABLE_LTO=ON)")
endif()

# ============================================================================
# Build Type Specific Flags
# ============================================================================

# Additional Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # -funroll-loops deliberately ABSENT (removed 2026-07-04). ASM census of all
        # hot loops (docs/bridge-evidence/unroll-audit/): every perf-critical kernel
        # (dif_pass r4/r8, fused2/fused3, codelets, col_dif, transpose packs) emits
        # byte-identical code with or without it — unrolling there is explicit
        # (poet::static_for / templates). Where the flag DID act (r2/r3 passes,
        # vecpass +30%, r2c f64 pack +380% static bloat) alternated PMU cycle A/Bs
        # measured it neutral at <1% both precisions. Explicit structure over flag
        # reliance: codegen must not depend on unroll heuristics.
        target_compile_options(admiral_optimization_flags INTERFACE
            -fomit-frame-pointer
        )
    endif()
endif()

# Debug build options
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_optimization_flags INTERFACE
            -g3              # Maximum debug info
            -fno-omit-frame-pointer  # Better stack traces
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(admiral_optimization_flags INTERFACE
            /Zi              # Debug info
            /Od              # No optimization
        )
    endif()
endif()

# ============================================================================
# User Override Support
# ============================================================================

# Allow users to add custom flags via:
# cmake -DFFT_EXTRA_C_FLAGS="-mavx512f" -DFFT_EXTRA_CXX_FLAGS="-mavx512f" ..

if(DEFINED ADM_EXTRA_C_FLAGS)
    separate_arguments(ADM_EXTRA_C_FLAGS_LIST UNIX_COMMAND "${ADM_EXTRA_C_FLAGS}")
    target_compile_options(admiral_optimization_flags INTERFACE
        $<$<COMPILE_LANGUAGE:C>:${ADM_EXTRA_C_FLAGS_LIST}>
    )
    message(STATUS "Custom C flags: ${ADM_EXTRA_C_FLAGS}")
endif()

if(DEFINED ADM_EXTRA_CXX_FLAGS)
    separate_arguments(ADM_EXTRA_CXX_FLAGS_LIST UNIX_COMMAND "${ADM_EXTRA_CXX_FLAGS}")
    target_compile_options(admiral_optimization_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${ADM_EXTRA_CXX_FLAGS_LIST}>
    )
    message(STATUS "Custom C++ flags: ${ADM_EXTRA_CXX_FLAGS}")
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "Optimization flags configured:")
if(ADM_USE_FAST_MATH)
    message(STATUS "  - Fast math enabled")
endif()
if(ADM_USE_NATIVE_ARCH)
    message(STATUS "  - Native CPU optimization enabled")
endif()
if(ADM_ENABLE_LTO AND ipo_supported)
    message(STATUS "  - Link-time optimization enabled")
endif()
