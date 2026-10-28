include_guard(GLOBAL)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

add_library(admiral_optimization_flags INTERFACE)
if(MSVC)
    # A codelet TU instantiates a few hundred templates, each landing in its own COMDAT section,
    # which overruns the 65279-section limit of the default object format.
    target_compile_options(admiral_optimization_flags INTERFACE /bigobj)
endif()

add_library(admiral_fast_math_flags INTERFACE)
if(ADM_USE_FAST_MATH)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_fast_math_flags INTERFACE
            -ffast-math
            -fno-math-errno
            -ffinite-math-only
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(admiral_fast_math_flags INTERFACE /fp:fast)
    endif()
    message(STATUS "Fast math: ENABLED (-ffast-math), admiral's own sources only")
else()
    message(STATUS "Fast math: DISABLED (enable with -DADM_USE_FAST_MATH=ON)")
endif()

if(ADM_TARGET_ARCH STREQUAL "none")
    message(STATUS "Target arch: none (compiler default)")
elseif(CMAKE_CROSSCOMPILING AND ADM_TARGET_ARCH STREQUAL "native")
    message(STATUS "Target arch: native is meaningless when cross-compiling, using compiler default")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv[0-9]|powerpc|ppc)")
        set(_adm_arch_flag -mcpu)
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(_adm_arch_flag -mcpu)
    else()
        set(_adm_arch_flag -march)
    endif()
    target_compile_options(admiral_optimization_flags INTERFACE ${_adm_arch_flag}=${ADM_TARGET_ARCH})
    if(ADM_TARGET_ARCH STREQUAL "native")
        if(_adm_arch_flag STREQUAL "-march")
            target_compile_options(admiral_optimization_flags INTERFACE -mtune=native)
        endif()
        message(WARNING "Binary will be optimized for this CPU and may not run on others")
    endif()
    message(STATUS "Target arch: ${_adm_arch_flag}=${ADM_TARGET_ARCH}")
elseif(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "^([xX]86|AMD64|amd64)")
    target_compile_options(admiral_optimization_flags INTERFACE /arch:AVX2)
    message(STATUS "Target arch: /arch:AVX2")
else()
    message(STATUS "Target arch: ${ADM_TARGET_ARCH} not expressible for this compiler, using default")
endif()

# Per-arch tiers and flags (cmake/tiers/<arch>.cmake): each file guards on its own processor.
include(${CMAKE_CURRENT_LIST_DIR}/tiers/x86.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tiers/aarch64.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tiers/riscv.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tiers/loongarch.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/tiers/s390x.cmake)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_optimization_flags INTERFACE
            -fomit-frame-pointer
        )
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # Measured 2026-10-26, gcc 14.2 at x86-64-v3, medians of 3 on the five slowest
        # Debug TUs (memory/admiral/compile-cost/2026-10-26-g2-flags.md). At -Og -g
        # -fno-var-tracking-assignments the summed CPU of those TUs is 170.7 s. At 6
        # parallel jobs the peak RSS is 12.30 GiB. The GH Debug job runs cmake --build
        # build with no explicit parallel level (ci.yml), so ninja runs its default of 6
        # jobs on the 4-vCPU ubuntu-24.04 runner. The 16 GiB runner memory holds the peak.
        # The setting keeps the line table and the local variables. It cuts only the
        # tracking of the variable locations across assignments. -g3 to -g pays nothing.
        # clang stays at -Og -g3: its worst TU reads 1.2e6 KB, far under the runner
        # memory.
        # A level spelled in CMAKE_CXX_FLAGS stays authoritative, because target options
        # land after it on the command line and the sanitizer presets pin -O1 there.
        string(REGEX MATCH "(^| )-O" _adm_flags_opt_level "${CMAKE_CXX_FLAGS}")
        if(NOT _adm_flags_opt_level)
            target_compile_options(admiral_optimization_flags INTERFACE -Og)
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            target_compile_options(admiral_optimization_flags INTERFACE
                -g
                -fno-var-tracking-assignments
            )
        else()
            target_compile_options(admiral_optimization_flags INTERFACE -g3)
        endif()
        target_compile_options(admiral_optimization_flags INTERFACE
            -fno-omit-frame-pointer
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(admiral_optimization_flags INTERFACE
            /Zi
            /Od
        )
    endif()
endif()

message(STATUS "Optimization flags: fast-math=${ADM_USE_FAST_MATH} arch=${ADM_TARGET_ARCH}")
