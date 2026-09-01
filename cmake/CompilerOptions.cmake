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

if(ADM_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
    if(ipo_supported)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
        message(STATUS "Link-time optimization (LTO): ENABLED")
        message(WARNING
            "LTO is not the shipping configuration: GCC writes GIMPLE, not machine "
            "code, into the installed .a archives, so they will only link with this "
            "exact compiler. Leave ADM_ENABLE_LTO off to ship.")
    else()
        message(STATUS "Link-time optimization (LTO): NOT SUPPORTED")
        message(STATUS "  Reason: ${ipo_error}")
    endif()
else()
    message(STATUS "Link-time optimization (LTO): DISABLED (enable with -DADM_ENABLE_LTO=ON)")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_optimization_flags INTERFACE
            -fomit-frame-pointer
        )
    endif()
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(admiral_optimization_flags INTERFACE
            -g3
            -fno-omit-frame-pointer
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(admiral_optimization_flags INTERFACE
            /Zi
            /Od
        )
    endif()
endif()

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

message(STATUS "Optimization flags: fast-math=${ADM_USE_FAST_MATH} arch=${ADM_TARGET_ARCH} lto=${ADM_ENABLE_LTO}")
