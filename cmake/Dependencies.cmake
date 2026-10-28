include_guard(GLOBAL)

function(adm_silence_target)
    foreach(target IN LISTS ARGN)
        if(TARGET ${target})
            if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options(${target} PRIVATE /w)
            else()
                target_compile_options(${target} PRIVATE -w)
            endif()
        endif()
    endforeach()
endfunction()

if(ADM_BUILD_BENCHMARKS)

CPMAddPackage(
    NAME ducc0
    GIT_REPOSITORY https://github.com/mreineck/ducc.git
    GIT_TAG c4dda23505856367b3aa36ac74e75dc5af993fc2
    DOWNLOAD_ONLY YES
)

if(ducc0_ADDED)
    add_library(ducc0 STATIC
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst1.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst2.cc
    )

    target_include_directories(ducc0 SYSTEM PUBLIC
        ${ducc0_SOURCE_DIR}/src
    )

    target_compile_features(ducc0 PUBLIC cxx_std_${ADM_CXX_STANDARD})

    adm_silence_target(ducc0)

    if(TARGET admiral_optimization_flags)
        target_link_libraries(ducc0 PRIVATE admiral_optimization_flags)
        if(TARGET admiral_fast_math_flags)
            target_link_libraries(ducc0 PRIVATE admiral_fast_math_flags)
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(ducc0 PRIVATE -ffast-math)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ducc0 PRIVATE /fp:fast)
        endif()
    endif()

    # The benchmark references always thread: every perf receipt compares against the
    # threaded baselines, so a serial ducc0/FFTW has no user here.
    find_package(Threads REQUIRED)
    target_link_libraries(ducc0 PUBLIC Threads::Threads)

    add_library(ducc::ducc0 ALIAS ducc0)

    message(STATUS "ducc0: Downloaded and configured")
endif()

endif()

if(ADM_BUILD_TESTS)
    CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.11.0
        SYSTEM YES
        OPTIONS
            "CATCH_BUILD_TESTING OFF"
            "CATCH_INSTALL_DOCS OFF"
            "CATCH_INSTALL_EXTRAS OFF"
    )

    if(Catch2_ADDED)
        adm_silence_target(Catch2 Catch2WithMain)
        message(STATUS "Catch2: Downloaded v3.11.0")
    endif()
endif()

if(NOT TARGET xsimd)
    CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY xtensor-stack/xsimd
        GIT_TAG 457dcf71ec0aa54a04ca8aa65bce7c1c72361226
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARK OFF"
            "BUILD_EXAMPLES OFF"
    )

    if(xsimd_ADDED)
        message(STATUS "xsimd: xtensor-stack/xsimd 457dcf71 (master)")
    endif()
else()
    message(STATUS "xsimd: reusing parent-provided target")
endif()

if(NOT TARGET poet::poet AND NOT TARGET poet)
    CPMAddPackage(
        NAME poet
        GITHUB_REPOSITORY DiamonDinoia/poet
        GIT_TAG e10d62c2468918dbb2a85cc22ccb39a635db633d
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "POET_STRICT_WARNINGS OFF"
            "POET_BUILD_TESTS OFF"
            "POET_BUILD_EXAMPLES OFF"
            "POET_BUILD_BENCHMARKS OFF"
    )

    if(poet_ADDED)
        message(STATUS "poet: DiamonDinoia/poet e10d62c (main tip)")
    endif()
else()
    message(STATUS "poet: reusing parent-provided target")
endif()

# The engine's scratch blocks reach multiple MiB and are allocated per execute. glibc unmaps
# any block past DEFAULT_MMAP_THRESHOLD_MAX (32 MiB) on free, so the next execute re-faults the
# whole thing, and even below that a fresh address costs the cache residency the previous call
# built up. snmalloc keeps the mapping and the address and releases only the PAGES, through
# MADV_FREE on Linux and BSD, MADV_FREE_REUSABLE on Apple and its own Windows PAL. It is used
# ONLY through src/scratch_alloc.cpp and it does not replace the global operator new, so an
# application's own allocator is untouched.
if(NOT TARGET snmalloc)
    if(ADM_CXX_STANDARD LESS 20)
        set(_adm_snmalloc_cxx17 ON)
    else()
        set(_adm_snmalloc_cxx17 OFF)
    endif()
    CPMAddPackage(
        NAME snmalloc
        GITHUB_REPOSITORY microsoft/snmalloc
        GIT_TAG 511e91a2de604ce716841f32f5e929a066c4a978
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "SNMALLOC_HEADER_ONLY_LIBRARY ON"
            "SNMALLOC_BUILD_TESTING OFF"
            "SNMALLOC_USE_CXX17 ${_adm_snmalloc_cxx17}"
    )

    if(snmalloc_ADDED)
        message(STATUS "snmalloc: microsoft/snmalloc 511e91ad (header-only, scratch only)")
    endif()
else()
    message(STATUS "snmalloc: reusing parent-provided target")
endif()

foreach(_hdr_only xsimd poet)
    if(TARGET ${_hdr_only})
        get_target_property(_inc ${_hdr_only} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            set_target_properties(${_hdr_only} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
        endif()
    endif()
endforeach()

if(ADM_BUILD_BENCHMARKS)
CPMAddPackage(
    NAME nanobench
    GITHUB_REPOSITORY martinus/nanobench
    VERSION 4.3.11
)

if(nanobench_ADDED)
    message(STATUS "nanobench: Downloaded v4.3.11")
endif()
endif()

# The FFTW reference is auto-detected: where pkg-config sees fftw3 and fftw3f,
# admiral_benchmark gains its --fftw-ab arm (perf-master installs libfftw3-dev). Where it
# does not, the benchmark builds without the arm.
if(ADM_BUILD_BENCHMARKS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW QUIET IMPORTED_TARGET fftw3 fftw3f)
    endif()
    if(FFTW_FOUND)
        find_library(FFTW3_THREADS_LIB  NAMES fftw3_threads  REQUIRED)
        find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads REQUIRED)
        set(_adm_fftw_ver "")
        if(FFTW_VERSION)
            set(_adm_fftw_ver " (${FFTW_VERSION})")
        endif()
        message(STATUS "FFTW: found${_adm_fftw_ver}; admiral_benchmark gets the --fftw-ab reference")
    else()
        message(STATUS "FFTW: not found; admiral_benchmark builds without the --fftw-ab reference")
    endif()
endif()
