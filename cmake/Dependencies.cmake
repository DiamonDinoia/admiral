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

option(ADM_BENCH_THREADS "Link threading into the benchmark references (ducc0/FFTW)" OFF)

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

    if(ADM_BENCH_THREADS)
        find_package(Threads REQUIRED)
        target_link_libraries(ducc0 PUBLIC Threads::Threads)
        message(STATUS "ducc0: threading ENABLED (ADM_BENCH_THREADS)")
    endif()

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
        GIT_TAG e3cdb6aee0fe0676d2eea50016d4b340acb56709
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARK OFF"
            "BUILD_EXAMPLES OFF"
    )

    if(xsimd_ADDED)
        message(STATUS "xsimd: xtensor-stack/xsimd e3cdb6ae (master + fused scalar fms)")
    endif()
else()
    message(STATUS "xsimd: reusing parent-provided target")
endif()

if(NOT TARGET poet::poet AND NOT TARGET poet)
    CPMAddPackage(
        NAME poet
        GITHUB_REPOSITORY DiamonDinoia/poet
        GIT_TAG bf3f917de8c03c0a7cd97ffe099cb990421ca049
        SYSTEM YES
        EXCLUDE_FROM_ALL YES
        OPTIONS
            "POET_STRICT_WARNINGS OFF"
            "POET_BUILD_TESTS OFF"
            "POET_BUILD_EXAMPLES OFF"
            "POET_BUILD_BENCHMARKS OFF"
    )

    if(poet_ADDED)
        message(STATUS "poet: DiamonDinoia/poet bf3f917d (feat/exact-unroll, exact dynamic_for unroll, constant-count fix)")
    endif()
else()
    message(STATUS "poet: reusing parent-provided target")
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

option(ADM_BENCH_FFTW "Add an FFTW reference to admiral_benchmark (needs system fftw3/fftw3f)" OFF)
if(ADM_BENCH_FFTW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3 fftw3f)
    message(STATUS "FFTW: enabled for benchmarks (${FFTW_VERSION})")
    if(ADM_BENCH_THREADS)
        find_library(FFTW3_THREADS_LIB  NAMES fftw3_threads  REQUIRED)
        find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads REQUIRED)
        message(STATUS "FFTW: threads companion libs = ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB}")
    endif()
endif()
