# Dependencies.cmake
# External dependency management using CPM
# All dependency fetching logic is centralized here

# Guard against multiple inclusion
if(_ADM_DEPENDENCIES_INCLUDED)
    return()
endif()
set(_ADM_DEPENDENCIES_INCLUDED TRUE)

# Opt-in multi-threaded benchmarking (-DFFT_BENCH_THREADS=ON). Off by default so
# the single-thread build stays EXACTLY as today: ducc0 links no threading libs
# and --nthreads defaults to 1 (serial regardless). When ON, ducc0 gets
# Threads::Threads (its threading.cc actually threads) and, if ADM_BENCH_FFTW is
# also set, the FFTW threads libs are added. The benchmark's --nthreads=N then
# races our N-thread plan vs ducc0(N) / FFTW(N).
option(ADM_BENCH_THREADS "Link threading into the benchmark references (ducc0/FFTW)" OFF)

# ============================================================================
# ducc0 - Reference FFT implementation (benchmarks only)
# ============================================================================
# ducc0 is a benchmark reference, not a dependency of the library or the tests
# (the tests are self-contained: analytical references + round-trip identities).
if(ADM_BUILD_BENCHMARKS)

# Custom ducc0 build - DOWNLOAD_ONLY, then create custom target
CPMAddPackage(
    NAME ducc0
    GIT_REPOSITORY https://github.com/mreineck/ducc.git
    GIT_TAG ducc0
    DOWNLOAD_ONLY YES
)

if(ducc0_ADDED)
    # Create custom STATIC library for ducc0 with only required sources.
    # fft_inst1.cc instantiates c2c<float>; fft_inst2.cc instantiates c2c<double>
    # (+ long double). Both are needed so the benchmark can race float and double.
    add_library(ducc0 STATIC
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst1.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst2.cc
    )

    # Set include directory (SYSTEM to suppress warnings)
    target_include_directories(ducc0 SYSTEM PUBLIC
        ${ducc0_SOURCE_DIR}/src
    )

    # Set C++20 standard for ducc0
    target_compile_features(ducc0 PUBLIC cxx_std_20)

    # Disable warnings for ducc0 (third-party code)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(ducc0 PRIVATE -w)  # Disable all warnings
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(ducc0 PRIVATE /w)  # Disable all warnings
    endif()

    # Apply optimization flags if available
    if(TARGET admiral_optimization_flags)
        target_link_libraries(ducc0 PRIVATE admiral_optimization_flags)
    else()
        # Fallback: Apply basic fast-math if optimization target not yet defined
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(ducc0 PRIVATE -ffast-math)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ducc0 PRIVATE /fp:fast)
        endif()
    endif()

    # Threading: OFF by default -> NO threading libraries linked (ducc0
    # threading.cc is compiled but idle, the single-thread reference). With
    # -DFFT_BENCH_THREADS=ON, link Threads::Threads so ducc0 threads the batch
    # dimension for the --nthreads=N > 1 benchmark comparison.
    if(ADM_BENCH_THREADS)
        find_package(Threads REQUIRED)
        target_link_libraries(ducc0 PUBLIC Threads::Threads)
        message(STATUS "ducc0: threading ENABLED (ADM_BENCH_THREADS)")
    endif()

    # Create alias for consistent target naming
    add_library(ducc::ducc0 ALIAS ducc0)

    message(STATUS "ducc0: Downloaded and configured")
endif()

endif()  # ADM_BUILD_BENCHMARKS (ducc0)

# ============================================================================
# Catch2 - Testing framework (only if building tests)
# ============================================================================

if(ADM_BUILD_TESTS)
    CPMAddPackage(
        NAME Catch2
        GITHUB_REPOSITORY catchorg/Catch2
        VERSION 3.11.0
        OPTIONS
            "CATCH_BUILD_TESTING OFF"
            "CATCH_INSTALL_DOCS OFF"
            "CATCH_INSTALL_EXTRAS OFF"
    )

    if(Catch2_ADDED)
        # Disable warnings for Catch2 (third-party code)
        if(TARGET Catch2)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(Catch2 PRIVATE -w)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options(Catch2 PRIVATE /w)
            endif()
        endif()

        if(TARGET Catch2WithMain)
            if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
                target_compile_options(Catch2WithMain PRIVATE -w)
            elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
                target_compile_options(Catch2WithMain PRIVATE /w)
            endif()
        endif()

        # Treat Catch2 headers as SYSTEM in consuming TUs so their internal
        # conversions (e.g. matching a float value promotes to double inside
        # matcher.match) do not trip our -Werror profile, same as xsimd/poet.
        foreach(_c2 Catch2 Catch2WithMain)
            if(TARGET ${_c2})
                get_target_property(_inc ${_c2} INTERFACE_INCLUDE_DIRECTORIES)
                if(_inc)
                    set_target_properties(${_c2} PROPERTIES
                        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
                endif()
            endif()
        endforeach()

        message(STATUS "Catch2: Downloaded v3.11.0")
    endif()
endif()

# ============================================================================
# xsimd - SIMD vectorization (header-only)
# ============================================================================

# Features yafft relies on: compile-time masked load
# `batch::load(ptr, batch_bool_constant<...>, unaligned)`, `make_sized_batch_t`/
# `is_void` for exactly-r-wide batches, and constant/prefix masks that lower to
# plain moves (not vmaskmov/kmask) on both AVX2 and AVX-512 — yafft's masked
# tails (dif_passes.hpp, codelet.hpp, good_thomas) hit exactly those kernels.
# After any bump: asm-audit the AoS/SoA permute paths per ISA (zip vpermt2,
# shuffle deinterleave, f32/f64 transpose networks).
CPMAddPackage(
    NAME xsimd
    GITHUB_REPOSITORY xtensor-stack/xsimd
    GIT_TAG 14.3.0
    OPTIONS
        "BUILD_TESTS OFF"
        "BUILD_BENCHMARK OFF"
        "BUILD_EXAMPLES OFF"
)

if(xsimd_ADDED)
    message(STATUS "xsimd: 14.3.0")
endif()

# ============================================================================
# poet - compile-time unroll + runtime->compile-time dispatch (header-only)
# ============================================================================

CPMAddPackage(
    NAME poet
    GIT_REPOSITORY https://github.com/DiamonDinoia/poet.git
    GIT_TAG 5e394535b790b1ead2b9cf3c10ea9558364d2649  # feat/unroll1-contract: dynamic_for<1> guaranteed rolled
    OPTIONS
        "POET_STRICT_WARNINGS OFF"
        "POET_BUILD_TESTS OFF"
        "POET_BUILD_EXAMPLES OFF"
        "POET_BUILD_BENCHMARKS OFF"
)

if(poet_ADDED)
    message(STATUS "poet: Downloaded (pinned 5e39453)")
endif()

# Treat xsimd/poet headers as SYSTEM so their internal warnings (e.g. C-style
# casts) do not trip our -Werror profile.
foreach(_hdr_only xsimd poet)
    if(TARGET ${_hdr_only})
        get_target_property(_inc ${_hdr_only} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            set_target_properties(${_hdr_only} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
        endif()
    endif()
endforeach()

# ============================================================================
# nanobench - robust microbenchmarking (median, MdAPE noise) for benchmarks
# ============================================================================

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

# ============================================================================
# FFTW - optional in-bench reference (system lib via pkg-config)
# ============================================================================

# Opt-in: -DFFT_BENCH_FFTW=ON adds an FFTW path to admiral_benchmark alongside
# ducc0, for a fresh same-machine A/B against FFTW_MEASURE plans. Requires
# system fftw3 + fftw3f (single- and double-precision). The PkgConfig::FFTW
# imported target carries both; benchmark/CMakeLists.txt links it when enabled.
option(ADM_BENCH_FFTW "Add an FFTW reference to admiral_benchmark (needs system fftw3/fftw3f)" OFF)
if(ADM_BENCH_FFTW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3 fftw3f)
    message(STATUS "FFTW: enabled for benchmarks (${FFTW_VERSION})")
    # The fftw3_threads / fftw3f_threads companion libs (for
    # fftw_plan_with_nthreads) ship WITHOUT a pkg-config .pc on Debian/Ubuntu, so
    # find them by name and expose the paths for benchmark/CMakeLists.txt to link.
    if(ADM_BENCH_THREADS)
        find_library(FFTW3_THREADS_LIB  NAMES fftw3_threads  REQUIRED)
        find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads REQUIRED)
        message(STATUS "FFTW: threads companion libs = ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB}")
    endif()
endif()
