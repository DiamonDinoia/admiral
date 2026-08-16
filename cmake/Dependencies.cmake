# Dependencies.cmake — external dependency management (CPM).

include_guard(GLOBAL)

# Third-party code is not held to the project's -Werror profile: silence its own
# build. (Its headers are handled by CPMAddPackage(SYSTEM YES), which forwards
# add_subdirectory(SYSTEM) so consumers see them as system includes.)
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

# Opt-in multi-threaded benchmarking (-DADM_BENCH_THREADS=ON). Off: ducc0 links no
# threading libs and --nthreads stays 1. On: ducc0 gets Threads::Threads and, with
# ADM_BENCH_FFTW, the FFTW threads libs; --nthreads=N then races the library's
# N-thread plan against ducc0(N) / FFTW(N).
option(ADM_BENCH_THREADS "Link threading into the benchmark references (ducc0/FFTW)" OFF)

# ducc0 - reference FFT implementation, benchmarks only (the tests are
# self-contained: analytical references + round-trip identities).
if(ADM_BUILD_BENCHMARKS)

# Pinned to a commit, not the branch: every speedup in README.md is a ratio against
# this exact reference. CPM's cache directory name is a hash of these arguments, not
# the git sha — read .git/HEAD in the source dir for what was actually built.
# Bumping the pin is a benchmark-methodology change: re-run the sweep and update the
# README table in the same commit.
CPMAddPackage(
    NAME ducc0
    GIT_REPOSITORY https://github.com/mreineck/ducc.git
    GIT_TAG c4dda23505856367b3aa36ac74e75dc5af993fc2
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

    adm_silence_target(ducc0)

    # Apply optimization flags if available. The baseline keeps fast math: it is the
    # arm admiral is timed against, so both sides must carry the same FP contract or
    # the comparison measures the flags instead of the algorithm.
    if(TARGET admiral_optimization_flags)
        target_link_libraries(ducc0 PRIVATE admiral_optimization_flags)
        if(TARGET admiral_fast_math_flags)
            target_link_libraries(ducc0 PRIVATE admiral_fast_math_flags)
        endif()
    else()
        # Fallback: Apply basic fast-math if optimization target not yet defined
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(ducc0 PRIVATE -ffast-math)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ducc0 PRIVATE /fp:fast)
        endif()
    endif()

    # Threading: OFF leaves ducc0 a single-thread reference (its threading.cc is
    # compiled but idle). ON links Threads::Threads for --nthreads=N > 1.
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
    # SYSTEM YES: Catch2's own headers must not trip the project's -Werror profile.
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

# ============================================================================
# xsimd - SIMD vectorization (header-only)
# ============================================================================

# Features admiral relies on: compile-time masked load, make_sized_batch_t/is_void
# for exactly-r-wide batches, and constant/prefix masks that lower to plain moves on
# both AVX2 and AVX-512 — the masked tails (dif_passes.hpp, codelet.hpp,
# good_thomas) hit exactly those kernels. After any bump: asm-audit the AoS/SoA
# permute paths per ISA (zip vpermt2, shuffle deinterleave, transpose networks).
# Reuse a parent-provided xsimd rather than fetch a second, conflicting copy.
#
# Upstream master, not the newest release tag (14.3.0): 14.3.0 lacks "make scalar
# fms fused", so its scalar `fms` rounds differently from the vector batch overload
# and two ISAs of different width disagree. Move to a tag once one carries it.
if(NOT TARGET xsimd)
    CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY xtensor-stack/xsimd
        GIT_TAG 67e96b04f9c0ed530326fb189907cf68ca4030b7  # master: scalar fms fused
        SYSTEM YES
        EXCLUDE_FROM_ALL YES   # its install() rules must not land in our prefix
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_BENCHMARK OFF"
            "BUILD_EXAMPLES OFF"
    )

    if(xsimd_ADDED)
        message(STATUS "xsimd: xtensor-stack/xsimd 67e96b04 (master + fused scalar fms)")
    endif()
else()
    message(STATUS "xsimd: reusing parent-provided target")
endif()

# ============================================================================
# poet - compile-time unroll + runtime->compile-time dispatch (header-only)
# ============================================================================

# Reuse a parent-provided poet (target poet::poet) when admiral is a subproject.
if(NOT TARGET poet::poet AND NOT TARGET poet)
    CPMAddPackage(
        NAME poet
        GITHUB_REPOSITORY DiamonDinoia/poet
        VERSION 0.0.1
        SYSTEM YES
        EXCLUDE_FROM_ALL YES   # its install() rules must not land in our prefix
        OPTIONS
            "POET_STRICT_WARNINGS OFF"
            "POET_BUILD_TESTS OFF"
            "POET_BUILD_EXAMPLES OFF"
            "POET_BUILD_BENCHMARKS OFF"
    )

    if(poet_ADDED)
        message(STATUS "poet: Downloaded v0.0.1")
    endif()
else()
    message(STATUS "poet: reusing parent-provided target")
endif()

# Treat xsimd/poet headers as SYSTEM so their internal warnings do not trip the
# project's -Werror profile. CPMAddPackage(SYSTEM YES) covers the copies fetched
# above; this covers a parent project's targets, never added via add_subdirectory.
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

# Opt-in: -DADM_BENCH_FFTW=ON adds an FFTW path to admiral_benchmark alongside
# ducc0, for a same-machine A/B against FFTW_MEASURE plans. Requires system
# fftw3 + fftw3f; PkgConfig::FFTW carries both and benchmark/CMakeLists.txt links it.
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
