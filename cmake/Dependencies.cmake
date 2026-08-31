include_guard(GLOBAL)

# The project's `-Werror` profile does not reach third-party targets: silence their
# own build. `CPMAddPackage(SYSTEM YES)` covers the headers.
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

# Opt-in multi-threaded benchmarking, default off. On: `ducc0` gets `Threads::Threads`
# and, with `ADM_BENCH_FFTW`, the FFTW thread libs; `--nthreads=N` then races the
# library's N-thread plan against `ducc0(N)` / `FFTW(N)`. Off: `--nthreads` stays 1.
option(ADM_BENCH_THREADS "Link threading into the benchmark references (ducc0/FFTW)" OFF)

# `ducc0` is the FFT reference for benchmarks only; the tests are self-contained
# (analytical references plus round-trip identities).
if(ADM_BUILD_BENCHMARKS)

# Pinned to a commit, not the branch: every speedup in `README.md` is a ratio against
# this exact reference. A pin move must re-run the sweep and update the README table
# in the same commit. CPM's cache dir hashes these arguments, not the sha. Read
# `.git/HEAD` in the source dir for the sha the build used.
CPMAddPackage(
    NAME ducc0
    GIT_REPOSITORY https://github.com/mreineck/ducc.git
    GIT_TAG c4dda23505856367b3aa36ac74e75dc5af993fc2
    DOWNLOAD_ONLY YES
)

if(ducc0_ADDED)
    # Only the required sources: `fft_inst1.cc` instantiates `c2c<float>`,
    # `fft_inst2.cc` `c2c<double>` (+ long double), so the benchmark can race both
    # precisions.
    add_library(ducc0 STATIC
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst1.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/fft/fft_inst2.cc
    )

    # SYSTEM to suppress warnings.
    target_include_directories(ducc0 SYSTEM PUBLIC
        ${ducc0_SOURCE_DIR}/src
    )

    # Tracks the library's standard: the bench includes admiral's headers, whose
    # `span` ABI depends on `ADM_CXX_STANDARD`.
    target_compile_features(ducc0 PUBLIC cxx_std_${ADM_CXX_STANDARD})

    adm_silence_target(ducc0)

    # The baseline keeps fast math: ducc0 is the arm admiral races. Both sides must
    # carry the same FP contract, or the comparison measures flags, not algorithm.
    if(TARGET admiral_optimization_flags)
        target_link_libraries(ducc0 PRIVATE admiral_optimization_flags)
        if(TARGET admiral_fast_math_flags)
            target_link_libraries(ducc0 PRIVATE admiral_fast_math_flags)
        endif()
    else()
        # Fallback for when the optimization target is not yet defined.
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(ducc0 PRIVATE -ffast-math)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
            target_compile_options(ducc0 PRIVATE /fp:fast)
        endif()
    endif()

    # OFF leaves `ducc0` single-thread (`threading.cc` compiled but idle); ON links
    # `Threads::Threads` for `--nthreads=N > 1`.
    if(ADM_BENCH_THREADS)
        find_package(Threads REQUIRED)
        target_link_libraries(ducc0 PUBLIC Threads::Threads)
        message(STATUS "ducc0: threading ENABLED (ADM_BENCH_THREADS)")
    endif()

    add_library(ducc::ducc0 ALIAS ducc0)

    message(STATUS "ducc0: Downloaded and configured")
endif()

endif()  # ADM_BUILD_BENCHMARKS (ducc0)

# Catch2, configured only when the tests are built.
if(ADM_BUILD_TESTS)
    # `SYSTEM YES`: Catch2's own headers must not trip the project's `-Werror` profile.
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

# `xsimd` carries every vector type and operation. Header-only.
# Features admiral relies on: compile-time masked load, `make_sized_batch_t`/`is_void`
# for exactly-r-wide batches, and constant/prefix masks lowering to plain moves on AVX2
# and AVX-512. After any bump, asm-audit the AoS/SoA permute paths per ISA.
# Reuse a parent-provided `xsimd` rather than fetch a second, conflicting copy.
#
# Pinned to master, not a release tag: the fused-scalar-`fms` fix landed after 14.3.0,
# and no release carries the fix. Without the fix a scalar `fms` rounds differently
# from the vector overload, and two ISA widths disagree. Kernels avoid `fms` outright
# (`piece_fnma` / `piece_fma`, `simd_swizzle.hpp`), because a parent-provided `xsimd`
# may predate the fix. Once a tag carries the fix, move to the tag.
if(NOT TARGET xsimd)
    CPMAddPackage(
        NAME xsimd
        GITHUB_REPOSITORY xtensor-stack/xsimd
        GIT_TAG e3cdb6aee0fe0676d2eea50016d4b340acb56709  # master 2026-08-13
        SYSTEM YES
        EXCLUDE_FROM_ALL YES   # its `install()` rules must not land in the prefix
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

# `poet` carries compile-time unrolling and runtime->compile-time dispatch.
# Header-only. If admiral is a subproject, reuse a parent-provided `poet` (target
# `poet::poet`).
if(NOT TARGET poet::poet AND NOT TARGET poet)
    CPMAddPackage(
        NAME poet
        GITHUB_REPOSITORY DiamonDinoia/poet
        VERSION 0.0.1
        SYSTEM YES
        EXCLUDE_FROM_ALL YES   # its `install()` rules must not land in the prefix
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

# Mark the `xsimd`/`poet` headers `SYSTEM`. `CPMAddPackage(SYSTEM YES)` covers the
# fetched copies; the loop below covers a parent project's targets, never added via
# `add_subdirectory`.
foreach(_hdr_only xsimd poet)
    if(TARGET ${_hdr_only})
        get_target_property(_inc ${_hdr_only} INTERFACE_INCLUDE_DIRECTORIES)
        if(_inc)
            set_target_properties(${_hdr_only} PROPERTIES
                INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_inc}")
        endif()
    endif()
endforeach()

# `nanobench` times the benchmarks (median plus MdAPE noise).
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

# FFTW is an optional in-bench reference, found as a system library by `pkg-config`.
# Opt-in, default off: add an FFTW path to `admiral_benchmark` alongside `ducc0`, for
# a same-machine A/B against `FFTW_MEASURE` plans. Requires system `fftw3` + `fftw3f`;
# `PkgConfig::FFTW` carries both and `benchmark/CMakeLists.txt` links it.
option(ADM_BENCH_FFTW "Add an FFTW reference to admiral_benchmark (needs system fftw3/fftw3f)" OFF)
if(ADM_BENCH_FFTW)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3 fftw3f)
    message(STATUS "FFTW: enabled for benchmarks (${FFTW_VERSION})")
    # The `fftw3_threads` / `fftw3f_threads` libs (for `fftw_plan_with_nthreads`) ship
    # WITHOUT a pkg-config `.pc` on Debian/Ubuntu; find by name and expose the paths.
    if(ADM_BENCH_THREADS)
        find_library(FFTW3_THREADS_LIB  NAMES fftw3_threads  REQUIRED)
        find_library(FFTW3F_THREADS_LIB NAMES fftw3f_threads REQUIRED)
        message(STATUS "FFTW: threads companion libs = ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB}")
    endif()
endif()
