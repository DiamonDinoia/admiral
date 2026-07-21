# test/CMakeLists.txt
# Unit tests using Catch2 v3.
#
# One executable per source file (not a single monolithic admiral_tests): the test
# TUs build in parallel under -j, and editing one test file only rebuilds and
# relinks that one small executable instead of the whole suite.

set(ADM_TEST_SOURCES
    test_fft.cpp
    test_fft_plan.cpp
    test_fft_fct.cpp
    test_fftw_compat.cpp
    test_fft_c.cpp
    test_fft_analytical.cpp
    test_fft_codelet.cpp
    test_fft_factor_plan.cpp
    test_fft_iterative.cpp
    test_fft_nd.cpp
    test_fft_threads.cpp
    test_fft_large_align.cpp
    test_fft_coverage.cpp
)

# Discover tests using Catch2's CMake integration
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

# Relax warnings-as-errors for test TUs only. This MUST be linked AFTER
# project_warnings so its -Wno-error is emitted after project_warnings' -Werror
# (CMake preserves link order for INTERFACE compile options, and last flag wins);
# a plain target_compile_options(-Wno-error) is emitted BEFORE the dependency's
# -Werror and gets overridden. The library itself stays -Werror-clean via src/,
# admiral_c and the benchmarks — tests exercise the library, they don't re-audit it,
# so cast boilerplate for -Wdouble-promotion/-Wsign-conversion in test-only
# scaffolding buys nothing.
add_library(test_warning_overrides INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_warning_overrides INTERFACE -Wno-error)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(test_warning_overrides INTERFACE /WX-)
endif()

get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)

foreach(test_src IN LISTS ADM_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)  # e.g. test_fft_plan

    add_executable(${test_name} ${test_src})

    target_link_libraries(${test_name}
        PRIVATE
            admiral::admiral
            admiral::admiral_c
            admiral::fftw
            $<BUILD_INTERFACE:Catch2::Catch2WithMain>
    )

    # Mark dependency include dirs as SYSTEM to suppress their warnings.
    if(CATCH2_INCLUDE_DIRS)
        target_include_directories(${test_name} SYSTEM PRIVATE ${CATCH2_INCLUDE_DIRS})
    endif()

    if(TARGET project_warnings)
        # Order matters: project_warnings (-Werror) then the -Wno-error override.
        target_link_libraries(${test_name} PRIVATE project_warnings test_warning_overrides)
    endif()
    if(ADM_ENABLE_ASAN AND TARGET project_sanitizers)
        target_link_libraries(${test_name} PRIVATE project_sanitizers)
    endif()

    # Register this executable's Catch2 test cases with ctest.
    catch_discover_tests(${test_name})
endforeach()
