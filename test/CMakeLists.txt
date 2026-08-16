# test/CMakeLists.txt
# Unit tests using Catch2 v3.
#
# Layout: one directory per area, each holding that area's tests -- including its
# coverage-driven ones, which are tests like any other and belong next to what
# they cover. utils/ holds the shared reference transform, error budget and
# matchers; every test dir includes it as "utils/<file>" (test/ is on the include
# path), so there is one definition of each helper instead of a copy per file.
#
#   utils/       reference DFT, tolerance model, Catch2 matcher adapters
#   api/         public surfaces: C++, C, FFTW compatibility
#   kernels/     codelets, the iterative DIF driver, factor planning, ct_math
#   transforms/  plan selection and routing, N-D, axis, fct, threads, alignment
#   accuracy/    ULP bounds, analytical closed forms, exhaustive sweeps
#
# One executable per source file (not a single monolithic admiral_tests): the test
# TUs build in parallel under -j, and editing one test file only rebuilds and
# relinks that one small executable instead of the whole suite.

# Longest TU first: ninja starts edges in declaration order, so the slowest compile
# must queue before the ones that hide behind it. Re-measure from .ninja_log before
# reordering.
set(ADM_TEST_SOURCES
    transforms/test_threads.cpp
    kernels/test_iterative.cpp
    transforms/test_nd.cpp
    transforms/test_fct.cpp
    accuracy/test_sweeps.cpp
    transforms/test_route_forced.cpp
    kernels/test_codelet.cpp
    api/test_api_cxx.cpp
    api/test_api_c.cpp
    api/test_api_c_errors.cpp
    api/test_api_fftw.cpp
    kernels/test_ct_math.cpp
    kernels/test_factor_plan.cpp
    kernels/test_cost_model.cpp
    transforms/test_plan.cpp
    transforms/test_axis.cpp
    transforms/test_large_align.cpp
    transforms/test_generic_pass.cpp
    transforms/test_r2r.cpp
    accuracy/test_ulp.cpp
    accuracy/test_analytical.cpp
)

# Discover tests using Catch2's CMake integration
list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)

# The compiled libraries each test needs, keyed by target name. Everything else in
# the suite is header-only C++ plus the engine objects, which admiral_internal
# carries. Keep in sync with the <admiral/admiral.h> / <admiral/fftw3.h> includes.
# The object libraries, not the shared ones: same code, but a test then does not
# wait for a shared library's link at all.
set(ADM_TEST_EXTRA_LIBS_test_api_c          admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_c_errors   admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_fftw       admiral_fftw_objects)

foreach(test_src IN LISTS ADM_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)  # e.g. test_plan

    add_executable(${test_name} ${test_src})

    # test/ on the include path so any area can say "utils/reference.hpp".
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Link only the compiled libraries a test actually includes, and the OBJECT
    # libraries rather than the shipped artifacts, so a test does not wait for a
    # shared library's link. admiral_internal already carries the engine objects
    # and codelet archive (see src/CMakeLists.txt).
    target_link_libraries(${test_name}
        PRIVATE
            admiral_internal
            ${ADM_TEST_EXTRA_LIBS_${test_name}}
            $<BUILD_INTERFACE:Catch2::Catch2WithMain>
    )

    # Mark dependency include dirs as SYSTEM to suppress their warnings.
    if(CATCH2_INCLUDE_DIRS)
        target_include_directories(${test_name} SYSTEM PRIVATE ${CATCH2_INCLUDE_DIRS})
    endif()

    if(TARGET project_warnings)
        target_link_libraries(${test_name} PRIVATE project_warnings)
    endif()
    if(TARGET project_sanitizers)
        target_link_libraries(${test_name} PRIVATE project_sanitizers)
    endif()

    # No PCH here even under ADM_ENABLE_PCH: one suite-wide PCH serializes the build
    # behind its generation, and the heaviest TUs are instantiation-bound — they
    # have no parse for a PCH to save.

    # Register this executable's Catch2 test cases with ctest.
    catch_discover_tests(${test_name})
endforeach()

# ============================================================================
# ABI gate: the shipped libraries export their public surface and nothing else
# ============================================================================
# The C++ pattern matches any Itanium-mangled name in namespace admiral, covering
# typeinfo and vtables (_ZTI.., _ZTV..) as well as functions.

# CMAKE_NM is derived from the toolchain prefix, so a cross build reads its own
# objects; plain nm is the host's, and a cross-only image may have none.
#
# Gated on ADM_HAVE_VERSION_SCRIPT: --version-script is an ELF linker feature, so
# on Mach-O nothing restricts the export list and the libraries legitimately
# export every non-hidden symbol.
find_program(ADM_NM NAMES ${CMAKE_NM} nm)
if(ADM_NM AND ADM_HAVE_VERSION_SCRIPT)
    # lib | allowed-export pattern | one regex per surface that must be present
    foreach(row
        "admiral;^_Z[A-Z]+7admiral;_ZN7admiral7forwardI|_ZN7admiral7inverseI|_ZN7admiral4planI|_ZN7admiral9axis_planI|_ZN7admiral8plan_r2cI|_ZN7admiral8plan_r2rI"
        "admiral_c;^(adm|admf)_;adm_plan_1d|adm_forward|adm_r2c_nd|admf_plan_1d|admf_forward"
        "admiral_fftw;^fftw[f]?_;fftw_plan_dft_1d|fftw_execute|fftwf_plan_dft_1d")
        list(GET row 0 lib)
        list(GET row 1 pattern)
        list(GET row 2 required)
        add_test(NAME exported_symbols.${lib}
            COMMAND ${CMAKE_COMMAND}
                -DNM=${ADM_NM}
                -DLIB=$<TARGET_FILE:${lib}>
                -DPATTERN=${pattern}
                -DREQUIRED=${required}
                -P ${PROJECT_SOURCE_DIR}/cmake/CheckExportedSymbols.cmake)
    endforeach()
else()
    # Say so: a silent skip would leave the gate not running on a whole build arm.
    message(STATUS "ABI export gate: SKIPPED (nm=${ADM_NM}, version-script=${ADM_HAVE_VERSION_SCRIPT})")
endif()
