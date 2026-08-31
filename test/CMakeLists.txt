# Unit tests with Catch2 v3. One directory per area. `utils/` holds the shared
# reference transform, tolerance model and matchers. Every test dir includes those
# headers as "utils/<file>" (`test/` is on the include path), so each helper has one
# definition:
#   utils/       reference DFT, tolerance model, Catch2 matcher adapters
#   api/         public interfaces: C++, C, FFTW compatibility
#   kernels/     codelets, the iterative DIF driver, factor planning, ct_math
#   transforms/  plan selection and routing, N-D, axis, fct, threads, alignment
#   accuracy/    ULP bounds, analytical closed forms, exhaustive sweeps
# One executable per source file: parallel builds, and editing one test rebuilds only
# that executable.

# Longest TU first: `ninja` starts edges in declaration order. Re-measure from
# `.ninja_log` before any reorder.
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
    kernels/test_cxx_compat.cpp
    transforms/test_plan.cpp
    transforms/test_axis.cpp
    transforms/test_strides.cpp
    transforms/test_large_align.cpp
    transforms/test_generic_pass.cpp
    transforms/test_r2r.cpp
    accuracy/test_ulp.cpp
    accuracy/test_analytical.cpp
    accuracy/test_long_double.cpp
)

if(NOT ADM_ENABLE_THREADS)
    # Every case in `test_threads.cpp` assumes a real pool; against the serial stub
    # the cases fail. An empty binary registers ctest's `NOT_BUILT` placeholder, a
    # suite failure.
    list(REMOVE_ITEM ADM_TEST_SOURCES transforms/test_threads.cpp)
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)

# The compiled libraries each test needs, keyed by target name; everything else is
# header-only plus the engine objects, which `admiral_internal` carries. OBJECT
# libraries, not the shipped artifacts, so no test waits for a shared library's link.
# Keep in sync with the `<admiral/admiral.h>` / `<admiral/fftw3.h>` includes.
set(ADM_TEST_EXTRA_LIBS_test_api_c          admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_c_errors   admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_fftw       admiral_fftw_objects)

foreach(test_src IN LISTS ADM_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)  # e.g. `test_plan`

    add_executable(${test_name} ${test_src})

    # `test/` on the include path so any area can say "utils/reference.hpp".
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # `admiral_internal` already carries the engine objects and codelet archive.
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

    # No PCH even under `ADM_ENABLE_PCH`: one suite-wide PCH serializes the build, and
    # these TUs are instantiation-bound, with no parse for a PCH to save.

    catch_discover_tests(${test_name})
endforeach()

# ABI gate: the shipped libraries export their public interface and nothing else. The
# C++ pattern matches any Itanium-mangled name in namespace admiral, covering typeinfo
# and vtables (`_ZTI..`, `_ZTV..`) as well as functions.

# `CMAKE_NM` follows the toolchain prefix, so a cross build reads its own objects;
# plain `nm` is the host's `nm`, which a cross-only image may lack. Gated on
# `ADM_HAVE_VERSION_SCRIPT`: `--version-script` is ELF-only; on Mach-O nothing
# restricts the export list.
find_program(ADM_NM NAMES ${CMAKE_NM} nm)
if(ADM_NM AND ADM_HAVE_VERSION_SCRIPT)
    # lib | allowed-export pattern | one regex per interface that must be present
    foreach(row
        "admiral;^_Z[A-Z]+7admiral;_ZN7admiral7forwardI|_ZN7admiral7inverseI|_ZN7admiral4planI|_ZN7admiral9axis_planI|_ZN7admiral12strides_planI|_ZN7admiral8plan_r2cI|_ZN7admiral8plan_r2rI"
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
    # A silent skip would stop the gate on a whole build arm.
    message(STATUS "ABI export gate: SKIPPED (nm=${ADM_NM}, version-script=${ADM_HAVE_VERSION_SCRIPT})")
endif()
