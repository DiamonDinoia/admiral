
set(ADM_TEST_SOURCES
    transforms/test_threads.cpp
    transforms/test_alloc.cpp
    kernels/test_iterative.cpp
    transforms/test_nd.cpp
    transforms/test_fct.cpp
    accuracy/test_sweeps.cpp
    transforms/test_route_forced.cpp
    kernels/test_codelet.cpp
    kernels/test_granule.cpp
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
    list(REMOVE_ITEM ADM_TEST_SOURCES transforms/test_threads.cpp)
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)

set(ADM_TEST_EXTRA_LIBS_test_api_c          admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_c_errors   admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_fftw       admiral_fftw_objects)

# The granule dialect's drivers are src-private; the test reaches them by the same path
# the engine does. The poison twin compiles the same source with one deliberately
# lane-flipped radix-3 pool entry and must fail its direct-DFT gates: it is the suite's
# committed positive control on the dialect's constant folding.
set(ADM_TEST_EXTRA_INCLUDES_test_granule        ${PROJECT_SOURCE_DIR}/src)
set(ADM_TEST_EXTRA_INCLUDES_test_granule_poison ${PROJECT_SOURCE_DIR}/src)

foreach(test_src IN LISTS ADM_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
        ${ADM_TEST_EXTRA_INCLUDES_${test_name}})

    target_link_libraries(${test_name}
        PRIVATE
            admiral_internal
            ${ADM_TEST_EXTRA_LIBS_${test_name}}
            $<BUILD_INTERFACE:Catch2::Catch2WithMain>
    )

    if(CATCH2_INCLUDE_DIRS)
        target_include_directories(${test_name} SYSTEM PRIVATE ${CATCH2_INCLUDE_DIRS})
    endif()

    if(TARGET project_warnings)
        target_link_libraries(${test_name} PRIVATE project_warnings)
    endif()
    if(TARGET project_sanitizers)
        target_link_libraries(${test_name} PRIVATE project_sanitizers)
    endif()

    catch_discover_tests(${test_name})
endforeach()

# Positive control for the granule constant folding. Under ADM_GRANULE_TEST_POISON every
# check that routes a radix-3 stage (N in {9, 12, 15, 24}) fails its direct-DFT gate, so
# the binary exits nonzero; WILL_FAIL makes that the asserted outcome. A poison build
# that PASSES would prove the folding cannot fail and this control would go red.
add_executable(test_granule_poison kernels/test_granule.cpp)
target_include_directories(test_granule_poison PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_TEST_EXTRA_INCLUDES_test_granule_poison})
target_compile_definitions(test_granule_poison PRIVATE ADM_GRANULE_TEST_POISON=1)
target_link_libraries(test_granule_poison
    PRIVATE admiral_internal $<BUILD_INTERFACE:Catch2::Catch2WithMain>)
if(CATCH2_INCLUDE_DIRS)
    target_include_directories(test_granule_poison SYSTEM PRIVATE ${CATCH2_INCLUDE_DIRS})
endif()
if(TARGET project_warnings)
    target_link_libraries(test_granule_poison PRIVATE project_warnings)
endif()
if(TARGET project_sanitizers)
    target_link_libraries(test_granule_poison PRIVATE project_sanitizers)
endif()
# The control exists only where the dialect does: at v2-and-below the family compiles
# out and this binary's checks are all skipped, so a WILL_FAIL entry would mis-assert on
# a healthy build. The arch string is the configure-time proxy for kGranuleFmaddsub.
if(ADM_TARGET_ARCH MATCHES "^(native|x86-64-v3|x86-64-v4)$")
    add_test(NAME test_granule_poison COMMAND test_granule_poison)
    set_tests_properties(test_granule_poison PROPERTIES WILL_FAIL TRUE)
endif()

# ELF only: `nm -D` reads a dynamic symbol table, which a PE DLL does not have, and the
# patterns below are Itanium mangling. On Windows llvm-nm reports
# `File format has no dynamic symbol table` and the gate fails on a healthy library.
find_program(ADM_NM NAMES ${CMAKE_NM} nm)
if(ADM_NM AND ADM_HAVE_VERSION_SCRIPT AND NOT WIN32)
    foreach(row
        "admiral;^_Z[A-Z]+7admiral(4plan|5error|7forward|7inverse|8plan_r2[cr]|9axis_plan|10size_error|12strides_plan|14internal_error|17unsupported_error);_ZN7admiral7forwardI|_ZN7admiral7inverseI|_ZN7admiral4planI|_ZN7admiral9axis_planI|_ZN7admiral12strides_planI|_ZN7admiral8plan_r2cI|_ZN7admiral8plan_r2rI|_ZTIN7admiral5error|_ZTIN7admiral10size_error|_ZTIN7admiral14internal_error|_ZTIN7admiral17unsupported_error"
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
    message(STATUS "ABI export gate: SKIPPED (nm=${ADM_NM}, version-script=${ADM_HAVE_VERSION_SCRIPT}, "
                   "win32=${WIN32})")
endif()
