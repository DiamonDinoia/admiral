
set(ADM_TEST_SOURCES
    transforms/test_threads.cpp
    transforms/test_alloc.cpp
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
    list(REMOVE_ITEM ADM_TEST_SOURCES transforms/test_threads.cpp)
endif()

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)

get_target_property(CATCH2_INCLUDE_DIRS Catch2::Catch2WithMain INTERFACE_INCLUDE_DIRECTORIES)

set(ADM_TEST_EXTRA_LIBS_test_api_c          admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_c_errors   admiral_c_objects)
set(ADM_TEST_EXTRA_LIBS_test_api_fftw       admiral_fftw_objects)

foreach(test_src IN LISTS ADM_TEST_SOURCES)
    get_filename_component(test_name ${test_src} NAME_WE)

    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

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

    # admiral_internal is a shared library, so Windows resolves it through PATH at load time
    # rather than through the link-time path. Harmless on ELF, where the RPATH answers it.
    catch_discover_tests(${test_name} DL_PATHS $<TARGET_FILE_DIR:admiral_internal>)
endforeach()

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
