
include(${CMAKE_CURRENT_SOURCE_DIR}/CodeletCatalog.cmake)

add_library(admiral_engine OBJECT
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp
    inst_dif_thunks_f.cpp inst_dif_thunks_d.cpp
    inst_plan_f.cpp inst_plan_d.cpp
    inst_gt_f.cpp inst_gt_d.cpp
    inst_nd_f.cpp inst_nd_d.cpp
    inst_real_f.cpp inst_real_d.cpp
    inst_api_f.cpp inst_api_d.cpp inst_api_ld.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_adm_col_numerics -ffp-contract=on -fno-associative-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(_adm_col_numerics /fp:precise)
elseif(ADM_USE_FAST_MATH)
    message(WARNING "admiral: no fast-math numerics pin for ${CMAKE_CXX_COMPILER_ID}; "
                    "the column engine may not be bit-stable across data alignments")
endif()
set_source_files_properties(
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    PROPERTIES COMPILE_OPTIONS "${_adm_col_numerics}")
target_compile_features(admiral_engine PRIVATE cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_engine PRIVATE xsimd poet::poet ${ADM_THREADS_LIB})
adm_apply_build_profile(admiral_engine)

add_library(admiral_codelets STATIC
    ${ADM_CODELET_GENERATED}
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp)

target_compile_features(admiral_codelets PUBLIC cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_codelets PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_codelets PUBLIC xsimd poet::poet)
adm_apply_build_profile(admiral_codelets PUBLIC)

adm_add_surface(admiral
    VERSION_SCRIPT admiral.map
    EXPORT_NAME admiral)

adm_add_surface(admiral_c
    SOURCE c_api.cpp
    VERSION_SCRIPT admiral_c.map
    EXPORT_NAME admiral_c)

adm_add_surface(admiral_fftw
    SOURCE fftw_compat.cpp
    VERSION_SCRIPT admiral_fftw.map
    EXPORT_NAME fftw)

add_library(admiral_internal INTERFACE)
target_include_directories(admiral_internal INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${ADM_GENERATED_INCLUDE_DIR})
target_compile_features(admiral_internal INTERFACE cxx_std_${ADM_CXX_STANDARD})
target_link_libraries(admiral_internal INTERFACE
    xsimd poet::poet admiral_codelets ${ADM_THREADS_LIB})
target_sources(admiral_internal INTERFACE $<TARGET_OBJECTS:admiral_engine>)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_internal INTERFACE admiral_optimization_flags)
endif()

if(ADM_ENABLE_PCH)
    foreach(target admiral_engine admiral_codelets)
        target_precompile_headers(${target} PRIVATE <xsimd/xsimd.hpp> <poet/poet.hpp>)
    endforeach()
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Install.cmake)
endif()
