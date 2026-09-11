
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
    # Overriding -ffast-math here is the whole point of the pin, and clang 20 started reporting it
    # under a name earlier releases do not know (-Woverriding-t-option), so the suppression is
    # version-gated rather than passed blind.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 20)
        list(APPEND _adm_col_numerics -Wno-overriding-option)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(_adm_col_numerics /fp:precise)
elseif(ADM_USE_FAST_MATH)
    message(WARNING "admiral: no fast-math numerics pin for ${CMAKE_CXX_COMPILER_ID}; "
                    "the column engine may not be bit-stable across data alignments")
endif()
set_source_files_properties(
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    PROPERTIES COMPILE_OPTIONS "${_adm_col_numerics}")

# ADM_PIN_1D: extend the same numerics pin to the unpinned 1-D engine instantiation TUs.
# Closes the per-inline-context contraction/reassociation clone channel there the same way
# the col pin does for the column engine; default OFF, i.e. no flag is emitted unless asked.
option(ADM_PIN_1D "Pin fp contraction/reassociation in the 1-D engine instantiation TUs" OFF)
if(ADM_PIN_1D)
    set_source_files_properties(
        inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp
        inst_dif_thunks_f.cpp inst_dif_thunks_d.cpp
        inst_real_f.cpp inst_real_d.cpp inst_api_ld.cpp
        PROPERTIES COMPILE_OPTIONS "${_adm_col_numerics}")
endif()
target_compile_features(admiral_engine PRIVATE cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_engine PRIVATE xsimd poet::poet ${ADM_THREADS_LIB})
target_compile_definitions(admiral_engine PRIVATE ADM_BUILDING)
# PRIVATE covers every route decision: plan_impl is extern-templated inside admiral_engine only.
if(ADM_PIN_PLAN_ESTIMATE)
    target_compile_definitions(admiral_engine PRIVATE ADM_PIN_PLAN_ESTIMATE)
endif()
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
target_compile_definitions(admiral_codelets PRIVATE ADM_BUILDING)
adm_apply_build_profile(admiral_codelets PUBLIC)

# The engine and the codelets are the only objects a test reaches through, and a test reaches
# admiral::detail. They therefore compile with default visibility, which makes src/*.map the
# only thing that decides the shipped ABI; admiral.map names the public entities for exactly
# that reason. ELF only: a PE has no visibility attribute and admiral_internal is not a DLL.
if(NOT WIN32)
    foreach(target admiral_engine admiral_codelets)
        set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET default)
    endforeach()
endif()

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

# One shared library, built once from the engine objects, linked by every consumer in this
# tree. As an archive it pasted the whole engine into each consumer instead.
#
# ELF ONLY, for two independent reasons. A PE export table holds at most 65535 entries and the
# engine carries far more template instantiations than that, so WINDOWS_EXPORT_ALL_SYMBOLS
# fails the link outright with LNK1189. And a DLL gives the engine its own CRT heap, which is
# the quantity "soa_scratch: small-n execute uses zero heap allocations" counts from the test
# executable. Windows keeps the INTERFACE target, which is what it built before.
if(WIN32)
    add_library(admiral_internal INTERFACE)
    set(adm_internal_scope INTERFACE)
    target_sources(admiral_internal INTERFACE $<TARGET_OBJECTS:admiral_engine>)
    target_link_libraries(admiral_internal INTERFACE admiral_codelets)
else()
    add_library(admiral_internal SHARED
        $<TARGET_OBJECTS:admiral_engine>
        $<TARGET_OBJECTS:admiral_codelets>)
    set(adm_internal_scope PUBLIC)
    # -Bsymbolic-functions binds the library's own cross-TU calls directly, the way the version
    # script already does inside the shipped surfaces, so only a consumer's entry call takes a
    # PLT hop. It leaves data symbols global, so a function-local static keeps one instance per
    # process.
    check_linker_flag(CXX "LINKER:-Bsymbolic-functions" ADM_HAVE_BSYMBOLIC_FUNCTIONS)
    if(ADM_HAVE_BSYMBOLIC_FUNCTIONS)
        target_link_options(admiral_internal PRIVATE "LINKER:-Bsymbolic-functions")
    endif()
endif()
target_include_directories(admiral_internal ${adm_internal_scope}
    ${PROJECT_SOURCE_DIR}/include
    ${ADM_GENERATED_INCLUDE_DIR})
target_compile_features(admiral_internal ${adm_internal_scope} cxx_std_${ADM_CXX_STANDARD})
# Consumers link the engine itself, not the shipped surface, so their headers must not say
# dllimport.
target_compile_definitions(admiral_internal ${adm_internal_scope} ADM_STATIC_DEFINE)
target_link_libraries(admiral_internal ${adm_internal_scope}
    xsimd poet::poet ${ADM_THREADS_LIB})
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_internal ${adm_internal_scope} admiral_optimization_flags)
endif()

if(ADM_ENABLE_PCH)
    foreach(target admiral_engine admiral_codelets)
        target_precompile_headers(${target} PRIVATE <xsimd/xsimd.hpp> <poet/poet.hpp>)
    endforeach()
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Install.cmake)
endif()
