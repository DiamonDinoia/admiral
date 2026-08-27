# The library targets. Declaration order matters here; see the notes below.
#
#   admiral_engine    one object per heavy route instantiation, plus the C++ API
#   admiral_codelets  one archive of straight-line kernel<N> codelets
#   admiral           the C++ API, shared + static (admiral::admiral{,_static})
#   admiral_c         the C API, shared + static (admiral::admiral_c{,_static})
#   admiral_fftw      the FFTW shim, shared + static (admiral::fftw{,_static})
#   admiral_internal  build-tree only, for the tests and benchmarks that include
#                     admiral/detail directly

include(${CMAKE_CURRENT_SOURCE_DIR}/CodeletCatalog.cmake)

# ============================================================================
# Engine instantiation objects
# ============================================================================
# OBJECT, not STATIC. Weak explicit-instantiation symbols never force archive
# member extraction, so the linker would silently drop a .a. Consumers
# get these objects directly through the admiral / admiral_c link interfaces.
#
# Longest TUs first, because ninja starts edges in declaration order, and a -jN build's
# wall follows its single longest edge. The heavy engines are inst_col_* (batched column
# engine) and inst_dif_* (1-D mixed-radix); both are split per direction, which halves that
# edge at no cost in codegen (leaf symbols are byte-identical across the split). inst_plan_*
# is NOT split. Its two route trees share nearly all sub-instantiations, so splitting
# measured slower. Whether a direction split pays is per-TU and must be measured.
#
# inst_gt_* is an ENGINE split, not a direction split. The Good-Thomas catalog expands
# 9 tuples times two aliasing arms per direction, and hosting that inside inst_plan_*
# exhausted GCC's per-TU inline budget partway through, which spilled the largest
# kernels. Its own TU compiles them all at the default budget.
#
# The direction split alone compiles the direction-FREE families twice per precision. Every
# inst_dif_thunks_* instantiates each of them once (dif_thunk: mid-pass body plain and
# chiplet, in-place pass, fused pair and triple), which is why the per-direction TUs are
# the smaller half. Hosting the second copy in the shipped archive also grew libadmiral.a.
#
# inst_api_* are here rather than with the C++ interface because both C shims call
# that same public API, so every interface needs its objects.
add_library(admiral_engine OBJECT
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp
    inst_dif_thunks_f.cpp inst_dif_thunks_d.cpp
    inst_plan_f.cpp inst_plan_d.cpp
    inst_gt_f.cpp inst_gt_d.cpp
    inst_nd_f.cpp inst_nd_d.cpp
    inst_real_f.cpp inst_real_d.cpp
    inst_api_f.cpp inst_api_d.cpp inst_api_ld.cpp)

# The column engine's last pass clones its store-policy lambdas per inline context
# (bulk/prefix/suffix), and under fast-math the compiler contracts and REASSOCIATES
# the butterfly math differently per clone: one transform then returns
# 1-ULP-different bits depending on which clone a column's width-and-alignment cover
# picks, and so on the data's alignment class. Pin the association at the call
# boundary: contract=on scopes contraction to single expressions, no-associative-math
# withdraws the permission to rewrite the add tree. Both flags are load-bearing and
# neither alone closes it. inst_col_* only: the 1-D engine runs one clone per length
# and is already bit-stable. Both gcc and clang need this (test_strides.cpp's
# "bit-identical across alignment classes" case fails without it on both); MSVC is
# untested, /fp:precise is the same withdrawal.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_adm_col_numerics -ffp-contract=on -fno-associative-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(_adm_col_numerics /fp:precise)
elseif(ADM_USE_FAST_MATH)
    # No spelling known for this compiler, so the col engine keeps whatever fast-math
    # licence it was given. Say so here rather than let the bit-stability test be the
    # first news of it.
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

# ============================================================================
# Compiled codelet archive
# ============================================================================
# STATIC, not OBJECT. admiral_tests links BOTH admiral::admiral and
# admiral::admiral_c, and a pure OBJECT library attached to both would inject its
# objects twice into that one executable (duplicate codelet_dispatch<T>). An
# archive gets pulled once.
#
# Declared after admiral_engine on purpose. The 64 sized codelet TUs are seconds each
# and would otherwise fill every job slot while the engine TUs sit queued behind them.
# Those TUs take minutes each and head the critical path.
add_library(admiral_codelets STATIC
    ${ADM_CODELET_GENERATED}
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp)

target_compile_features(admiral_codelets PUBLIC cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_codelets PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}           # codelet_apply.hpp
    ${ADM_GENERATED_INCLUDE_DIR})         # admiral/detail/codelet_max.hpp
target_link_libraries(admiral_codelets PUBLIC xsimd poet::poet)
# PUBLIC arch flags. Without -march the codelets fall back to SSE2 and are slow,
# so a consumer compiling against them needs the same flags.
adm_apply_build_profile(admiral_codelets PUBLIC)

# ============================================================================
# Public surfaces (see adm_add_surface in cmake/AdmiralTargets.cmake)
# ============================================================================

# No source of its own. inst_api_* are in admiral_engine above.
adm_add_surface(admiral
    VERSION_SCRIPT admiral.map
    EXPORT_NAME admiral)

adm_add_surface(admiral_c
    SOURCE c_api.cpp
    VERSION_SCRIPT admiral_c.map
    EXPORT_NAME admiral_c)

# Drop-in fftw3.h subset (the core complex-DFT plan API) over admiral::plan<T>.
# Installed as admiral::fftw, not admiral::admiral_fftw.
adm_add_surface(admiral_fftw
    SOURCE fftw_compat.cpp
    VERSION_SCRIPT admiral_fftw.map
    EXPORT_NAME fftw)

# ============================================================================
# Internal interface for the tests and benchmarks
# ============================================================================
# They include admiral/detail directly, so they need what the installed package
# hides: the dependency headers, the generated codelet_max.hpp, and the codelets'
# arch flags. Not installed and not aliased into admiral::, so it cannot be
# mistaken for a consumer target. The engine reaches them as objects rather than a
# linked OBJECT library, which is what makes extern template resolve across linkers.
add_library(admiral_internal INTERFACE)
target_include_directories(admiral_internal INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${ADM_GENERATED_INCLUDE_DIR})
target_compile_features(admiral_internal INTERFACE cxx_std_${ADM_CXX_STANDARD})
# With ADM_ENABLE_THREADS the plan-owned std::jthread pool needs pthread in every
# TU that builds a pool; without it, ADM_THREADS_LIB is empty.
target_link_libraries(admiral_internal INTERFACE
    xsimd poet::poet admiral_codelets ${ADM_THREADS_LIB})
target_sources(admiral_internal INTERFACE $<TARGET_OBJECTS:admiral_engine>)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_internal INTERFACE admiral_optimization_flags)
endif()

# ============================================================================
# Precompiled headers (-DADM_ENABLE_PCH=ON)
# ============================================================================
# Opt-in, not default. A .gch that goes stale against its flags fails the build with an
# error the user can only clear by deleting the build tree, which is a bad trade for
# seconds. It pays on the codelet TUs, which are almost all <xsimd/xsimd.hpp> parse
# (1.74s -> 0.20s per TU). It barely helps the engine TUs, which are instantiation-bound,
# and it measured a LOSS on the test suite. The C interfaces parse neither dependency.
if(ADM_ENABLE_PCH)
    foreach(target admiral_engine admiral_codelets)
        target_precompile_headers(${target} PRIVATE <xsimd/xsimd.hpp> <poet/poet.hpp>)
    endforeach()
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Install.cmake)
endif()
