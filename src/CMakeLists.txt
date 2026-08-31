# The library targets. Declaration order matters here; see the notes below.

#   `admiral_engine`    one object per heavy route instantiation, plus the C++ API
#   `admiral_codelets`  one archive of straight-line `kernel<N>` codelets
#   `admiral`           the C++ API, shared + static (`admiral::admiral{,_static}`)
#   `admiral_c`         the C API, shared + static (`admiral::admiral_c{,_static}`)
#   `admiral_fftw`      the FFTW shim, shared + static (`admiral::fftw{,_static}`)
#   `admiral_internal`  build-tree only, for the tests and benchmarks that include
#                       `admiral/detail` directly

include(${CMAKE_CURRENT_SOURCE_DIR}/CodeletCatalog.cmake)

# `OBJECT`, not `STATIC`: weak explicit-instantiation symbols never force archive
# member extraction, so the linker would silently drop a `.a`. Consumers get these
# objects through the `admiral` / `admiral_c` link interfaces.
#
# Longest TUs first: `ninja` starts edges in declaration order, and a `-jN` build's
# wall follows its longest edge. `inst_col_*` and `inst_dif_*` split per direction,
# halving that edge with byte-identical leaf codegen; `inst_plan_*` does not split,
# because its two route trees share sub-instantiations. Whether a split pays is per-TU
# and measured.
#
# `inst_gt_*` is an ENGINE split, not a direction split: hosting the Good-Thomas
# catalog inside `inst_plan_*` exhausted GCC's per-TU inline budget and spilled the
# largest kernels.
#
# `inst_dif_thunks_*` hosts the direction-free families (`dif_thunk` bodies, in-place
# pass, fused pair and triple) once each, keeping per-direction TUs and `libadmiral.a`
# smaller.
#
# `inst_api_*` lives here, not with the C++ interface: both C shims call that API.
add_library(admiral_engine OBJECT
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp
    inst_dif_thunks_f.cpp inst_dif_thunks_d.cpp
    inst_plan_f.cpp inst_plan_d.cpp
    inst_gt_f.cpp inst_gt_d.cpp
    inst_nd_f.cpp inst_nd_d.cpp
    inst_real_f.cpp inst_real_d.cpp
    inst_api_f.cpp inst_api_d.cpp inst_api_ld.cpp)

# The col engine's last pass clones its store-policy lambdas per inline context, and
# fast math contracts and reassociates each clone's butterfly differently. A
# transform's bits then depend on the data's alignment class, because the alignment
# picks the clone that covers a column. Pin the four `inst_col_*` TUs with
# `-ffp-contract=on` plus `-fno-associative-math`. The first flag scopes contraction
# to single expressions; the second forbids rewriting the add tree. Both flags are
# load-bearing; neither alone closes the hole. The 1-D engine runs one clone per
# length and is already bit-stable. Without the pin, the "bit-identical across
# alignment classes" case in `test_strides.cpp` fails on both gcc and clang. MSVC is
# untested; `/fp:precise` is the same withdrawal.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_adm_col_numerics -ffp-contract=on -fno-associative-math)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    set(_adm_col_numerics /fp:precise)
elseif(ADM_USE_FAST_MATH)
    # No pin spelling known for this compiler; warn here so the bit-stability test
    # is not the first report of the hole.
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

# `STATIC`, not `OBJECT`: `admiral_tests` links BOTH `admiral::admiral` and
# `admiral::admiral_c`, and an `OBJECT` library attached to both would inject its
# objects twice into that one executable (duplicate `codelet_dispatch<T>`). An archive
# gets pulled once.
#
# The codelets come after `admiral_engine` on purpose. Otherwise the second-scale
# codelet TUs fill every job slot, and the minute-scale engine TUs on the critical
# path queue behind.
add_library(admiral_codelets STATIC
    ${ADM_CODELET_GENERATED}
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp)

target_compile_features(admiral_codelets PUBLIC cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_codelets PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}           # `codelet_apply.hpp`
    ${ADM_GENERATED_INCLUDE_DIR})         # `admiral/detail/codelet_max.hpp`
target_link_libraries(admiral_codelets PUBLIC xsimd poet::poet)
# `PUBLIC` arch flags: without `-march` the codelets fall back to SSE2, so a consumer
# needs them too.
adm_apply_build_profile(admiral_codelets PUBLIC)

# Public surfaces via `adm_add_surface` (`cmake/AdmiralTargets.cmake`).

# No source of its own. `inst_api_*` are in `admiral_engine` above.
adm_add_surface(admiral
    VERSION_SCRIPT admiral.map
    EXPORT_NAME admiral)

adm_add_surface(admiral_c
    SOURCE c_api.cpp
    VERSION_SCRIPT admiral_c.map
    EXPORT_NAME admiral_c)

# Drop-in `fftw3.h` subset over `admiral::plan<T>`, installed as `admiral::fftw`.
adm_add_surface(admiral_fftw
    SOURCE fftw_compat.cpp
    VERSION_SCRIPT admiral_fftw.map
    EXPORT_NAME fftw)

# Tests and benchmarks include `admiral/detail` directly, so they need what the
# installed package hides: dependency headers, generated `codelet_max.hpp`, the
# codelets' arch flags. Not installed, not aliased into `admiral::`, so
# `admiral_internal` cannot be mistaken for a consumer target. The engine reaches
# them as objects, which makes `extern template` resolve across linkers.
add_library(admiral_internal INTERFACE)
target_include_directories(admiral_internal INTERFACE
    ${PROJECT_SOURCE_DIR}/include
    ${ADM_GENERATED_INCLUDE_DIR})
target_compile_features(admiral_internal INTERFACE cxx_std_${ADM_CXX_STANDARD})
# With `ADM_ENABLE_THREADS` the plan-owned `std::jthread` pool needs pthread in every
# TU; without the option, `ADM_THREADS_LIB` is empty.
target_link_libraries(admiral_internal INTERFACE
    xsimd poet::poet admiral_codelets ${ADM_THREADS_LIB})
target_sources(admiral_internal INTERFACE $<TARGET_OBJECTS:admiral_engine>)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_internal INTERFACE admiral_optimization_flags)
endif()

# Precompiled headers (`-DADM_ENABLE_PCH=ON`), opt-in: a `.gch` stale against its
# flags fails the build until the build tree is wiped. PCH pays on the codelet TUs,
# which are almost all `<xsimd/xsimd.hpp>` parse; not on the instantiation-bound
# engine TUs, and the C interfaces parse neither dependency.
if(ADM_ENABLE_PCH)
    foreach(target admiral_engine admiral_codelets)
        target_precompile_headers(${target} PRIVATE <xsimd/xsimd.hpp> <poet/poet.hpp>)
    endforeach()
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Install.cmake)
endif()
