
include(${CMAKE_CURRENT_SOURCE_DIR}/CodeletCatalog.cmake)

# The ISA-dependent engine. ADM_DISPATCH builds it once at the floor and once more per upper
# tier. The shared TUs (one snmalloc state, one probe cache, one set of error key functions and
# the dispatch seam) build only at the floor.
set(ADM_TIER_SOURCES
    inst_col_f_fwd.cpp inst_col_f_inv.cpp inst_col_d_fwd.cpp inst_col_d_inv.cpp
    inst_col_first_f_fwd.cpp inst_col_first_f_inv.cpp inst_col_first_d_fwd.cpp
    inst_col_first_d_inv.cpp inst_col_mid_f.cpp inst_col_mid_d.cpp
    inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp
    inst_dif_thunks_f.cpp inst_dif_thunks_d.cpp
    inst_dif_thunks_ip_f.cpp inst_dif_thunks_ip_d.cpp
    inst_dif_thunks_chip_f.cpp inst_dif_thunks_chip_d.cpp
    inst_plan_f.cpp inst_plan_d.cpp
    inst_gt_f.cpp inst_gt_d.cpp
    inst_nd_f.cpp inst_nd_d.cpp
    inst_real_f.cpp inst_real_d.cpp
    inst_api_f.cpp inst_api_d.cpp inst_api_ld.cpp)
# The dif instantiation TUs are the memory monsters. A /usr/bin/time compiler launcher measures
# 6893 MiB of peak RSS at the v4 tier (Release g++-14) and 9836 MiB under asan g++-14; two of
# them at once overshoot the 15 GiB CI runner (ci.yml:26 holds the per-TU table). Each one
# therefore compiles in the one-slot "heavy" Ninja job pool, in every config that builds them.
# The pool is a target property (CMake ignores JOB_POOL_COMPILE on a source), which is why
# these sources build in their own OBJECT library next to each engine variant.
set(ADM_HEAVY_SOURCES inst_dif_f_fwd.cpp inst_dif_f_inv.cpp inst_dif_d_fwd.cpp inst_dif_d_inv.cpp)
set(ADM_TIER_LIGHT_SOURCES ${ADM_TIER_SOURCES})
list(REMOVE_ITEM ADM_TIER_LIGHT_SOURCES ${ADM_HEAVY_SOURCES})
set_property(GLOBAL PROPERTY JOB_POOLS heavy=1)
option(ADM_DISPATCH "Add run-time engine tiers above the floor (tier map: cmake/tiers/)" OFF)
if(ADM_DISPATCH)
    if(NOT ADM_TIER_TAGS)
        message(FATAL_ERROR "ADM_DISPATCH has no tiers for ${CMAKE_SYSTEM_PROCESSOR} (cmake/tiers)")
    endif()
    if(NOT ADM_TARGET_ARCH STREQUAL ADM_TIER_FLOOR)
        message(FATAL_ERROR "ADM_DISPATCH needs ADM_TARGET_ARCH=${ADM_TIER_FLOOR} (the floor), not ${ADM_TARGET_ARCH}")
    endif()
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        message(FATAL_ERROR "ADM_DISPATCH conflicts with CMAKE_INTERPROCEDURAL_OPTIMIZATION: LTO objects hold IR, not machine code, and the tier rename needs machine code")
    endif()
    # llvm-nm/llvm-objcopy preferred; GNU binutils nm/objcopy do the same rename job
    # (cmake/rename_tier.cmake); a tool that cannot do it fails loud there.
    find_program(ADM_TIER_NM NAMES llvm-nm nm)
    find_program(ADM_TIER_OBJCOPY NAMES llvm-objcopy objcopy)
    if(NOT ADM_TIER_NM OR NOT ADM_TIER_OBJCOPY)
        message(FATAL_ERROR "ADM_DISPATCH needs nm and objcopy, llvm or GNU binutils (nm=${ADM_TIER_NM}, objcopy=${ADM_TIER_OBJCOPY})")
    endif()
endif()
if(ADM_DISPATCH)
    add_library(admiral_engine OBJECT
        ${ADM_TIER_LIGHT_SOURCES} api.cpp errors.cpp scratch_alloc.cpp large_route_probe.cpp)
    target_compile_definitions(admiral_engine PRIVATE ADM_DISPATCH)
else()
    add_library(admiral_engine OBJECT
        ${ADM_TIER_LIGHT_SOURCES} scratch_alloc.cpp large_route_probe.cpp)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_adm_col_numerics -ffp-contract=on -fno-associative-math)
    # Overriding -ffast-math here is the whole point of the pin, and newer clangs report it
    # (AppleClang included: its ID is "AppleClang", not "Clang"). Probe for the suppression
    # rather than version-gating so older compilers that do not know it do not choke on it.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Wno-overriding-option" ADM_HAVE_WNO_OVERRIDING_OPTION)
    if(ADM_HAVE_WNO_OVERRIDING_OPTION)
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
    inst_col_first_f_fwd.cpp inst_col_first_f_inv.cpp inst_col_first_d_fwd.cpp
    inst_col_first_d_inv.cpp inst_col_mid_f.cpp inst_col_mid_d.cpp
    PROPERTIES COMPILE_OPTIONS "${_adm_col_numerics}")

# The long double engine (dd_fft.hpp) splits values exactly; fast math folds the splits to zero.
# Contraction keeps the 0.5 ulp bound but moves low-order output bits, and the compilers default
# to different modes. Off costs at most 0.5% of the engine's instructions. icpx reassociates by
# default but defines no __FAST_MATH__, so only this flag, not kDdStrictFp, catches it.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    set_property(SOURCE inst_api_ld.cpp APPEND PROPERTY COMPILE_OPTIONS
        -fno-fast-math -ffp-contract=off)
    # Fast math also kept the x87 engine's complex products inline; without it each one calls
    # __mulxc3, and the x87 lines run at half speed.
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-fcx-limited-range" ADM_HAVE_CX_LIMITED_RANGE)
    if(ADM_USE_FAST_MATH AND ADM_HAVE_CX_LIMITED_RANGE)
        set_property(SOURCE inst_api_ld.cpp APPEND PROPERTY COMPILE_OPTIONS -fcx-limited-range)
    endif()
    if(ADM_HAVE_WNO_OVERRIDING_OPTION)
        set_property(SOURCE inst_api_ld.cpp APPEND PROPERTY COMPILE_OPTIONS -Wno-overriding-option)
    endif()
endif()
target_compile_features(admiral_engine PRIVATE cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_engine PRIVATE xsimd poet::poet snmalloc ${ADM_THREADS_LIB})
target_compile_definitions(admiral_engine PRIVATE ADM_BUILDING)
# scratch_alloc.cpp reads LD_PRELOAD via std::getenv; Windows CRT deprecates it under every driver.
if(WIN32)
    target_compile_definitions(admiral_engine PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
adm_apply_build_profile(admiral_engine)

# The heavy dif TUs, pooled one compile at a time (see the ADM_HEAVY_SOURCES note). Same flags
# as admiral_engine, so the split moves objects without moving a single flag.
add_library(admiral_engine_heavy OBJECT ${ADM_HEAVY_SOURCES})
set_property(TARGET admiral_engine_heavy PROPERTY JOB_POOL_COMPILE heavy)
if(ADM_DISPATCH)
    target_compile_definitions(admiral_engine_heavy PRIVATE ADM_DISPATCH)
endif()
target_compile_features(admiral_engine_heavy PRIVATE cxx_std_${ADM_CXX_STANDARD})
target_include_directories(admiral_engine_heavy PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_engine_heavy PRIVATE xsimd poet::poet snmalloc ${ADM_THREADS_LIB})
target_compile_definitions(admiral_engine_heavy PRIVATE ADM_BUILDING)
if(WIN32)
    target_compile_definitions(admiral_engine_heavy PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
adm_apply_build_profile(admiral_engine_heavy)

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
    foreach(target admiral_engine admiral_engine_heavy admiral_codelets)
        set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET default)
    endforeach()
endif()

# ADM_DISPATCH: the floor is ADM_TARGET_ARCH=${ADM_TIER_FLOOR}. The engine and the codelets build once
# one more per tier. cmake/rename_tier.cmake then gives every symbol that a tier defines, except
# the tier entry and @feat.00, the prefix admiral_<tag>_, so no symbol of one tier can bind to
# another. The rename covers local symbols too: a local COMDAT group signature binds a tier's
# groups to the floor's by name at link time (check_dispatch.sh (a) pins the prefixed state).
# The per-source COMPILE_OPTIONS of this file (the col pin and the inst_api_ld strict-FP set)
# ride the tier copies through CMake's directory-scoped source properties; check (g) asserts
# this per tier from compile_commands.json. Tier code keeps master's mutable header statics
# (thread_pool.hpp, the plan.hpp probe/pool caches, plan.hpp's route_memo, nd_plan.hpp's
# workspace slot): one tier runs per process, and a measured route choice is ISA-specific, so
# route_memo must be -- and is -- a per-tier copy.
set(ADM_TIER_OBJECTS "")
if(ADM_DISPATCH)
    if(MSVC)
        set(_adm_obj_ext .obj)
    else()
        set(_adm_obj_ext .o)
    endif()
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/floor.list
         CONTENT "$<JOIN:$<TARGET_OBJECTS:admiral_engine>,\n>\n$<JOIN:$<TARGET_OBJECTS:admiral_engine_heavy>,\n>\n$<JOIN:$<TARGET_OBJECTS:admiral_codelets>,\n>\n")
    foreach(tag ${ADM_TIER_TAGS})
        set(eng admiral_engine_${tag})
        add_library(${eng} OBJECT
            ${ADM_TIER_LIGHT_SOURCES} ${ADM_CODELET_GENERATED}
            ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp)
        add_library(${eng}_heavy OBJECT ${ADM_HEAVY_SOURCES})
        set_property(TARGET ${eng}_heavy PROPERTY JOB_POOL_COMPILE heavy)
        target_compile_features(${eng} PRIVATE cxx_std_${ADM_CXX_STANDARD})
        target_include_directories(${eng} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${ADM_GENERATED_INCLUDE_DIR})
        target_link_libraries(${eng} PRIVATE xsimd poet::poet snmalloc ${ADM_THREADS_LIB})
        target_compile_definitions(${eng} PRIVATE ADM_BUILDING ADM_DISPATCH ADM_STATIC_DEFINE)
        if(WIN32)
            target_compile_definitions(${eng} PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        adm_apply_build_profile(${eng})
        # The tier flag library comes after the profile, so its -march wins over the floor's.
        add_library(admiral_tier_${tag}_flags INTERFACE)
        target_compile_options(admiral_tier_${tag}_flags INTERFACE ${_adm_tier_flag_${tag}})
        target_link_libraries(${eng} PRIVATE admiral_tier_${tag}_flags)
        if(NOT WIN32)
            set_target_properties(${eng} PROPERTIES CXX_VISIBILITY_PRESET default)
        endif()

        # The heavy dif TUs of this tier, pooled one compile at a time (see ADM_HEAVY_SOURCES).
        target_compile_features(${eng}_heavy PRIVATE cxx_std_${ADM_CXX_STANDARD})
        target_include_directories(${eng}_heavy PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${ADM_GENERATED_INCLUDE_DIR})
        target_link_libraries(${eng}_heavy PRIVATE xsimd poet::poet snmalloc ${ADM_THREADS_LIB})
        target_compile_definitions(${eng}_heavy PRIVATE ADM_BUILDING ADM_DISPATCH ADM_STATIC_DEFINE)
        if(WIN32)
            target_compile_definitions(${eng}_heavy PRIVATE _CRT_SECURE_NO_WARNINGS)
        endif()
        adm_apply_build_profile(${eng}_heavy)
        target_link_libraries(${eng}_heavy PRIVATE admiral_tier_${tag}_flags)
        if(NOT WIN32)
            set_target_properties(${eng}_heavy PROPERTIES CXX_VISIBILITY_PRESET default)
        endif()

        file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/tier_${tag}.list
             CONTENT "$<JOIN:$<TARGET_OBJECTS:${eng}>,\n>\n$<JOIN:$<TARGET_OBJECTS:${eng}_heavy>,\n>\n")
        set(renamed "")
        foreach(src ${ADM_TIER_SOURCES} ${ADM_CODELET_GENERATED}
                    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp)
            get_filename_component(base ${src} NAME)
            list(APPEND renamed ${CMAKE_CURRENT_BINARY_DIR}/tier_${tag}/${base}${_adm_obj_ext})
        endforeach()
        add_custom_command(OUTPUT ${renamed}
            COMMAND ${CMAKE_COMMAND}
                    -DNM=${ADM_TIER_NM} -DOBJCOPY=${ADM_TIER_OBJCOPY}
                    -DTAG=${tag} -DLIST=${CMAKE_CURRENT_BINARY_DIR}/tier_${tag}.list
                    -DOUT=${CMAKE_CURRENT_BINARY_DIR}/tier_${tag}
                    -P ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rename_tier.cmake
            DEPENDS ${eng} $<TARGET_OBJECTS:${eng}> ${eng}_heavy $<TARGET_OBJECTS:${eng}_heavy> ${CMAKE_CURRENT_BINARY_DIR}/tier_${tag}.list
                    ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/rename_tier.cmake
            VERBATIM)
        list(APPEND ADM_TIER_OBJECTS ${renamed})
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
# fftw_compat.cpp calls dlsym on POSIX. CMAKE_DL_LIBS is empty on Windows.
target_link_libraries(admiral_fftw PRIVATE ${CMAKE_DL_LIBS})
target_link_libraries(admiral_fftw_static PUBLIC ${CMAKE_DL_LIBS})
target_link_libraries(admiral_fftw_objects INTERFACE ${CMAKE_DL_LIBS})

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
    target_sources(admiral_internal INTERFACE
        $<TARGET_OBJECTS:admiral_engine>
        $<TARGET_OBJECTS:admiral_engine_heavy>
        ${ADM_TIER_OBJECTS})
    target_link_libraries(admiral_internal INTERFACE admiral_codelets)
else()
    add_library(admiral_internal SHARED
        $<TARGET_OBJECTS:admiral_engine>
        $<TARGET_OBJECTS:admiral_engine_heavy>
        $<TARGET_OBJECTS:admiral_codelets>
        ${ADM_TIER_OBJECTS})
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

if(PROJECT_IS_TOP_LEVEL)
    include(${CMAKE_CURRENT_SOURCE_DIR}/Install.cmake)
endif()
