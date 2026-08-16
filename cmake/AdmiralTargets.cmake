# Helpers shared by the library targets.

include(CheckLinkerFlag)

# Attach the project's flag profile: optimization flags, warnings, sanitizers; each
# is an INTERFACE target that may be absent, hence the guards. `scope` applies to
# the optimization flags only — the codelets pass PUBLIC so consumers compile with
# the same arch flags. Fast math stays PRIVATE regardless (admiral_fast_math_flags
# in CompilerOptions.cmake).
function(adm_apply_build_profile target)
    set(scope PRIVATE)
    if(ARGC GREATER 1)
        set(scope ${ARGV1})
    endif()
    if(TARGET admiral_optimization_flags)
        target_link_libraries(${target} ${scope} admiral_optimization_flags)
    endif()
    if(TARGET admiral_fast_math_flags)
        target_link_libraries(${target} PRIVATE admiral_fast_math_flags)
    endif()
    if(TARGET project_warnings)
        target_link_libraries(${target} PRIVATE project_warnings)
    endif()
    if(TARGET project_sanitizers)
        target_link_libraries(${target} PRIVATE project_sanitizers)
    endif()
endfunction()

# Restrict a shared library's exported ABI to the symbols a version script names.
# Attributes plus hidden default visibility cannot reach compiler-emitted RTTI
# (polymorphic template instantiations carry default visibility);
# test_exported_symbols guards what is left.
function(adm_restrict_exports target map)
    set(script ${CMAKE_CURRENT_SOURCE_DIR}/${map})
    check_linker_flag(CXX "LINKER:--version-script=${script}" ADM_HAVE_VERSION_SCRIPT)
    if(ADM_HAVE_VERSION_SCRIPT)
        target_link_options(${target} PRIVATE "LINKER:--version-script=${script}")
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${script})
    endif()
endfunction()

# One public surface (C++, the C API, or the FFTW shim) = shared + static from one
# compile:
#
#   <name>_objects   OBJECT, compiled once — only with SOURCE
#   <name>           SHARED, exported as admiral::<EXPORT_NAME>
#   <name>_static    STATIC, exported as admiral::<EXPORT_NAME>_static
#
# Both artifacts bake in the engine objects, so an installed one links standalone
# and the export set stays free of build-only dependencies.
#
# SOURCE is the shim over the engine. The C++ surface has none: the C shims call
# the C++ API too, so its definitions live in admiral_engine.
function(adm_add_surface name)
    cmake_parse_arguments(PARSE_ARGV 1 A "" "SOURCE;VERSION_SCRIPT;EXPORT_NAME" "")

    set(own_objects "")
    if(A_SOURCE)
        # The objects are a target of their own so the API tests can link the code
        # directly instead of queueing behind either artifact's link.
        add_library(${name}_objects OBJECT ${A_SOURCE})
        set_property(TARGET ${name}_objects PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_compile_features(${name}_objects PRIVATE cxx_std_20)
        target_include_directories(${name}_objects
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${ADM_GENERATED_INCLUDE_DIR})
        # Catalog sizes route to codelet_dispatch, and the shims build plan<T>,
        # hence thread_pool and std::jthread.
        target_link_libraries(${name}_objects PRIVATE
            xsimd poet::poet admiral_codelets ${ADM_THREADS_LIB})
        adm_apply_build_profile(${name}_objects)
        set(own_objects $<TARGET_OBJECTS:${name}_objects>)
    endif()

    add_library(${name} SHARED
        ${own_objects}
        $<TARGET_OBJECTS:admiral_engine>)
    # The archive lists the codelet objects: a PRIVATE link would leave a consumer
    # of the .a to find an internal admiral target itself.
    add_library(${name}_static STATIC
        ${own_objects}
        $<TARGET_OBJECTS:admiral_engine>
        $<TARGET_OBJECTS:admiral_codelets>)

    foreach(lib ${name} ${name}_static)
        target_include_directories(${lib} PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        # admiral.hpp uses concepts and std::span, so a consumer needs C++20 even
        # though the library is already compiled. Harmless for the C surfaces.
        target_compile_features(${lib} PUBLIC cxx_std_20)
    endforeach()

    # Shared links the archive rather than listing its objects, so the linker can
    # still drop unused members. The archive takes threads PUBLIC, because nothing
    # bakes libpthread into a .a — the consumer links it.
    target_link_libraries(${name} PRIVATE admiral_codelets ${ADM_THREADS_LIB})
    target_link_libraries(${name}_static PUBLIC ${ADM_THREADS_LIB})

    add_library(admiral::${A_EXPORT_NAME} ALIAS ${name})
    add_library(admiral::${A_EXPORT_NAME}_static ALIAS ${name}_static)
    set_target_properties(${name} PROPERTIES EXPORT_NAME ${A_EXPORT_NAME})
    set_target_properties(${name}_static PROPERTIES
        EXPORT_NAME ${A_EXPORT_NAME}_static
        OUTPUT_NAME ${name})

    # Shared only — a static archive has no export table to restrict.
    adm_restrict_exports(${name} ${A_VERSION_SCRIPT})
endfunction()
