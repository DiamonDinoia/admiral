include(CheckLinkerFlag)

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

function(adm_restrict_exports target map)
    set(script ${CMAKE_CURRENT_SOURCE_DIR}/${map})
    check_linker_flag(CXX "LINKER:--version-script=${script}" ADM_HAVE_VERSION_SCRIPT)
    if(ADM_HAVE_VERSION_SCRIPT)
        target_link_options(${target} PRIVATE "LINKER:--version-script=${script}")
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${script})
    endif()
endfunction()

function(adm_add_surface name)
    cmake_parse_arguments(PARSE_ARGV 1 A "" "SOURCE;VERSION_SCRIPT;EXPORT_NAME" "")

    set(own_objects "")
    if(A_SOURCE)
        add_library(${name}_objects OBJECT ${A_SOURCE})
        set_property(TARGET ${name}_objects PROPERTY POSITION_INDEPENDENT_CODE ON)
        target_compile_features(${name}_objects PRIVATE cxx_std_${ADM_CXX_STANDARD})
        target_include_directories(${name}_objects
            PUBLIC
                $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}
                ${ADM_GENERATED_INCLUDE_DIR})
        target_link_libraries(${name}_objects PRIVATE
            xsimd poet::poet admiral_codelets ${ADM_THREADS_LIB})
        adm_apply_build_profile(${name}_objects)
        set(own_objects $<TARGET_OBJECTS:${name}_objects>)
    endif()

    add_library(${name} SHARED
        ${own_objects}
        $<TARGET_OBJECTS:admiral_engine>)
    add_library(${name}_static STATIC
        ${own_objects}
        $<TARGET_OBJECTS:admiral_engine>
        $<TARGET_OBJECTS:admiral_codelets>)

    foreach(lib ${name} ${name}_static)
        target_include_directories(${lib} PUBLIC
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(${lib} PUBLIC cxx_std_${ADM_CXX_STANDARD})
    endforeach()

    target_link_libraries(${name} PRIVATE admiral_codelets ${ADM_THREADS_LIB})
    target_link_libraries(${name}_static PUBLIC ${ADM_THREADS_LIB})

    add_library(admiral::${A_EXPORT_NAME} ALIAS ${name})
    add_library(admiral::${A_EXPORT_NAME}_static ALIAS ${name}_static)
    set_target_properties(${name} PROPERTIES EXPORT_NAME ${A_EXPORT_NAME})
    set_target_properties(${name}_static PROPERTIES
        EXPORT_NAME ${A_EXPORT_NAME}_static
        OUTPUT_NAME ${name})

    adm_restrict_exports(${name} ${A_VERSION_SCRIPT})
endfunction()
