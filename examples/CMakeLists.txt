
set(ADM_EXAMPLES
    quickstart          admiral
    real_transforms     admiral
    r2r_dct             admiral
    axis_transform      admiral
    strided_lines       admiral
    c_api               admiral_c
    fftw_dropin         fftw
)

list(LENGTH ADM_EXAMPLES ADM_NEX)
math(EXPR ADM_LAST "${ADM_NEX} - 1")
foreach(i RANGE 0 ${ADM_LAST} 2)
    math(EXPR j "${i} + 1")
    list(GET ADM_EXAMPLES ${i} name)
    list(GET ADM_EXAMPLES ${j} surface)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.c)
        add_executable(example_${name} ${name}.c)
    else()
        add_executable(example_${name} ${name}.cpp)
    endif()
    target_link_libraries(example_${name} PRIVATE admiral::${surface})
    if(TARGET project_sanitizers)
        target_link_libraries(example_${name} PRIVATE project_sanitizers)
    endif()
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.c)
        target_link_libraries(example_${name} PRIVATE m)
    endif()
    add_test(NAME example_${name} COMMAND example_${name})
endforeach()
