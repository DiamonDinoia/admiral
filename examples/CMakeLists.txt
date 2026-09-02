
set(ADM_EXAMPLES
    quickstart          admiral
    real_transforms     admiral
    r2r_dct             admiral
    axis_transform      admiral
    strided_lines       admiral
    c_api               admiral_c
    fftw_dropin         fftw
)

# The C examples call libm. It is a separate library on ELF and part of the CRT on Windows,
# where linking a literal `m` fails as `lld-link: could not open 'm.lib'`.
find_library(ADM_LIBM m)
if(NOT ADM_LIBM)
    set(ADM_LIBM "")
endif()

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
        target_link_libraries(example_${name} PRIVATE ${ADM_LIBM})
    endif()
    add_test(NAME example_${name} COMMAND example_${name})
    # Windows resolves a DLL through PATH at load time, not through the link-time path, so an
    # example linked to a shared surface exits 0xc0000135 (STATUS_DLL_NOT_FOUND) without this.
    # Harmless on ELF, where the RPATH already answers the question.
    set_tests_properties(example_${name} PROPERTIES ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:admiral::${surface}>")
endforeach()
