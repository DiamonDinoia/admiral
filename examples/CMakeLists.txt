# Runnable examples. Each is registered as a ctest test, so every snippet the
# docs show is compiled and executed; a snippet that drifts from the API breaks
# the build instead of lying to the reader.
#
# The C++ examples link the C++ target and compile as C++20. The C examples
# compile as plain C and link the C and FFTW shims, doubling as the C-only
# consumer check. In-tree target names match the installed admiral:: ones, so
# the sources read like consumer code.

# example -> library it links.
set(ADM_EXAMPLES
    quickstart          admiral
    real_transforms     admiral
    r2r_dct             admiral
    axis_transform      admiral
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
    # Consumer code: no project_warnings. The C examples additionally link libm
    # only for their own sin/fabs, not because the shims need it.
    target_link_libraries(example_${name} PRIVATE admiral::${surface})
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${name}.c)
        target_link_libraries(example_${name} PRIVATE m)
    endif()
    add_test(NAME example_${name} COMMAND example_${name})
endforeach()
