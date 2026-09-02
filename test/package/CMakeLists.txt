cmake_minimum_required(VERSION 3.25)
project(admiral_package_test C CXX)

# Two consumption paths, one set of sources. ADM_PACKAGE_TEST_FETCH points at a source tree and
# selects the FetchContent path; unset, the project consumes an installed prefix through
# find_package. Both must produce the same admiral::* targets.
if(ADM_PACKAGE_TEST_FETCH)
    include(FetchContent)
    FetchContent_Declare(admiral SOURCE_DIR ${ADM_PACKAGE_TEST_FETCH})
    FetchContent_MakeAvailable(admiral)
else()
    find_package(admiral REQUIRED)
endif()
enable_testing()

find_library(ADM_LIBM m)
if(NOT ADM_LIBM)
    set(ADM_LIBM "")
endif()

# Windows resolves a DLL from the executable's own directory, so a consumer that links the
# shared surface needs it copied there before ctest runs.
function(adm_stage_dlls target)
    if(WIN32)
        # The list is empty for the static links, and `copy` with no source is an error, so the
        # command itself is selected by whether there is anything to copy.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
                    $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${target}>>,copy,true>
                    $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()

foreach(row "use_cxx;use_cxx.cpp;admiral;admiral_static"
            "use_c;use_c.c;admiral_c;admiral_c_static"
            "use_fftw;use_fftw.c;fftw;fftw_static")
    list(GET row 0 stem)
    list(GET row 1 src)
    list(GET row 2 shared_link)
    list(GET row 3 static_link)
    foreach(link ${shared_link} ${static_link})
        add_executable(${stem}_${link} ${src})
        target_link_libraries(${stem}_${link} PRIVATE admiral::${link} ${ADM_LIBM})
        adm_stage_dlls(${stem}_${link})
        add_test(NAME ${stem}_${link} COMMAND ${stem}_${link})
    endforeach()
endforeach()
