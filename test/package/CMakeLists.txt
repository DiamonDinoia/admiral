cmake_minimum_required(VERSION 3.25)
# OFF makes a C-only project with only the C consumers.
option(ADM_PACKAGE_CXX "Enable C++ in the consumer" ON)
project(admiral_package_test C)
if(ADM_PACKAGE_CXX)
    enable_language(CXX)
endif()

# One set of sources, seven ways to get admiral. ADM_PACKAGE_ROUTE selects the way. Every route
# must produce the same admiral::* targets. run_routes.sh sets the route and its inputs.
#   prefix_shared, prefix_static  find_package on an installed prefix (CMAKE_PREFIX_PATH), one linkage
#   prefix                        the same, both linkages
#   subdir                        add_subdirectory of a source tree placed at ./admiral
#   fetch_git                     FetchContent from GitHub
#   fetch_dir                     the same lines, with FETCHCONTENT_SOURCE_DIR_ADMIRAL set
#   cpm                           CPMAddPackage from GitHub, with the CPM.cmake of ADM_PACKAGE_SRC
# The git routes name GitHub. The driver redirects that URL to the commit under test.
set(ADM_PACKAGE_ROUTE prefix CACHE STRING "How the consumer gets admiral")
set(ADM_PACKAGE_LINKS shared static)
if(ADM_PACKAGE_ROUTE MATCHES "_(shared|static)$")
    set(ADM_PACKAGE_LINKS ${CMAKE_MATCH_1})
endif()

if(ADM_PACKAGE_ROUTE MATCHES "^prefix")
    find_package(admiral REQUIRED CONFIG)
elseif(ADM_PACKAGE_ROUTE STREQUAL "subdir")
    add_subdirectory(admiral)
elseif(ADM_PACKAGE_ROUTE MATCHES "^fetch_")
    include(FetchContent)
    FetchContent_Declare(admiral GIT_REPOSITORY https://github.com/DiamonDinoia/admiral.git GIT_TAG master)
    FetchContent_MakeAvailable(admiral)
elseif(ADM_PACKAGE_ROUTE STREQUAL "cpm")
    include(${ADM_PACKAGE_SRC}/cmake/CPM.cmake)
    CPMAddPackage("gh:DiamonDinoia/admiral#master")
else()
    message(FATAL_ERROR "ADM_PACKAGE_ROUTE=${ADM_PACKAGE_ROUTE} is not a route")
endif()
# The driver reads this line to check that the route came from where it was asked to.
message(STATUS "adm_package: route=${ADM_PACKAGE_ROUTE} admiral_dir=${admiral_DIR} "
               "source_dir=${admiral_SOURCE_DIR}")
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
    if(src MATCHES "\\.cpp$" AND NOT ADM_PACKAGE_CXX)
        continue()
    endif()
    foreach(kind ${ADM_PACKAGE_LINKS})
        set(link ${${kind}_link})
        string(TOUPPER ${kind} KIND)
        get_target_property(type admiral::${link} TYPE)
        if(NOT type STREQUAL ${KIND}_LIBRARY)
            message(FATAL_ERROR "admiral::${link} is ${type}, the ${kind} route needs ${KIND}_LIBRARY")
        endif()
        add_executable(${stem}_${link} ${src})
        target_link_libraries(${stem}_${link} PRIVATE admiral::${link} ${ADM_LIBM})
        adm_stage_dlls(${stem}_${link})
        add_test(NAME ${stem}_${link} COMMAND ${stem}_${link})
    endforeach()
endforeach()
