cmake_minimum_required(VERSION 3.25)
# A C-only consumer of the shared C targets. It never enables C++, so the shared libraries must
# carry their own C++ runtime. A static twin fails to link here.
project(admiral_c_only_test C)

find_package(admiral REQUIRED CONFIG)
# The driver reads this line to check that the route came from where it was asked to.
message(STATUS "adm_package: route=c_only admiral_dir=${admiral_DIR} source_dir=${admiral_SOURCE_DIR}")
enable_testing()
include(${CMAKE_CURRENT_LIST_DIR}/stage_dlls.cmake)

find_library(ADM_LIBM m)
if(NOT ADM_LIBM)
    set(ADM_LIBM "")
endif()

foreach(row "use_c;use_c.c;admiral_c" "use_fftw;use_fftw.c;fftw")
    list(GET row 0 stem)
    list(GET row 1 src)
    list(GET row 2 link)
    add_executable(${stem}_${link} ${src})
    target_link_libraries(${stem}_${link} PRIVATE admiral::${link} ${ADM_LIBM})
    adm_stage_dlls(${stem}_${link})
    add_test(NAME ${stem}_${link} COMMAND ${stem}_${link})
endforeach()
