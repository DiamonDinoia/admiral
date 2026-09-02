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

foreach(link admiral admiral_static)
    add_executable(use_cxx_${link} use_cxx.cpp)
    target_link_libraries(use_cxx_${link} PRIVATE admiral::${link})
    add_test(NAME use_cxx_${link} COMMAND use_cxx_${link})
endforeach()

foreach(link admiral_c admiral_c_static)
    add_executable(use_c_${link} use_c.c)
    target_link_libraries(use_c_${link} PRIVATE admiral::${link} ${ADM_LIBM})
    add_test(NAME use_c_${link} COMMAND use_c_${link})
endforeach()

foreach(link fftw fftw_static)
    add_executable(use_fftw_${link} use_fftw.c)
    target_link_libraries(use_fftw_${link} PRIVATE admiral::${link} ${ADM_LIBM})
    add_test(NAME use_fftw_${link} COMMAND use_fftw_${link})
endforeach()
