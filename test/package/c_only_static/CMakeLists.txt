cmake_minimum_required(VERSION 3.25)
# A consumer of the static C target. Without ADM_CONSUMER_CXX it is C-only, and the link must
# fail: the archive is C++ behind a C API. With it the project enables C++, and the link must work.
option(ADM_CONSUMER_CXX "Enable C++ in the consumer" OFF)
set(langs C)
if(ADM_CONSUMER_CXX)
    list(APPEND langs CXX)
endif()
project(admiral_c_only_static_test LANGUAGES ${langs})

find_package(admiral REQUIRED CONFIG)
enable_testing()

find_library(ADM_LIBM m)
if(NOT ADM_LIBM)
    set(ADM_LIBM "")
endif()

add_executable(use_c_static use_c.c)
target_link_libraries(use_c_static PRIVATE admiral::admiral_c_static ${ADM_LIBM})
add_test(NAME use_c_static COMMAND use_c_static)
