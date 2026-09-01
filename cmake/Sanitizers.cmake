
if(ADM_SANITIZER STREQUAL "none")
    return()
endif()

set(adm_sanitizers address undefined address+undefined thread)
if(NOT ADM_SANITIZER IN_LIST adm_sanitizers)
    message(FATAL_ERROR "ADM_SANITIZER=${ADM_SANITIZER}; expected one of: none;${adm_sanitizers}")
endif()
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "ADM_SANITIZER needs GCC or Clang, not ${CMAKE_CXX_COMPILER_ID}")
endif()

string(REPLACE "+" "," adm_san_arg "${ADM_SANITIZER}")
add_library(project_sanitizers INTERFACE)
target_compile_options(project_sanitizers INTERFACE
    -fsanitize=${adm_san_arg} -fno-omit-frame-pointer -g)
target_link_options(project_sanitizers INTERFACE -fsanitize=${adm_san_arg})
message(STATUS "Sanitizers: ${adm_san_arg}")
