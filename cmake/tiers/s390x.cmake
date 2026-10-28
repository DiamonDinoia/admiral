# s390x: one VXE build. xsimd needs __VEC__ >= 10304 and __ARCH__ >= 12 (z14 or later), and
# __VEC__ needs -mzvector. ADM_TARGET_ARCH=none keeps the scalar z13 build.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^s390x$" AND NOT ADM_TARGET_ARCH STREQUAL "none"
   AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(admiral_optimization_flags INTERFACE -mzvector)
endif()
