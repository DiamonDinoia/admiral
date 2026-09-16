add_executable(admiral_benchmark
    bench_pass_f.cpp
    bench_pass_d.cpp
    bench_fft.cpp
    bench_nd.cpp
    bench_execute_overhead.cpp
)

find_package(Threads REQUIRED)
target_link_libraries(admiral_benchmark
    PRIVATE
        admiral_internal
        ducc::ducc0
        nanobench
        Threads::Threads
)

foreach(dep ducc::ducc0 nanobench)
    get_target_property(dep_includes ${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
        target_include_directories(admiral_benchmark SYSTEM PRIVATE ${dep_includes})
    endif()
endforeach()

if(ADM_BENCH_FFTW)
    target_link_libraries(admiral_benchmark PRIVATE PkgConfig::FFTW)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_FFTW)
endif()

if(ADM_BENCH_THREADS)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_THREADS)
    if(ADM_BENCH_FFTW)
        target_link_libraries(admiral_benchmark PRIVATE ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB})
    endif()
endif()

target_link_libraries(admiral_benchmark PRIVATE admiral_optimization_flags)
if(TARGET admiral_fast_math_flags)
    target_link_libraries(admiral_benchmark PRIVATE admiral_fast_math_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_benchmark PRIVATE project_warnings)
endif()

# bench_granule_ab stays its OWN executable (WI-0c-ii): the granule-admission arm in
# scripts/validate.sh builds it under both knob states and A/Bs the two binaries, so it
# joins neither the admiral_benchmark registry contract (bench_pass.hpp) nor its
# ducc0/nanobench linkage. It includes src/granule_apply.hpp the way test_granule does:
# admission gates the engine choke points, never the drivers.
add_executable(bench_granule_ab bench_granule_ab.cpp)
target_include_directories(bench_granule_ab PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(bench_granule_ab PRIVATE admiral_internal)
if(TARGET admiral_optimization_flags)
    target_link_libraries(bench_granule_ab PRIVATE admiral_optimization_flags)
endif()
if(TARGET admiral_fast_math_flags)
    target_link_libraries(bench_granule_ab PRIVATE admiral_fast_math_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(bench_granule_ab PRIVATE project_warnings)
endif()

# bench_large_route stays its OWN executable (WI-2c / P-2c-D): the kLargeRoute* re-derivation
# wave runs it stand-alone on cluster nodes, so it joins neither the admiral_benchmark
# registry contract (bench_pass.hpp) nor its ducc0/nanobench linkage. It includes
# detail/plan.hpp the way test_route_forced.cpp does: the forced-route plan_impl constructor
# is the hook, and admission gates the engine chooser, never the drivers.
add_executable(bench_large_route bench_large_route.cpp)
target_link_libraries(bench_large_route PRIVATE admiral_internal)
if(TARGET admiral_optimization_flags)
    target_link_libraries(bench_large_route PRIVATE admiral_optimization_flags)
endif()
if(TARGET admiral_fast_math_flags)
    target_link_libraries(bench_large_route PRIVATE admiral_fast_math_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(bench_large_route PRIVATE project_warnings)
endif()

# Identity baked into the binary's stderr header; the header's consumers are per-host class
# receipts, so the sha and the arch string must survive a copied binary.
find_package(Git QUIET)
set(_blr_sha "unknown")
if(Git_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --always --dirty
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE _blr_sha_out
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_blr_sha_out)
        set(_blr_sha "${_blr_sha_out}")
    endif()
endif()
target_compile_definitions(bench_large_route PRIVATE
    ADM_BENCH_SHA="${_blr_sha}"
    ADM_BENCH_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    ADM_BENCH_ARCH="${ADM_TARGET_ARCH}"
    ADM_BENCH_CXXSTD="${ADM_CXX_STANDARD}")
