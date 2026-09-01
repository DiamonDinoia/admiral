add_executable(admiral_benchmark
    bench_pass_f.cpp
    bench_pass_d.cpp
    bench_fft.cpp
    bench_nd.cpp
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
