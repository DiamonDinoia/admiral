# benchmark/CMakeLists.txt
# Benchmarks using ducc0

# One benchmark, four TUs: bench_nd.cpp and bench_pass_{f,d}.cpp are compile-time splits,
# not separate benchmarks. bench_nd.cpp holds the N-D/r2c compares (the nd_plan ->
# col_dif_dispatch chain at both precisions); bench_pass instantiates dif_pass /
# dif_pass_last for the whole radix set, so it splits again per precision. Shared
# harness in bench_harness.hpp.
add_executable(admiral_benchmark
    bench_pass_f.cpp
    bench_pass_d.cpp
    bench_fft.cpp
    bench_nd.cpp
)

# Link against: admiral_internal, ducc::ducc0 (reference), nanobench (timing).
# Threads::Threads is unconditional even at ADM_ENABLE_THREADS=OFF: ducc0's
# threading.cc is compiled either way, and the link misses pthread_* on toolchains
# whose libc predates the 2.34 merge.
find_package(Threads REQUIRED)
target_link_libraries(admiral_benchmark
    PRIVATE
        admiral_internal
        ducc::ducc0
        nanobench
        Threads::Threads
)

# Third-party headers as SYSTEM so they cannot trip our -Werror profile.
foreach(dep ducc::ducc0 nanobench)
    get_target_property(dep_includes ${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
        target_include_directories(admiral_benchmark SYSTEM PRIVATE ${dep_includes})
    endif()
endforeach()

# Optional FFTW reference path (-DADM_BENCH_FFTW=ON). The option and the
# PkgConfig::FFTW imported target are defined in cmake/Dependencies.cmake.
if(ADM_BENCH_FFTW)
    target_link_libraries(admiral_benchmark PRIVATE PkgConfig::FFTW)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_FFTW)
endif()

# Multi-threaded benchmarking (-DADM_BENCH_THREADS=ON, defined in Dependencies.cmake):
# gates the FFTW threads init/plan_with_nthreads calls and links the fftw3_threads
# companion libs (found by name in Dependencies.cmake).
if(ADM_BENCH_THREADS)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_THREADS)
    if(ADM_BENCH_FFTW)
        target_link_libraries(admiral_benchmark PRIVATE ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB})
    endif()
endif()

# The same arch/fast-math flags as the library, from the one target that defines them.
# Deliberately not adm_apply_build_profile — benchmarks never link sanitizers. Fast
# math is named explicitly: the benchmark is a timing harness, and dropping it would
# change what every number means; tests deliberately do NOT get it.
target_link_libraries(admiral_benchmark PRIVATE admiral_optimization_flags)
if(TARGET admiral_fast_math_flags)
    target_link_libraries(admiral_benchmark PRIVATE admiral_fast_math_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_benchmark PRIVATE project_warnings)
endif()

# LTO is not set per target: ADM_ENABLE_LTO drives
# CMAKE_INTERPROCEDURAL_OPTIMIZATION_<CONFIG>, which takes precedence over the
# plain INTERPROCEDURAL_OPTIMIZATION property.
