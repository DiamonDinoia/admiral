# One benchmark, four TUs: a compile-time split, not separate benchmarks.
# `bench_nd.cpp` holds the N-D/r2c compares (`nd_plan` -> `col_dif_dispatch` at both
# precisions); `bench_pass_{f,d}.cpp` instantiates `dif_pass` / `dif_pass_last` for
# the whole radix set, split per precision. Shared harness in `bench_harness.hpp`.
add_executable(admiral_benchmark
    bench_pass_f.cpp
    bench_pass_d.cpp
    bench_fft.cpp
    bench_nd.cpp
)

# `Threads::Threads` is unconditional even at `ADM_ENABLE_THREADS=OFF`: the build
# compiles `threading.cc` from `ducc0` either way, and the link misses `pthread_*` on
# libc before 2.34.
find_package(Threads REQUIRED)
target_link_libraries(admiral_benchmark
    PRIVATE
        admiral_internal
        ducc::ducc0
        nanobench
        Threads::Threads
)

# Third-party headers as `SYSTEM` so they cannot trip the project's `-Werror` profile.
foreach(dep ducc::ducc0 nanobench)
    get_target_property(dep_includes ${dep} INTERFACE_INCLUDE_DIRECTORIES)
    if(dep_includes)
        target_include_directories(admiral_benchmark SYSTEM PRIVATE ${dep_includes})
    endif()
endforeach()

# Optional FFTW reference (`-DADM_BENCH_FFTW=ON`); `cmake/Dependencies.cmake` defines
# the option and `PkgConfig::FFTW`.
if(ADM_BENCH_FFTW)
    target_link_libraries(admiral_benchmark PRIVATE PkgConfig::FFTW)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_FFTW)
endif()

# `-DADM_BENCH_THREADS=ON` (defined in `cmake/Dependencies.cmake`) gates the FFTW
# `plan_with_nthreads` calls and links the `fftw3_threads` companion libs.
if(ADM_BENCH_THREADS)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_THREADS)
    if(ADM_BENCH_FFTW)
        target_link_libraries(admiral_benchmark PRIVATE ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB})
    endif()
endif()

# The same arch/fast-math flags as the library, from the one target that defines them.
# Not `adm_apply_build_profile`: benchmarks never link sanitizers. Fast math is named
# on purpose: the benchmark is a timing harness, and tests deliberately do NOT get
# fast math.
target_link_libraries(admiral_benchmark PRIVATE admiral_optimization_flags)
if(TARGET admiral_fast_math_flags)
    target_link_libraries(admiral_benchmark PRIVATE admiral_fast_math_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_benchmark PRIVATE project_warnings)
endif()

# No per-target LTO: `ADM_ENABLE_LTO` drives
# `CMAKE_INTERPROCEDURAL_OPTIMIZATION_<CONFIG>`, which overrides the plain property.
