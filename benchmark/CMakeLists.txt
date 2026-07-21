# benchmark/CMakeLists.txt
# Benchmarks using ducc0

# Create benchmark executable
add_executable(admiral_benchmark
    bench_fft.cpp
)

# Link against:
# - admiral target (our main playground target)
# - ducc::ducc0 (for FFT benchmarking)
# - nanobench (robust median/noise microbenchmark timing)
target_link_libraries(admiral_benchmark
    PRIVATE
        admiral::admiral
        ducc::ducc0
        nanobench
)

# Mark ducc0 includes as SYSTEM to suppress warnings
get_target_property(DUCC0_INCLUDE_DIRS ducc::ducc0 INTERFACE_INCLUDE_DIRECTORIES)
if(DUCC0_INCLUDE_DIRS)
    target_include_directories(admiral_benchmark SYSTEM PRIVATE ${DUCC0_INCLUDE_DIRS})
endif()

# Mark nanobench includes as SYSTEM so its header does not trip our -Werror profile.
get_target_property(NANOBENCH_INCLUDE_DIRS nanobench INTERFACE_INCLUDE_DIRECTORIES)
if(NANOBENCH_INCLUDE_DIRS)
    target_include_directories(admiral_benchmark SYSTEM PRIVATE ${NANOBENCH_INCLUDE_DIRS})
endif()

# Optional FFTW reference path (-DFFT_BENCH_FFTW=ON). The option and the
# PkgConfig::FFTW imported target are defined in cmake/Dependencies.cmake.
if(ADM_BENCH_FFTW)
    target_link_libraries(admiral_benchmark PRIVATE PkgConfig::FFTW)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_FFTW)
endif()

# Multi-threaded benchmarking (-DFFT_BENCH_THREADS=ON, defined in Dependencies.cmake):
# ducc0 is already linked against Threads there; here it gates the FFTW threads
# init/plan_with_nthreads calls and links the fftw3_threads companion libs (found
# by name in Dependencies.cmake; shared .so, so link order vs the base fftw3 lib
# doesn't matter — bench_fft.o references both directly).
if(ADM_BENCH_THREADS)
    target_compile_definitions(admiral_benchmark PRIVATE ADM_BENCH_THREADS)
    if(ADM_BENCH_FFTW)
        target_link_libraries(admiral_benchmark PRIVATE ${FFTW3_THREADS_LIB} ${FFTW3F_THREADS_LIB})
    endif()
endif()

# if gcc or clang enable -march=native (opt-in), fast-math for benchmarks
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(ADM_USE_NATIVE_ARCH)
        target_compile_options(admiral_benchmark PRIVATE -march=native)
    endif()
    target_compile_options(admiral_benchmark PRIVATE -ffast-math)
endif()

set_target_properties(admiral_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)

# Apply project warnings to benchmarks
if(TARGET project_warnings)
    target_link_libraries(admiral_benchmark PRIVATE project_warnings)
endif()

# Note: Benchmarks should NOT use ASAN (performance impact)
# So we don't link project_sanitizers here
