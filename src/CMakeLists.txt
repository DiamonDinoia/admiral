# src/CMakeLists.txt
# Main FFT library targets

# ============================================================================
# Compiled codelet library (one TU per explicit catalog size)
# ============================================================================
#
# kernel<N> straight-line codelets (radix-31/61 ≈ P² unrolled, ×4
# {float,double}×{fwd,inv}) are heavy to instantiate. Compiling them once here,
# one small TU per N, keeps consumer TUs free of that cost (they see only the
# narrow codelet_dispatch<T> declaration + extern template in kernels.hpp).
# One TU per N maximizes parallel compile and bounds peak memory per TU.
#
# STATIC (not OBJECT): admiral_tests links BOTH admiral::admiral and admiral::admiral_c; a pure
# OBJECT lib attached to both would inject its objects twice into that one
# executable (duplicate codelet_dispatch<T>). A static archive is pulled once.

# Generate the explicit catalog from a bounded integer range minus measured
# exclusions. This keeps the build knob simple (min/max), still emits one TU per
# selected size, and still generates an exact C++ dispatch sequence.
set(ADM_CODELET_MIN_N 2 CACHE STRING
    "Smallest N considered for compiled straight-line codelets")
set(ADM_CODELET_MAX_N 64 CACHE STRING
    "Largest N considered for compiled straight-line codelets")
# Full spill-free catalog [2..64]: every composite is a small-radix combination
# and every prime > 11 is a Rader chiplet (length-(p-1) convolution via the
# smaller codelets), so no size degrades to a flat O(N^2) unroll. All catalog
# sizes are verified vs a reference DFT (`admiral_benchmark --verify`) and beat ducc0
# on a pinned cycle-true --compare (primes 0.21-0.57x; one marginal composite,
# N=52, is ~1.13x pending an optimal-decomposition pass).
set(ADM_CODELET_EXCLUDE_SIZES ""
    CACHE STRING
    "Semicolon-separated sizes excluded from the generated codelet catalog")

# AddressSanitizer instrumentation of the full straight-line catalog OOMs the
# compiler (tens of GB per TU on the larger Rader codelets — see memory
# asan-full-catalog-oom). Build only a small catalog under ASan; large-codelet
# correctness is covered by the Release `--verify` sweep, not the ASan build.
if(ADM_ENABLE_ASAN)
    set(ADM_CODELET_MAX_N 16)
endif()

if(ADM_CODELET_MIN_N LESS 2)
    message(FATAL_ERROR "ADM_CODELET_MIN_N must be >= 2")
endif()
if(ADM_CODELET_MAX_N LESS ADM_CODELET_MIN_N)
    message(FATAL_ERROR "ADM_CODELET_MAX_N must be >= ADM_CODELET_MIN_N")
endif()

set(ADM_CODELET_EXTRA_SIZES "120"
    CACHE STRING
    "Semicolon-separated sizes added to the catalog beyond [MIN_N, MAX_N]")
if(ADM_ENABLE_ASAN)
    set(ADM_CODELET_EXTRA_SIZES "")  # keep the ASan catalog minimal (see MAX_N above)
endif()

# The contiguous [MIN, MAX] range drives poet::dispatch (a dense jump table —
# appending the sparse extras to the sequence degrades it to a compare chain,
# measured +12-14 cyc on EVERY codelet transform). Extras are dispatched by an
# explicit generated if-chain ahead of the table (see codelet_dispatch.cpp.in).
set(ADM_CODELET_RANGE_SIZES "")
foreach(CODELET_N RANGE ${ADM_CODELET_MIN_N} ${ADM_CODELET_MAX_N})
    if(NOT CODELET_N IN_LIST ADM_CODELET_EXCLUDE_SIZES)
        list(APPEND ADM_CODELET_RANGE_SIZES ${CODELET_N})
    endif()
endforeach()

set(ADM_CODELET_SIZES ${ADM_CODELET_RANGE_SIZES} ${ADM_CODELET_EXTRA_SIZES})
list(REMOVE_DUPLICATES ADM_CODELET_SIZES)
list(SORT ADM_CODELET_SIZES COMPARE NATURAL)
if(NOT ADM_CODELET_SIZES)
    message(FATAL_ERROR "ADM_CODELET_SIZES must contain at least one size")
endif()
foreach(CODELET_N IN LISTS ADM_CODELET_SIZES)
    if(CODELET_N LESS 2)
        message(FATAL_ERROR "ADM_CODELET_SIZES entries must be >= 2: ${CODELET_N}")
    endif()
endforeach()
list(GET ADM_CODELET_SIZES -1 ADM_CODELET_MAX)
list(LENGTH ADM_CODELET_SIZES ADM_CODELET_COUNT)
string(REPLACE ";" ", " ADM_CODELET_SIZE_CSV "${ADM_CODELET_SIZES}")
string(REPLACE ";" ", " ADM_CODELET_RANGE_CSV "${ADM_CODELET_RANGE_SIZES}")
set(ADM_CODELET_SEQUENCE "std::integer_sequence<int, ${ADM_CODELET_RANGE_CSV}>")

# Explicit dispatch lines for the extras (empty when there are none).
set(ADM_CODELET_EXTRA_DISPATCH "")
foreach(CODELET_N IN LISTS ADM_CODELET_EXTRA_SIZES)
    if(NOT CODELET_N IN_LIST ADM_CODELET_RANGE_SIZES)
        string(APPEND ADM_CODELET_EXTRA_DISPATCH
            "    if (N == ${CODELET_N}) {\n"
            "        codelet_apply<${CODELET_N}, T, Forward>(in, out);\n"
            "        return;\n"
            "    }\n")
    endif()
endforeach()

# Generate the header constants from the same ADM_CODELET_SIZES (single source
# of truth shared with kernels.hpp). Lives in a build-tree include dir added
# to every target that pulls in kernels.hpp.
set(ADM_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_max.hpp.in
    ${ADM_GENERATED_INCLUDE_DIR}/admiral/detail/codelet_max.hpp
    @ONLY)

# One TU per catalog N (codelet_instance.cpp.in), plus a single dispatch TU whose
# extern-template block is generated here (no C preprocessor macros).
set(ADM_CODELET_GENERATED "")
set(ADM_CODELET_EXTERN "")
foreach(CODELET_N IN LISTS ADM_CODELET_SIZES)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/codelet_instance.cpp.in
        ${CMAKE_CURRENT_BINARY_DIR}/codelet_${CODELET_N}.cpp
        @ONLY)
    list(APPEND ADM_CODELET_GENERATED
         ${CMAKE_CURRENT_BINARY_DIR}/codelet_${CODELET_N}.cpp)
    string(APPEND ADM_CODELET_EXTERN
        "extern template void codelet_apply<${CODELET_N}, float,  true >(const std::complex<float>*, std::complex<float>*);\n"
        "extern template void codelet_apply<${CODELET_N}, float,  false>(const std::complex<float>*, std::complex<float>*);\n"
        "extern template void codelet_apply<${CODELET_N}, double, true >(const std::complex<double>*, std::complex<double>*);\n"
        "extern template void codelet_apply<${CODELET_N}, double, false>(const std::complex<double>*, std::complex<double>*);\n")
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_dispatch.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp
    @ONLY)

add_library(admiral_codelets STATIC
    ${ADM_CODELET_GENERATED}
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp
)

target_compile_features(admiral_codelets PUBLIC cxx_std_20)

target_include_directories(admiral_codelets PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}          # codelet_apply.hpp
    ${ADM_GENERATED_INCLUDE_DIR}         # admiral/detail/codelet_max.hpp
)

# Arch flags (-march=native/FMA) are essential — without them the codelets fall
# back to SSE2 and are slow. xsimd + poet are header-only build deps.
target_link_libraries(admiral_codelets PUBLIC xsimd poet::poet)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_codelets PUBLIC admiral_optimization_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_codelets PRIVATE project_warnings)
endif()
if(ADM_ENABLE_ASAN AND TARGET project_sanitizers)
    target_link_libraries(admiral_codelets PRIVATE project_sanitizers)
endif()

# xsimd.hpp is ~97% of each codelet TU's parse cost (~1.2s / 273 MiB, measured);
# the kernel<N> instantiation is <12%. Parsing it once via a PCH cuts per-TU
# compile ~9x (1.74s->0.20s) and peak RSS ~2x (273->136 MiB, the .gch is mmap'd).
target_precompile_headers(admiral_codelets PRIVATE
    <xsimd/xsimd.hpp>
    <poet/poet.hpp>)

# ============================================================================
# Engine instantiation object (single site for the heavy route templates)
# ============================================================================
# OBJECT (not STATIC): weak explicit-instantiation symbols never force
# archive-member extraction, so a .a would be silently dropped by the linker.
# Consumers get these objects directly via the admiral / admiral_c link interfaces.
find_package(Threads REQUIRED)
add_library(admiral_engine OBJECT admiral_engine.cpp)
target_compile_features(admiral_engine PRIVATE cxx_std_20)
target_include_directories(admiral_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ADM_GENERATED_INCLUDE_DIR})
target_link_libraries(admiral_engine PRIVATE xsimd poet::poet Threads::Threads)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_engine PRIVATE admiral_optimization_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_engine PRIVATE project_warnings)
endif()
if(ADM_ENABLE_ASAN AND TARGET project_sanitizers)
    target_link_libraries(admiral_engine PRIVATE project_sanitizers)
endif()
# Same xsimd/poet parse-cost amortization as admiral_codelets (this is the one heavy TU).
target_precompile_headers(admiral_engine PRIVATE <xsimd/xsimd.hpp> <poet/poet.hpp>)

# ============================================================================
# Header-only C++ Interface
# ============================================================================

add_library(admiral INTERFACE)
add_library(admiral::admiral ALIAS admiral)

target_include_directories(admiral INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
    $<BUILD_INTERFACE:${ADM_GENERATED_INCLUDE_DIR}>  # admiral/detail/codelet_max.hpp
    $<INSTALL_INTERFACE:include>
)

target_compile_features(admiral INTERFACE cxx_std_20)

# SIMD + compile-time dispatch dependencies (both header-only) + compiled codelets.
# Threads: the opt-in plan-owned worker pool uses std::jthread (thread_pool.hpp),
# so every consumer TU that instantiates plan<T> needs pthread linked.
find_package(Threads REQUIRED)
target_link_libraries(admiral INTERFACE xsimd poet::poet admiral_codelets Threads::Threads)
# The single engine instantiation, propagated as objects to every consumer of
# admiral::admiral (reliable across linkers; OBJECT-lib link propagation through an
# INTERFACE target is not). This is what makes the extern-template decls resolve.
target_sources(admiral INTERFACE $<TARGET_OBJECTS:admiral_engine>)

if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral INTERFACE admiral_optimization_flags)
endif()

# ============================================================================
# Compiled C API Library
# ============================================================================

# Self-contained SHARED library: bakes in the engine objects and absorbs the
# static codelet archive, so the installed artifact links standalone. Hidden
# visibility (project default) exports only the ADM_C_API C symbols; xsimd/poet/
# codelets stay internal, keeping the install export free of build-only deps.
add_library(admiral_c SHARED
    c_api.cpp
    $<TARGET_OBJECTS:admiral_engine>
)

add_library(admiral::admiral_c ALIAS admiral_c)

target_include_directories(admiral_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ADM_GENERATED_INCLUDE_DIR}     # admiral/detail/codelet_max.hpp
)

target_compile_features(admiral_c PRIVATE cxx_std_20)

# c_api.cpp routes catalog sizes to codelet_dispatch -> needs admiral_codelets.
# Threads: c_api.cpp includes admiral.hpp (plan<T> -> thread_pool -> std::jthread).
target_link_libraries(admiral_c PRIVATE xsimd poet::poet admiral_codelets Threads::Threads)

if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_c PRIVATE admiral_optimization_flags)
endif()

if(TARGET project_warnings)
    target_link_libraries(admiral_c PRIVATE project_warnings)
endif()

# Apply sanitizers if enabled
if(ADM_ENABLE_ASAN AND TARGET project_sanitizers)
    target_link_libraries(admiral_c PRIVATE project_sanitizers)
endif()

# ============================================================================
# FFTW3 Compatibility Layer
# ============================================================================
# Drop-in fftw3.h surface (core complex-DFT plan API) over admiral::plan<T>.
# Self-contained SHARED library, same rationale as admiral_c above.
add_library(admiral_fftw SHARED
    fftw_compat.cpp
    $<TARGET_OBJECTS:admiral_engine>
)

add_library(admiral::fftw ALIAS admiral_fftw)
set_target_properties(admiral_fftw PROPERTIES EXPORT_NAME fftw)  # installed as admiral::fftw, not admiral::admiral_fftw

target_include_directories(admiral_fftw
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ADM_GENERATED_INCLUDE_DIR}
)

target_compile_features(admiral_fftw PRIVATE cxx_std_20)
target_link_libraries(admiral_fftw PRIVATE xsimd poet::poet admiral_codelets Threads::Threads)
if(TARGET admiral_optimization_flags)
    target_link_libraries(admiral_fftw PRIVATE admiral_optimization_flags)
endif()
if(TARGET project_warnings)
    target_link_libraries(admiral_fftw PRIVATE project_warnings)
endif()
if(ADM_ENABLE_ASAN AND TARGET project_sanitizers)
    target_link_libraries(admiral_fftw PRIVATE project_sanitizers)
endif()

# ============================================================================
# Install / export: find_package(admiral) -> admiral::admiral_c, admiral::fftw
# ============================================================================
# Only the two self-contained SHARED libs are exported. admiral::admiral is header-only
# and would drag xsimd/poet into the export set, so it stays build-tree only.
if(IS_MAIN_PROJECT)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    install(TARGETS admiral_c admiral_fftw EXPORT admiralTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/admiral
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
    install(FILES ${ADM_GENERATED_INCLUDE_DIR}/admiral/detail/codelet_max.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/admiral/detail)

    install(EXPORT admiralTargets NAMESPACE admiral::
        FILE admiralTargets.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/admiralConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/admiralConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/fftConfigVersion.cmake
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/admiralConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/fftConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/admiral)
endif()
