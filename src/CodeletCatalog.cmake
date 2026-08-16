# Which straight-line codelet sizes get compiled, and the sources generated for
# them. Included by src/CMakeLists.txt before the targets are declared.
#
# kernel<N> codelets (radix-31/61 are ~P^2 unrolled, times {float,double}) are
# heavy to instantiate. Only ONE direction is ever built: codelet_apply<N,T,false>
# conjugates in and out around kernel<N,T,true>, so {fwd,inv} multiplies the thin
# boundary loops, not the unrolled body. Compiling them here, one small TU per N,
# keeps consumer TUs free of that cost — they see only the narrow
# codelet_dispatch<T> declaration plus the extern template in math.hpp. One TU
# per N maximizes parallel compile and bounds peak memory per TU.
#
# Outputs, all consumed by src/CMakeLists.txt:
#   ADM_GENERATED_INCLUDE_DIR   build-tree include dir for the generated headers
#   ADM_CODELET_GENERATED       one .cpp per catalog size, plus the dispatch TU
#   ADM_THREADS_LIB             Threads::Threads, or empty

# The catalog is a bounded range minus measured exclusions: one simple knob pair,
# still one TU per selected size, still an exact C++ dispatch sequence.
set(ADM_CODELET_MIN_N 2 CACHE STRING
    "Smallest N considered for compiled straight-line codelets")
set(ADM_CODELET_MAX_N 64 CACHE STRING
    "Largest N considered for compiled straight-line codelets")
# [2..64] is spill-free throughout: every composite is a small-radix combination
# and every prime > 11 is a Rader chiplet (a length-(p-1) convolution over the
# smaller codelets), so no size degrades to a flat O(N^2) unroll. Every catalog
# size is checked against a reference DFT (`admiral_benchmark --verify`) and beats
# ducc0 on a pinned cycle-true --compare.
set(ADM_CODELET_EXCLUDE_SIZES "" CACHE STRING
    "Semicolon-separated sizes excluded from the generated codelet catalog")
set(ADM_CODELET_EXTRA_SIZES "120;65;85;143;100;360" CACHE STRING
    "Semicolon-separated sizes added to the catalog beyond [MIN_N, MAX_N]")
# 65=5*13, 85=5*17 and 143=11*13 have no other engine. The DIF tape admits a prime
# above 11 only as a MIDDLE pass (dif_generic_radix_seq), and a two-factor size has
# no middle slot, so each one falls through the whole ranking to Bluestein and pays
# a 7-smooth pad: 65->135, 85->175, 143->288. A catalog codelet peels the small
# factor and batches the 13 or 17 cofactor, which the model prices well below Bluestein.
#
# 100=4*25 has an engine but not its cheapest one. good_thomas needs a DFT-25 and its
# butterfly set stops at 8, and 4*25 is the only coprime split, so the ranking drops to
# iterative_dif with radices 10,10. A codelet is cheaper.
#
# 360=8*9*5 likewise falls to iterative_dif, which the model over-prices, so no cheaper
# engine is ever reached. The codelet peels the 8 and batches the 45 cofactor, filling
# every f32 lane.

# Sanitizer instrumentation costs several GB per TU on the larger Rader codelets, so the
# catalog shrinks here; large-codelet correctness is covered by the Release --verify sweep
# instead. This is not the whole memory story: even with this cap active, the planner TU
# (before the direction split) still peaked at 19.3 GB under gcc ASan+UBSan (57.8 GB at
# -march=native), because the hog is `template class plan_impl<float>` instantiating the
# route tree per SIMD width, which no catalog knob reaches. Hence the pinned ISA and
# clang++ in the asan preset — see CMakePresets.json and README.
if(NOT ADM_SANITIZER STREQUAL "none")
    set(ADM_CODELET_MAX_N 16)
    set(ADM_CODELET_EXTRA_SIZES "")
endif()

if(ADM_CODELET_MIN_N LESS 2)
    message(FATAL_ERROR "ADM_CODELET_MIN_N must be >= 2")
endif()
if(ADM_CODELET_MAX_N LESS ADM_CODELET_MIN_N)
    message(FATAL_ERROR "ADM_CODELET_MAX_N must be >= ADM_CODELET_MIN_N")
endif()

# The contiguous [MIN, MAX] range drives poet::dispatch as a dense jump table.
# Appending the sparse extras to that sequence degrades it to a compare chain,
# measured at +12-14 cycles on EVERY codelet transform, so the extras get an
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
set(ADM_CODELET_SEQUENCE "std::integer_sequence<std::size_t, ${ADM_CODELET_RANGE_CSV}>")

set(ADM_CODELET_EXTRA_DISPATCH "")
set(ADM_CODELET_EXTRA_DISPATCH_MANY "")
foreach(CODELET_N IN LISTS ADM_CODELET_EXTRA_SIZES)
    if(NOT CODELET_N IN_LIST ADM_CODELET_RANGE_SIZES)
        string(APPEND ADM_CODELET_EXTRA_DISPATCH
            "    if (N == ${CODELET_N}) {\n"
            "        codelet_apply<${CODELET_N}, T, Forward>(in, out);\n"
            "        return;\n"
            "    }\n")
        string(APPEND ADM_CODELET_EXTRA_DISPATCH_MANY
            "    if (N == ${CODELET_N}) {\n"
            "        codelet_apply_many<${CODELET_N}, T, Forward>(data, nlines, stride, fct);\n"
            "        return;\n"
            "    }\n")
    endif()
endforeach()

# codelet_max.hpp carries the same catalog to the headers, so the headers and the
# generated TUs cannot disagree about which sizes exist.
set(ADM_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
set(ADM_GENERATED_INCLUDE_DIR ${ADM_GENERATED_INCLUDE_DIR} PARENT_SCOPE)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_max.hpp.in
    ${ADM_GENERATED_INCLUDE_DIR}/admiral/detail/codelet_max.hpp
    @ONLY)

# One generated header for both flags so the library and its consumers agree.
set(ADM_MEASURE_01 0)
if(ADM_MEASURE)
    set(ADM_MEASURE_01 1)
endif()
if(ADM_ENABLE_THREADS)
    set(ADM_THREADS_01 1)
    find_package(Threads REQUIRED)
    set(ADM_THREADS_LIB Threads::Threads)
else()
    set(ADM_THREADS_01 0)
    set(ADM_THREADS_LIB "")
endif()
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
    ${ADM_GENERATED_INCLUDE_DIR}/admiral/detail/config.hpp
    @ONLY)

# One TU per catalog N, plus one dispatch TU whose extern-template block is
# generated here rather than built from preprocessor macros.
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
        "extern template void codelet_apply<${CODELET_N}, double, false>(const std::complex<double>*, std::complex<double>*);\n"
        "extern template void codelet_apply_many<${CODELET_N}, float,  true >(std::complex<float>*,  std::size_t, std::size_t, float);\n"
        "extern template void codelet_apply_many<${CODELET_N}, float,  false>(std::complex<float>*,  std::size_t, std::size_t, float);\n"
        "extern template void codelet_apply_many<${CODELET_N}, double, true >(std::complex<double>*, std::size_t, std::size_t, double);\n"
        "extern template void codelet_apply_many<${CODELET_N}, double, false>(std::complex<double>*, std::size_t, std::size_t, double);\n")
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_dispatch.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp
    @ONLY)
