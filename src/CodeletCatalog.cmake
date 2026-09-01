
set(ADM_CODELET_MIN_N 2 CACHE STRING
    "Smallest N considered for compiled straight-line codelets")
set(ADM_CODELET_MAX_N 64 CACHE STRING
    "Largest N considered for compiled straight-line codelets")
set(ADM_CODELET_EXCLUDE_SIZES "" CACHE STRING
    "Semicolon-separated sizes excluded from the generated codelet catalog")
set(ADM_CODELET_EXTRA_SIZES "120;65;85;143;100;360" CACHE STRING
    "Semicolon-separated sizes added to the catalog beyond [MIN_N, MAX_N]")

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
set(ADM_CODELET_EXTRA_DISPATCH_MANY_OOP "")
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
        string(APPEND ADM_CODELET_EXTRA_DISPATCH_MANY_OOP
            "    if (N == ${CODELET_N}) {\n"
            "        codelet_apply_many_oop<${CODELET_N}, T, Forward>(in, out, nlines, in_stride, out_stride, fct);\n"
            "        return;\n"
            "    }\n")
    endif()
endforeach()

set(ADM_GENERATED_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated/include)
set(ADM_GENERATED_INCLUDE_DIR ${ADM_GENERATED_INCLUDE_DIR} PARENT_SCOPE)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_max.hpp.in
    ${ADM_GENERATED_INCLUDE_DIR}/admiral/detail/codelet_max.hpp
    @ONLY)

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

set(ADM_CODELET_GENERATED "")
set(ADM_CODELET_EXTERN "")
foreach(CODELET_N IN LISTS ADM_CODELET_SIZES)
    set(CODELET_COL_INST "")
    if(NOT CODELET_N GREATER 64)
        # One string, not four: `set()` with four arguments builds a LIST, and substituting a
        # list into the template joins it with `;`, which clang rejects under -Wextra-semi.
        string(CONCAT CODELET_COL_INST
            "template void col_codelet_apply<${CODELET_N}, float,  true >(const std::complex<float>*, std::size_t, std::complex<float>*, std::size_t, std::size_t, float);\n"
            "template void col_codelet_apply<${CODELET_N}, float,  false>(const std::complex<float>*, std::size_t, std::complex<float>*, std::size_t, std::size_t, float);\n"
            "template void col_codelet_apply<${CODELET_N}, double, true >(const std::complex<double>*, std::size_t, std::complex<double>*, std::size_t, std::size_t, double);\n"
            "template void col_codelet_apply<${CODELET_N}, double, false>(const std::complex<double>*, std::size_t, std::complex<double>*, std::size_t, std::size_t, double);\n")
    endif()
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
            "extern template void codelet_apply_many<${CODELET_N}, double, false>(std::complex<double>*, std::size_t, std::size_t, double);\n"
            "extern template void codelet_apply_many_oop<${CODELET_N}, float,  true >(const std::complex<float>*,  std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);\n"
            "extern template void codelet_apply_many_oop<${CODELET_N}, float,  false>(const std::complex<float>*,  std::complex<float>*,  std::size_t, std::size_t, std::size_t, float);\n"
            "extern template void codelet_apply_many_oop<${CODELET_N}, double, true >(const std::complex<double>*, std::complex<double>*, std::size_t, std::size_t, std::size_t, double);\n"
            "extern template void codelet_apply_many_oop<${CODELET_N}, double, false>(const std::complex<double>*, std::complex<double>*, std::size_t, std::size_t, std::size_t, double);\n")
    if(NOT CODELET_N GREATER 64)
        string(APPEND ADM_CODELET_EXTERN
            "extern template void col_codelet_apply<${CODELET_N}, float,  true >(const std::complex<float>*, std::size_t, std::complex<float>*, std::size_t, std::size_t, float);\n"
            "extern template void col_codelet_apply<${CODELET_N}, float,  false>(const std::complex<float>*, std::size_t, std::complex<float>*, std::size_t, std::size_t, float);\n"
            "extern template void col_codelet_apply<${CODELET_N}, double, true >(const std::complex<double>*, std::size_t, std::complex<double>*, std::size_t, std::size_t, double);\n"
            "extern template void col_codelet_apply<${CODELET_N}, double, false>(const std::complex<double>*, std::size_t, std::complex<double>*, std::size_t, std::size_t, double);\n")
    endif()
endforeach()

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/codelet_dispatch.cpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/codelet_dispatch.cpp
    @ONLY)
