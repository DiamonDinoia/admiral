
function(set_project_warnings target_name)
    set(MSVC_WARNINGS
        /W4
        /w14242
        /w14254
        /w14263
        /w14265
        /w14287
        /we4289
        /w14296
        /w14311
        /w14545
        /w14546
        /w14547
        /w14549
        /w14555
        /w14619
        /w14640
        /w14826
        /w14905
        /w14906
        /w14928
        /permissive-
        # C4275: admiral::error is exported and derives from std::runtime_error, which no DLL
        # exports. Every C++ library with an exception type hits this; the base is header-only.
        /wd4275
        # C4127: a template whose parameter reaches a plain `if` condition. The kernels are written
        # against widths and radices, so the constant conditions are the point.
        /wd4127
        # C4702: MSVC issues it from the optimizer, so a generic function reads as dead code
        # wherever one caller pins a runtime parameter. Every call in `src/codelet_apply.hpp`
        # passes xstride 1, which makes `kernel_batched::apply_impl`'s scalar tail unreachable in
        # a codelet TU and live in the engine TUs. gcc and clang have no equivalent diagnostic,
        # and two of the four reported sites were inside xsimd, which SYSTEM did not cover.
        /wd4702
    )

    set(COMMON_WARNINGS
        -Wall
        -Wextra
        -Wshadow
        -Wnon-virtual-dtor
        -Wold-style-cast
        -Wcast-align
        -Wunused
        -Woverloaded-virtual
        -Wpedantic
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Wdouble-promotion
        -Wformat=2
        -Wimplicit-fallthrough
        -Wsuggest-override
        -Wcast-qual
        -Wswitch-enum
        -Wundef
        -Wredundant-decls
    )

    set(CLANG_WARNINGS
        ${COMMON_WARNINGS}
        -Wzero-as-null-pointer-constant
        -Wextra-semi
    )

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-Werror -Wno-c2y-extensions" ADM_HAVE_WNO_C2Y_EXTENSIONS)
    if(ADM_HAVE_WNO_C2Y_EXTENSIONS)
        list(APPEND CLANG_WARNINGS -Wno-c2y-extensions)
    endif()

    set(GCC_WARNINGS
        ${COMMON_WARNINGS}
        -Wmisleading-indentation
        -Wduplicated-cond
        -Wduplicated-branches
        -Wlogical-op
        -Wuseless-cast
    )

    if(ADM_ENABLE_WARNINGS_AS_ERRORS)
        list(APPEND CLANG_WARNINGS -Werror)
        list(APPEND GCC_WARNINGS -Werror)
        list(APPEND MSVC_WARNINGS /WX)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        set(PROJECT_WARNINGS ${MSVC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(PROJECT_WARNINGS ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        set(PROJECT_WARNINGS ${GCC_WARNINGS})
    endif()

    target_compile_options(${target_name} INTERFACE ${PROJECT_WARNINGS})
endfunction()

add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)
