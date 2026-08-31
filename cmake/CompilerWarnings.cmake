# Warning flags for the project, exported as the `project_warnings` INTERFACE target.

function(set_project_warnings target_name)
    set(MSVC_WARNINGS
        /W4     # Baseline reasonable warnings
        /w14242 # 'identifier': conversion from 'type1' to 'type2', possible loss of data
        /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
        /w14263 # 'function': member function does not override any base class virtual member function
        /w14265 # 'classname': class has virtual functions, but destructor is not virtual
        /w14287 # 'operator': unsigned/negative constant mismatch
        /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside the for-loop scope
        /w14296 # 'operator': expression is always 'boolean_value'
        /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545 # expression before comma evaluates to a function which is missing an argument list
        /w14546 # function call before comma missing argument list
        /w14547 # 'operator': operator before comma has no effect; expected operator with side-effect
        /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
        /w14555 # expression has no effect; expected expression with side-effect
        /w14619 # pragma warning: there is no warning number 'number'
        /w14640 # Enable warning on thread un-safe static member initialization
        /w14826 # Conversion from 'type1' to 'type2' is sign-extended
        /w14905 # wide string literal cast to 'LPSTR'
        /w14906 # string literal cast to 'LPWSTR'
        /w14928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        /permissive- # standards conformance mode
    )

    # Shared by Clang and GCC; verified zero-warning-clean on both.
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
        -Wcast-qual          # drop-const casts
        # `-Wfloat-equal` is OFF: `butterfly.hpp` compares exact compile-time
        # constants via `if constexpr(w.c == 0.0)`; the check false-positives
        # throughout.
        -Wswitch-enum        # switch missing an enum case
        -Wundef              # macro used in #if before definition
        -Wredundant-decls    # duplicate declarations
    )

    set(CLANG_WARNINGS
        ${COMMON_WARNINGS}
        # Clang-only (GCC does not accept these as standalone flags):
        -Wzero-as-null-pointer-constant  # literal 0 used as null pointer
        -Wextra-semi                     # redundant semicolons after definitions
    )

    # The `-Wpedantic` of clang 22 rejects Catch2's `__COUNTER__` as a C2y extension,
    # so this suppression must follow `-Wpedantic`. Older clang errors on the unknown
    # option under `-Werror`, so probe. The probe needs `-Werror`: without the flag,
    # clang reports unknown `-Wno-` options only when another diagnostic fires, and a
    # bare check false-positives.
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
