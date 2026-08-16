# CompilerWarnings.cmake
# Sets up compiler warnings for the project

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

    # Warnings shared by both Clang and GCC. Verified zero-warning-clean on clang
    # and gcc.
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
        # Additional flags — zero-warning-clean on both compilers:
        -Wcast-qual          # drop-const casts
        # -Wfloat-equal is OFF: butterfly.hpp uses if constexpr(w.c == 0.0) for
        # compile-time special-case detection; those are exact constant comparisons,
        # not runtime imprecision. The check is a false positive throughout.
        -Wswitch-enum        # switch missing an enum case
        -Wundef              # macro used in #if before definition
        -Wredundant-decls    # duplicate declarations
    )

    set(CLANG_WARNINGS
        ${COMMON_WARNINGS}
        # Clang-only (not supported by GCC as standalone flags):
        -Wzero-as-null-pointer-constant  # literal 0 used as null pointer
        -Wextra-semi                     # redundant semicolons after definitions
        # -Wpedantic above makes clang 22 reject Catch2's __COUNTER__ as a C2y extension,
        # reported against the test files where its macros expand. This must follow
        # -Wpedantic, or -Wpedantic re-enables the diagnostic.
        -Wno-c2y-extensions
    )

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

# Create an interface library for project warnings
add_library(project_warnings INTERFACE)
set_project_warnings(project_warnings)
