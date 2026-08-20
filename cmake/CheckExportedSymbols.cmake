# ctest gate: the shipped shared library exports its public API and nothing else.
# Run: cmake -DNM=<nm> -DLIB=<path> -DPATTERN=<regex> -DREQUIRED=<regex;...>
#      -P CheckExportedSymbols.cmake
# PATTERN catches a leak, REQUIRED a silently missing export. Hidden visibility does
# not reach compiler-emitted RTTI, so the version script in src/*.map is what closes
# the ABI. A linker that ignores it must fail this test.


execute_process(
    COMMAND ${NM} -D --defined-only --format=posix ${LIB}
    OUTPUT_VARIABLE symbols
    RESULT_VARIABLE rc
    ERROR_VARIABLE err
)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "nm failed on ${LIB}: ${err}")
endif()

string(REPLACE "\n" ";" lines "${symbols}")
set(leaked "")
foreach(line IN LISTS lines)
    # --format=posix is "<name> <type> <value> <size>". Bind the name before the
    # next if(): a MATCHES test overwrites CMAKE_MATCH_<n>.
    if(line MATCHES "^([^ ]+) ")
        set(symbol ${CMAKE_MATCH_1})
        if(NOT symbol MATCHES "${PATTERN}")
            list(APPEND leaked ${symbol})
        endif()
    endif()
endforeach()

if(leaked)
    string(REPLACE ";" "\n    " leaked "${leaked}")
    message(FATAL_ERROR
        "${LIB} exports symbols outside ${PATTERN}:\n    ${leaked}")
endif()

# Pipe-separated, because a ';' would have been split by add_test.
string(REPLACE "|" ";" required "${REQUIRED}")
foreach(want IN LISTS required)
    if(NOT symbols MATCHES "${want}")
        message(FATAL_ERROR "${LIB} exports nothing matching ${want}")
    endif()
endforeach()
