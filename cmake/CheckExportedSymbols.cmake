
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

string(REPLACE "|" ";" required "${REQUIRED}")
foreach(want IN LISTS required)
    if(NOT symbols MATCHES "${want}")
        message(FATAL_ERROR "${LIB} exports nothing matching ${want}")
    endif()
endforeach()
