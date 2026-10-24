# Windows resolves a DLL from the executable's own directory, so a consumer that links the
# shared surface needs it copied there before ctest runs.
function(adm_stage_dlls target)
    if(WIN32)
        # The list is empty for the static links, and `copy` with no source is an error, so the
        # command itself is selected by whether there is anything to copy.
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E
                    $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:${target}>>,copy,true>
                    $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endif()
endfunction()
