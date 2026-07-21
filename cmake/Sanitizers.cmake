# Sanitizers.cmake
# Provides AddressSanitizer (ASAN) support

function(enable_sanitizers target_name)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target_name}
            INTERFACE
                -fsanitize=address
                -fsanitize=undefined
                -fno-omit-frame-pointer
                -g
        )
        target_link_options(${target_name}
            INTERFACE
                -fsanitize=address
                -fsanitize=undefined
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_compile_options(${target_name}
            INTERFACE
                /fsanitize=address
                /Zi
        )
    endif()
endfunction()

if(ADM_ENABLE_ASAN)
    add_library(project_sanitizers INTERFACE)
    enable_sanitizers(project_sanitizers)
endif()
