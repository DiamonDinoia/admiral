# CPM.cmake Setup
# Downloads and initializes CPM.cmake dependency manager
# https://github.com/cpm-cmake/CPM.cmake

# Guard against multiple inclusion
if(_FFT_CPM_INCLUDED)
    return()
endif()
set(_FFT_CPM_INCLUDED TRUE)

set(CPM_DOWNLOAD_VERSION 0.38.7)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake v${CPM_DOWNLOAD_VERSION}")
    file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()

include(${CPM_DOWNLOAD_LOCATION})

# Enable CPM source cache for faster rebuilds
# Set via: export CPM_SOURCE_CACHE=$HOME/.cache/CPM
if(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_SOURCE_CACHE $ENV{CPM_SOURCE_CACHE})
    message(STATUS "CPM source cache: ${CPM_SOURCE_CACHE}")
endif()
