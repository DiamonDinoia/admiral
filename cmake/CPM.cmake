# CPM.cmake bootstrap, from https://github.com/cpm-cmake/CPM.cmake
# Upstream's `cmake/get_cpm.cmake` with the release pinned and hashed. The wiki
# snippet hardcodes `CMAKE_BINARY_DIR` and cannot configure a fresh build dir offline;
# honour `CPM_SOURCE_CACHE` instead. `file(DOWNLOAD)` skips the transfer when
# `EXPECTED_HASH` matches. CPM records `CPM_DIRECTORY` (`CACHE INTERNAL`) and hits a
# silent `return()` when a later configure includes CPM from elsewhere. A pre-existing
# build dir must be wiped once.
include_guard(GLOBAL)

set(CPM_DOWNLOAD_VERSION 0.38.7)
set(CPM_HASH_SUM "83e5eb71b2bbb8b1f2ad38f1950287a057624e385c238f6087f94cdfc44af9c5")
if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()
# Absolute, because the cache path is allowed to contain a tilde.
get_filename_component(CPM_DOWNLOAD_LOCATION "${CPM_DOWNLOAD_LOCATION}" ABSOLUTE)

# A runner's one-off `SSL connect error` is not a configure failure, so retry. With
# EXPECTED_HASH the transfer is skipped once the file is present and matches, and a hash
# mismatch reports a nonzero status like any other failure.
foreach(attempt RANGE 1 3)
    file(
        DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
        EXPECTED_HASH SHA256=${CPM_HASH_SUM}
        STATUS cpm_status
    )
    list(GET cpm_status 0 cpm_code)
    if(cpm_code EQUAL 0)
        break()
    endif()
endforeach()
if(NOT cpm_code EQUAL 0)
    list(GET cpm_status 1 cpm_message)
    message(FATAL_ERROR "CPM.cmake download failed after 3 attempts: ${cpm_message}")
endif()

include(${CPM_DOWNLOAD_LOCATION})
