# clang-tidy and cppcheck run as compiler launchers: a finding fails the build instead of
# scrolling past in a log. The launchers are the native CMake hooks: scripts/validate.sh's tidy
# and cppcheck arms configure with -DCMAKE_CXX_CLANG_TIDY / -DCMAKE_CXX_CPPCHECK, and a user can
# do the same.
#
# Write the resolved launcher command lines out, one per line, so the positive control
# (`scripts/static_analysis_control.sh`) runs the SAME flags the build ran instead of a second
# copy that can drift.
file(WRITE "${CMAKE_BINARY_DIR}/static_analysis_cmd.txt" "")
foreach(_adm_sa_var CMAKE_CXX_CLANG_TIDY CMAKE_CXX_CPPCHECK)
  if(NOT "${${_adm_sa_var}}" STREQUAL "")
    string(REPLACE ";" " " _adm_sa_cmd "${${_adm_sa_var}}")
    file(APPEND "${CMAKE_BINARY_DIR}/static_analysis_cmd.txt" "${_adm_sa_cmd}\n")
  endif()
endforeach()
