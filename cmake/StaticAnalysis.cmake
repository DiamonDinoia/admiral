# clang-tidy and cppcheck as compiler launchers: a finding fails the build instead of scrolling
# past in a log. Include AFTER Dependencies so third-party targets stay unanalysed.

option(ADM_ENABLE_CLANG_TIDY "Run clang-tidy on admiral's own sources during the build" OFF)
option(ADM_ENABLE_CPPCHECK "Run cppcheck on admiral's own sources during the build" OFF)

if(ADM_ENABLE_CLANG_TIDY)
  find_program(ADM_CLANG_TIDY NAMES clang-tidy REQUIRED)
  # Checks and WarningsAsErrors come from .clang-tidy; HeaderFilterRegex there keeps the
  # diagnostics on admiral headers and off xsimd/poet/catch2.
  # `--config-file` is not optional: the generated codelet sources live in the BUILD tree, and
  # clang-tidy's upward search from there can miss the repo's .clang-tidy entirely, which silently
  # analyses them with the default (near-empty) check set.
  set(CMAKE_CXX_CLANG_TIDY "${ADM_CLANG_TIDY}" "--quiet"
      "--config-file=${PROJECT_SOURCE_DIR}/.clang-tidy")
  message(STATUS "clang-tidy: ${ADM_CLANG_TIDY}")
endif()

if(ADM_ENABLE_CPPCHECK)
  find_program(ADM_CPPCHECK NAMES cppcheck REQUIRED)
  # cppcheck cannot follow every system header, and it reports the template-heavy SIMD headers as
  # unreachable configurations; those two classes are suppressed and nothing else is.
  set(CMAKE_CXX_CPPCHECK
      "${ADM_CPPCHECK}"
      "--enable=warning,performance,portability"
      "--inline-suppr"
      "--suppress=missingInclude"
      "--suppress=missingIncludeSystem"
      "--suppress=unmatchedSuppression"
      "--suppress=checkersReport"
      # cppcheck 2.21 cannot parse a variable template initialised by an immediately-invoked
      # lambda (butterfly.hpp `crt_index`). Scoped to that file so a parse failure anywhere else
      # still fails the build.
      "--suppress=syntaxError:*/butterfly.hpp"
      "--error-exitcode=2"
      "--std=c++${ADM_CXX_STANDARD}")
  message(STATUS "cppcheck: ${ADM_CPPCHECK}")
endif()

# Write the resolved launcher command lines out, one per line, so the positive control
# (`scripts/static_analysis_control.sh`) runs the SAME flags the build ran instead of a second
# copy that can drift.
file(WRITE "${CMAKE_BINARY_DIR}/static_analysis_cmd.txt" "")
if(ADM_ENABLE_CLANG_TIDY)
  string(REPLACE ";" " " _adm_tidy_cmd "${CMAKE_CXX_CLANG_TIDY}")
  file(APPEND "${CMAKE_BINARY_DIR}/static_analysis_cmd.txt" "${_adm_tidy_cmd}\n")
endif()
if(ADM_ENABLE_CPPCHECK)
  string(REPLACE ";" " " _adm_cppcheck_cmd "${CMAKE_CXX_CPPCHECK}")
  file(APPEND "${CMAKE_BINARY_DIR}/static_analysis_cmd.txt" "${_adm_cppcheck_cmd}\n")
endif()
