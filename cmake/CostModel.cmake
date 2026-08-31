# Optional regeneration of `include/admiral/detail/base_cost_model.hpp`: coefficients
# only, pooled over every swept build; `effort::automatic` recovers a mispriced size
# at plan time. OFF by default; turn on to fold a new compiler, version or target
# into the pool. The fitter is C++20 stdlib-only (`tools/fit_cost_model.cpp`), no
# Python.
#   cmake -B b -DADM_FIT_COST_MODEL=ON
#   cmake --build b --target admiral_cost_sweep   # measure THIS build
#   cmake --build b --target admiral_cost_model   # refit from all receipts

option(ADM_FIT_COST_MODEL
       "Expose targets that re-measure and refit the routing cost model" OFF)

if(NOT ADM_FIT_COST_MODEL)
  return()
endif()

set(ADM_COST_MODEL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/bench-results" CACHE PATH
    "Directory of BASECOST receipts to fit the routing cost model from")
file(MAKE_DIRECTORY "${ADM_COST_MODEL_DATA}")

# The default overwrites the shipped header in-tree; point elsewhere to dry-run a refit.
set(ADM_COST_MODEL_OUT
    "${CMAKE_CURRENT_SOURCE_DIR}/include/admiral/detail/base_cost_model.hpp"
    CACHE FILEPATH "Header the admiral_cost_model target (re)generates")

# The receipt name only has to be unique; the receipt contents self-describe via
# `BASECOST-ENV` (compiler, version, width, register count). Flags hash in, and the
# uarch too. Two builds sharing flags but not silicon must not share a receipt.
get_target_property(_adm_opts admiral_optimization_flags INTERFACE_COMPILE_OPTIONS)
set(_adm_flags)
foreach(_o IN LISTS _adm_opts)
  if(NOT _o MATCHES "^\\$<")
    string(APPEND _adm_flags " ${_o}")
  endif()
endforeach()
if(NOT CMAKE_CROSSCOMPILING)
  # `try_run` takes (<runResult> <compileResult>): here run-exit = 0, compiled = TRUE.
  try_run(_adm_ua_exit _adm_ua_compiled ${CMAKE_CURRENT_BINARY_DIR}/uarch_probe
    SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/probe_uarch.cpp
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include;${ADM_GENERATED_INCLUDE_DIR};${xsimd_SOURCE_DIR}/include;${poet_SOURCE_DIR}/include"
    COMPILE_DEFINITIONS "${_adm_flags} ${ADM_EXTRA_CXX_FLAGS}"
    RUN_OUTPUT_VARIABLE _adm_uarch)
  if(_adm_ua_compiled AND _adm_ua_exit EQUAL 0)
    string(STRIP "${_adm_uarch}" _adm_uarch)
  else()
    set(_adm_uarch "unknown")
  endif()
else()
  set(_adm_uarch "generic")
endif()
string(TOLOWER "${CMAKE_CXX_COMPILER_ID}" _adm_cc)
string(REGEX MATCH "^[0-9]+" _adm_ccver "${CMAKE_CXX_COMPILER_VERSION}")
string(SHA1 _adm_flagsha "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}_${_adm_uarch}")
string(SUBSTRING "${_adm_flagsha}" 0 8 _adm_flagsha)
set(ADM_COST_MODEL_TAG "${_adm_cc}${_adm_ccver}_${CMAKE_SYSTEM_PROCESSOR}_${_adm_flagsha}"
    CACHE STRING "Name of the receipt this build writes")

if(NOT TARGET admiral_benchmark)
  message(WARNING "ADM_FIT_COST_MODEL: no admiral_benchmark target "
                  "(ADM_BUILD_BENCHMARKS=OFF); the sweep target is unavailable")
else()
  add_custom_target(
    admiral_cost_sweep
    COMMAND admiral_benchmark --base-cost=2-512 --prec=both
            --out=${ADM_COST_MODEL_DATA}/base_cost_${ADM_COST_MODEL_TAG}.txt
    COMMENT "Sweeping base kernel costs for 2..512 -> "
            "${ADM_COST_MODEL_DATA}/base_cost_${ADM_COST_MODEL_TAG}.txt"
    VERBATIM)
endif()

# `tools/fit_cost_model.cpp` is stdlib-only and reads the measured codelet-cost table
# in `math.hpp`, so the fitter cannot silently drift from what the engine runs.
add_executable(admiral_fit_cost_model tools/fit_cost_model.cpp)
target_include_directories(admiral_fit_cost_model PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${ADM_GENERATED_INCLUDE_DIR})
target_compile_features(admiral_fit_cost_model PRIVATE cxx_std_20)
set_target_properties(admiral_fit_cost_model PROPERTIES POSITION_INDEPENDENT_CODE OFF)

add_custom_target(
  admiral_cost_model
  COMMAND admiral_fit_cost_model
          --data ${ADM_COST_MODEL_DATA}
          --out ${ADM_COST_MODEL_OUT}
  COMMENT "Refitting ${ADM_COST_MODEL_OUT} from ${ADM_COST_MODEL_DATA}"
  VERBATIM)
