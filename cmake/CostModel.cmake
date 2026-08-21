# Optional regeneration of include/admiral/detail/base_cost_model.hpp.
#
# OFF by default, and no build needs it. The checked-in header is coefficients
# only, pooled over every swept build. effort::automatic recovers a size those
# coefficients misprice at plan time.
#
# Turn it on to fold a new compiler, version or target into the pool. No Python.
# The fitter is C++ (tools/fit_cost_model.cpp).
#   cmake -B b -DADM_FIT_COST_MODEL=ON
#   cmake --build b --target admiral_cost_sweep   # measure THIS build
#   cmake --build b --target admiral_cost_model   # refit the header from all
#                                                 # receipts in the data dir

option(ADM_FIT_COST_MODEL
       "Expose targets that re-measure and refit the routing cost model" OFF)

if(NOT ADM_FIT_COST_MODEL)
  return()
endif()

set(ADM_COST_MODEL_DATA "${CMAKE_CURRENT_SOURCE_DIR}/bench-results" CACHE PATH
    "Directory of BASECOST receipts to fit the routing cost model from")
file(MAKE_DIRECTORY "${ADM_COST_MODEL_DATA}")

# Defaults write the shipped header in-tree, like any code generator; point this
# elsewhere to dry-run a refit without dirtying the checkout.
set(ADM_COST_MODEL_OUT
    "${CMAKE_CURRENT_SOURCE_DIR}/include/admiral/detail/base_cost_model.hpp"
    CACHE FILEPATH "Header the admiral_cost_model target (re)generates")

# The receipt file name only has to be unique; its contents are self-describing
# (BASECOST-ENV records compiler, version, width and register count). Compile
# flags hash in as well. Two builds of one compiler at different -march are different
# codegen and must not share a receipt. The uarch mixes in too, because identical
# flags on different silicon share the flags-hash, and one machine must not clobber
# another's.
get_target_property(_adm_opts admiral_optimization_flags INTERFACE_COMPILE_OPTIONS)
set(_adm_flags)
foreach(_o IN LISTS _adm_opts)
  if(NOT _o MATCHES "^\\$<")
    string(APPEND _adm_flags " ${_o}")
  endif()
endforeach()
if(NOT CMAKE_CROSSCOMPILING)
  # try_run is (<runResult> <compileResult>): here, run-exit = 0, compiled = TRUE.
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

# tools/fit_cost_model.cpp is stdlib-only and deterministic, so the header is
# reproducible from the receipts. It reads math.hpp for the measured codelet cost
# table that the fitted features use. A second copy in the fitter could
# silently drift from what the engine runs.
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
