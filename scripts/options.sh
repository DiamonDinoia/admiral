#!/usr/bin/env bash
# Configure once for each build option and assert the effect. docs/build-options.md quotes each
# block. Run it from the source root. It builds nothing and removes ./build on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset NINJA_STATUS
export CMAKE_GENERATOR=Ninja
trap 'rm -rf build' EXIT

rm -rf build
# [doc:opt-ADM_BUILD_TESTS]
cmake -S . -B build -DADM_BUILD_TESTS=OFF
test "$(cmake --build build --target help | grep -c '^test_api_fftw:')" -eq 0
# [doc:opt-ADM_BUILD_TESTS-end]

rm -rf build
# [doc:opt-ADM_BUILD_BENCHMARKS]
cmake -S . -B build -DADM_BUILD_BENCHMARKS=OFF
test "$(cmake --build build --target help | grep -c '^admiral_benchmark:')" -eq 0
# [doc:opt-ADM_BUILD_BENCHMARKS-end]

rm -rf build
# [doc:opt-ADM_BUILD_EXAMPLES]
cmake -S . -B build -DADM_BUILD_EXAMPLES=OFF
test "$(cmake --build build --target help | grep -c '^example_quickstart:')" -eq 0
# [doc:opt-ADM_BUILD_EXAMPLES-end]

rm -rf build
# [doc:opt-ADM_CXX_STANDARD]
cmake -S . -B build -DADM_CXX_STANDARD=17
scripts/check_flags.sh build want:-std=c++17 not:-std=c++20
# [doc:opt-ADM_CXX_STANDARD-end]

rm -rf build
# [doc:opt-ADM_ENABLE_THREADS]
cmake -S . -B build -DADM_ENABLE_THREADS=OFF
grep -F '#define ADM_THREADS 0' build/src/generated/include/admiral/detail/config.hpp
# [doc:opt-ADM_ENABLE_THREADS-end]

rm -rf build
# [doc:opt-ADM_MEASURE]
cmake -S . -B build -DADM_MEASURE=OFF
grep -F 'adm_measure = 0 != 0' build/src/generated/include/admiral/detail/config.hpp
# [doc:opt-ADM_MEASURE-end]

rm -rf build
# [doc:opt-ADM_TARGET_ARCH]
cmake -S . -B build -DADM_TARGET_ARCH=x86-64-v3
scripts/check_flags.sh build want:-march=x86-64-v3 not:-march=native
# [doc:opt-ADM_TARGET_ARCH-end]

rm -rf build
# [doc:opt-ADM_ENABLE_WARNINGS_AS_ERRORS]
cmake -S . -B build -DADM_ENABLE_WARNINGS_AS_ERRORS=OFF
scripts/check_flags.sh build not:-Werror
# [doc:opt-ADM_ENABLE_WARNINGS_AS_ERRORS-end]

rm -rf build
# [doc:opt-ADM_ENABLE_LTO]
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DADM_ENABLE_LTO=ON
scripts/check_flags.sh build want:-flto
# [doc:opt-ADM_ENABLE_LTO-end]

rm -rf build
# [doc:opt-ADM_ENABLE_PCH]
cmake -S . -B build -DADM_ENABLE_PCH=ON
scripts/check_flags.sh build want:cmake_pch
# [doc:opt-ADM_ENABLE_PCH-end]

rm -rf build
# [doc:opt-ADM_SANITIZER]
cmake -S . -B build -DADM_SANITIZER=address+undefined
scripts/check_flags.sh build want:-fsanitize=address,undefined
# [doc:opt-ADM_SANITIZER-end]

rm -rf build
# [doc:opt-ADM_USE_FAST_MATH]
cmake -S . -B build -DADM_USE_FAST_MATH=OFF
scripts/check_flags.sh build not:-ffast-math
# [doc:opt-ADM_USE_FAST_MATH-end]

rm -rf build
# [doc:opt-ADM_GRANULE_ADMIT]
cmake -S . -B build -DADM_GRANULE_ADMIT=OFF
scripts/check_flags.sh build want:-DADM_GRANULE_ADMIT_OFF
# [doc:opt-ADM_GRANULE_ADMIT-end]

rm -rf build
# [doc:opt-ADM_CODELET_MIN_N]
cmake -S . -B build -DADM_CODELET_MIN_N=4
grep -F '4, 5, 6' build/src/generated/include/admiral/detail/codelet_max.hpp
# [doc:opt-ADM_CODELET_MIN_N-end]

rm -rf build
# [doc:opt-ADM_CODELET_MAX_N]
cmake -S . -B build -DADM_CODELET_MAX_N=32
grep -F '32, 65' build/src/generated/include/admiral/detail/codelet_max.hpp
# [doc:opt-ADM_CODELET_MAX_N-end]

rm -rf build
# [doc:opt-ADM_CODELET_EXCLUDE_SIZES]
cmake -S . -B build '-DADM_CODELET_EXCLUDE_SIZES=7;9'
grep -F '6, 8, 10' build/src/generated/include/admiral/detail/codelet_max.hpp
# [doc:opt-ADM_CODELET_EXCLUDE_SIZES-end]

rm -rf build
# [doc:opt-ADM_CODELET_EXTRA_SIZES]
cmake -S . -B build -DADM_CODELET_EXTRA_SIZES=128
grep -F '64, 128' build/src/generated/include/admiral/detail/codelet_max.hpp
# [doc:opt-ADM_CODELET_EXTRA_SIZES-end]

rm -rf build
# [doc:opt-ADM_EXTRA_CXX_FLAGS]
cmake -S . -B build -DADM_EXTRA_CXX_FLAGS=-DDOC_CXX
scripts/check_flags.sh build want:-DDOC_CXX
# [doc:opt-ADM_EXTRA_CXX_FLAGS-end]

rm -rf build
# [doc:opt-ADM_BENCH_FFTW]
cmake -S . -B build -DADM_BENCH_FFTW=ON | grep -F 'FFTW: enabled for benchmarks'
# [doc:opt-ADM_BENCH_FFTW-end]

rm -rf build
# [doc:opt-ADM_BENCH_THREADS]
cmake -S . -B build -DADM_BENCH_THREADS=ON | grep -F 'ducc0: threading ENABLED'
# [doc:opt-ADM_BENCH_THREADS-end]

rm -rf build
# [doc:opt-ADM_PIN_1D]
cmake -S . -B build -DADM_PIN_1D=ON
grep inst_dif_d_fwd.cpp build/compile_commands.json | grep -F -- -ffp-contract=on
# [doc:opt-ADM_PIN_1D-end]

rm -rf build
# [doc:opt-ADM_PIN_PLAN_ESTIMATE]
cmake -S . -B build -DADM_PIN_PLAN_ESTIMATE=ON
scripts/check_flags.sh build want:-DADM_PIN_PLAN_ESTIMATE
# [doc:opt-ADM_PIN_PLAN_ESTIMATE-end]

rm -rf build
# [doc:opt-ADM_ENABLE_CLANG_TIDY]
cmake -S . -B build -DADM_ENABLE_CLANG_TIDY=ON
grep -m1 -o -- --tidy= build/build.ninja
# [doc:opt-ADM_ENABLE_CLANG_TIDY-end]

rm -rf build
# [doc:opt-ADM_ENABLE_CPPCHECK]
cmake -S . -B build -DADM_ENABLE_CPPCHECK=ON
grep -m1 -o -- --cppcheck= build/build.ninja
# [doc:opt-ADM_ENABLE_CPPCHECK-end]

rm -rf build
# [doc:opt-ADM_FIT_COST_MODEL]
cmake -S . -B build -DADM_FIT_COST_MODEL=ON
cmake --build build --target help | grep '^admiral_cost_model:'
# [doc:opt-ADM_FIT_COST_MODEL-end]

rm -rf build
# [doc:opt-ADM_COST_MODEL_DATA]
cmake -S . -B build -DADM_FIT_COST_MODEL=ON -DADM_COST_MODEL_DATA="$PWD/build/receipts"
cmake -LA -N build | grep -F ADM_COST_MODEL_DATA:PATH="$PWD/build/receipts"
# [doc:opt-ADM_COST_MODEL_DATA-end]

rm -rf build
# [doc:opt-ADM_COST_MODEL_OUT]
cmake -S . -B build -DADM_FIT_COST_MODEL=ON -DADM_COST_MODEL_OUT="$PWD/build/model.hpp"
cmake -LA -N build | grep -F ADM_COST_MODEL_OUT:FILEPATH="$PWD/build/model.hpp"
# [doc:opt-ADM_COST_MODEL_OUT-end]

rm -rf build
# [doc:opt-ADM_COST_MODEL_TAG]
cmake -S . -B build -DADM_FIT_COST_MODEL=ON -DADM_COST_MODEL_TAG=mine
cmake -LA -N build | grep -F ADM_COST_MODEL_TAG:STRING=mine
# [doc:opt-ADM_COST_MODEL_TAG-end]
