#!/usr/bin/env bash
# Configure once for each build option and assert the effect. docs/build-options.md quotes each
# block. Run it from the source root. It builds nothing. It needs ./build free, creates it and
# removes it on exit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset NINJA_STATUS
export CMAKE_GENERATOR=Ninja
[[ ! -e build ]] || { echo "FAIL: ./build exists, this script would delete it. Move it first."; exit 1; }
trap 'rm -rf build' EXIT

skipped=()
skip() { echo "SKIP: $2"; skipped+=("$1"); }

# Control for the three BUILD_* blocks: they expect a target to be absent, so a wrong name would
# pass. At the default configure each of these targets must exist.
cmake -S . -B build >/dev/null
for t in test_api_fftw admiral_benchmark example_quickstart; do
    test "$(cmake --build build --target help | grep -c "^$t:")" -eq 1 ||
        { echo "FAIL: the default configure has no target $t"; exit 1; }
done

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

if [[ $(uname -m) = x86_64 ]]; then
rm -rf build
# [doc:opt-ADM_TARGET_ARCH]
cmake -S . -B build -DADM_TARGET_ARCH=x86-64-v3
scripts/check_flags.sh build want:-march=x86-64-v3 not:-march=native
# [doc:opt-ADM_TARGET_ARCH-end]
else
    skip ADM_TARGET_ARCH "ADM_TARGET_ARCH example uses an x86-64 name"
fi

rm -rf build
# [doc:opt-ADM_USE_FAST_MATH]
cmake -S . -B build -DADM_USE_FAST_MATH=OFF
scripts/check_flags.sh build not:-ffast-math
# [doc:opt-ADM_USE_FAST_MATH-end]

rm -rf build
# [doc:opt-ADM_FIT_COST_MODEL]
cmake -S . -B build -DADM_FIT_COST_MODEL=ON
cmake --build build --target help | grep '^admiral_cost_model:'
# [doc:opt-ADM_FIT_COST_MODEL-end]
# The documented defaults of the three cost-model options.
cmake -LA -N build | grep -E 'ADM_COST_MODEL_DATA:PATH=.*/bench-results$'
cmake -LA -N build | grep -E 'ADM_COST_MODEL_OUT:FILEPATH=.*/include/admiral/detail/base_cost_model\.hpp$'
cmake -LA -N build | grep -E 'ADM_COST_MODEL_TAG:STRING=[A-Za-z]+[0-9]+_[A-Za-z0-9_]+_[0-9a-f]{8,}$'

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

total=$(grep -c '^# \[doc:opt-[A-Za-z0-9_]*\]$' "${BASH_SOURCE[0]}")
((total > 0))
echo "PASS: $((total - ${#skipped[@]})) options checked, ${#skipped[@]} skipped (${skipped[*]:-none})"
