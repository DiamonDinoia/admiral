#!/usr/bin/env bash

set -uo pipefail

unset NINJA_STATUS

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${ADM_VALIDATE_OUT:-$src/build/validate}
jobs=${ADM_VALIDATE_JOBS:-16}
read -ra cxx_list <<<"${CXX_LIST:-g++ clang++}"
mkdir -p "$out"

export TMPDIR=${TMPDIR:-$out/tmp}
mkdir -p "$TMPDIR"

pass=0
declare -a failed=()

check_flags() {
    local dir=$1 spec pat n rc=0
    shift
    for spec in "$@"; do
        pat=${spec#*:}
        n=$(grep -c -F -- "$pat" "$dir/compile_commands.json" 2>/dev/null || true)
        case ${spec%%:*} in
            want) ((n > 0)) || { echo "  FLAG MISSING: $pat"; rc=1; } ;;
            not)  ((n == 0)) || { echo "  FLAG PRESENT: $pat ($n TUs)"; rc=1; } ;;
        esac
    done
    return $rc
}

run_arm() {
    local name=$1 dir log; shift
    local -a checks=()
    while [[ $1 != -- ]]; do checks+=("$1"); shift; done
    shift
    dir=$out/$name log=$out/$name.log
    echo "=== $name"
    rm -rf "$dir"
    step() { local what=$1; shift; "$@" >>"$log" 2>&1 || { echo "  FAILED: $what (see $log)"; return 1; }; }
    : >"$log"
    step configure cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "$@" &&
        check_flags "$dir" "${checks[@]}" &&
        step build cmake --build "$dir" -j "$jobs" &&
        step ctest ctest --test-dir "$dir" -j "$jobs" --output-on-failure ||
        { failed+=("$name"); return; }
    rm -rf "$dir"
    echo "  OK"; ((pass++))
}

arm_isa() {
    for level in x86-64 x86-64-v2 x86-64-v3 x86-64-v4; do
        run_arm "isa-$level" "want:-march=$level" "not:-march=native" -- \
            -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH="$level" \
            -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
    done
}

arm_compilers() {
    for cxx in "${cxx_list[@]}"; do
        if ! command -v "$cxx" >/dev/null; then
            echo "=== compiler-$cxx: skipped, not on PATH"; continue
        fi
        run_arm "compiler-${cxx//+/p}" "want:-march=x86-64-v3" "want:$cxx" -- \
            -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Release \
            -DADM_TARGET_ARCH=x86-64-v3 -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
    done
}

arm_sanitize() {
    local cap=$jobs per_tu=3 avail
    avail=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo 2>/dev/null)
    local jobs=$(( ${avail:-0} / per_tu ))
    ((jobs < 1)) && jobs=1
    ((jobs > cap)) && jobs=$cap
    echo "=== sanitize: -j $jobs (${avail:-?} GB available, ~$per_tu GB per sanitized TU)"
    local spec san cxx
    for spec in address+undefined:clang++ thread:g++; do
        san=${spec%%:*} cxx=${spec#*:}
        command -v "$cxx" >/dev/null || { echo "=== san-$san: skipped, $cxx not on PATH"; continue; }

        local cxxflags=-O1
        [[ $cxx == clang++ ]] && cxxflags="-O1 -fconstexpr-steps=100000000"
        run_arm "san-${san//+/-}" "want:-fsanitize=${san/+/,}" "want:-O1" \
            "want:-march=x86-64-v2" "not:-march=native" -- \
            -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Debug \
            -DADM_SANITIZER="$san" -DADM_TARGET_ARCH=x86-64-v2 -DCMAKE_CXX_FLAGS="$cxxflags" \
            -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
    done
}

arm_cxx17() {
    run_arm cxx17 "want:-std=c++17" "want:-march=x86-64-v3" "not:std=c++20" -- \
        -DCMAKE_BUILD_TYPE=Release -DADM_CXX_STANDARD=17 -DADM_TARGET_ARCH=x86-64-v3 \
        -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
}

arm_catalog() {
    run_arm catalog "want:codelet_66" "want:-march=x86-64-v3" -- \
        -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH=x86-64-v3 \
        -DADM_CODELET_EXTRA_SIZES="120;66" \
        -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
}

arm_valgrind() {
    local dir=$out/valgrind log=$out/valgrind.log rc=0
    echo "=== valgrind"
    command -v valgrind >/dev/null || { echo "  skipped: valgrind not on PATH"; return; }
    rm -rf "$dir"; : >"$log"
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DADM_TARGET_ARCH=x86-64-v2 \
        -DADM_USE_FAST_MATH=OFF -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF >>"$log" 2>&1 &&
        check_flags "$dir" "want:-march=x86-64-v2" "not:-march=native" "not:-ffast-math" &&
        cmake --build "$dir" -j "$jobs" >>"$log" 2>&1 || {
            echo "  FAILED: configure or build (see $log)"; failed+=("valgrind"); return; }
    for bin in "$dir"/test/test_*; do
        [[ -x $bin && ! -d $bin ]] || continue
        [[ $(basename "$bin") == test_ulp ]] && continue
        [[ $(basename "$bin") == test_long_double ]] && continue
        # test_alloc replaces global operator new/delete; valgrind redirects the same symbols
        # and the counter misreads (6f130b4). CI's valgrind job skips it the same way.
        [[ $(basename "$bin") == test_alloc ]] && continue
        echo "--- $(basename "$bin")" >>"$log"
        valgrind --error-exitcode=1 --errors-for-leak-kinds=definite \
            --leak-check=full "$bin" '~[ulp]' '~[longdouble]' >>"$log" 2>&1 || { echo "  ERRORS: $(basename "$bin")"; rc=1; }
    done
    ((rc == 0)) && { echo "  OK"; ((pass++)); rm -rf "$dir"; } || failed+=("valgrind")
}

# clang-tidy runs as a compiler launcher, so it sees every TU of the library, not a hand-picked
# pair, and a finding fails the build. The same wiring runs in CI's static-analysis job.
arm_tidy() {
    local dir=$out/tidy log=$out/tidy.log
    echo "=== tidy"
    if ! command -v clang-tidy >/dev/null; then
        echo "  skipped: clang-tidy not on PATH"; return
    fi
    grep -q 'HeaderFilterRegex' "$src/.clang-tidy" ||
        { echo "  MISSING: HeaderFilterRegex in .clang-tidy"; failed+=("tidy"); return; }
    rm -rf "$dir"; : >"$log"
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
        -DADM_TARGET_ARCH=x86-64-v3 -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF \
        -DADM_BUILD_EXAMPLES=OFF -DADM_ENABLE_CLANG_TIDY=ON \
        >>"$log" 2>&1 || { echo "  FAILED: configure (see $log)"; failed+=("tidy"); return; }
    check_flags "$dir" "want:clang++" "want:-march=x86-64-v3" || { failed+=("tidy"); return; }
    cmake --build "$dir" --target admiral -j "$jobs" >>"$log" 2>&1 ||
        { echo "  FAILED: clang-tidy (see $log)"; failed+=("tidy"); return; }
    "$src/scripts/static_analysis_control.sh" "$dir" >>"$log" 2>&1 &&
        { echo "  OK"; ((pass++)); rm -rf "$dir"; } ||
        { echo "  FAILED: positive control (see $log)"; failed+=("tidy"); }
}

arm_cppcheck() {
    local dir=$out/cppcheck log=$out/cppcheck.log
    echo "=== cppcheck"
    if ! command -v cppcheck >/dev/null; then
        echo "  skipped: cppcheck not on PATH"; return
    fi
    rm -rf "$dir"; : >"$log"
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release \
        -DADM_TARGET_ARCH=x86-64-v3 -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF \
        -DADM_BUILD_EXAMPLES=OFF -DADM_ENABLE_CPPCHECK=ON \
        >>"$log" 2>&1 || { echo "  FAILED: configure (see $log)"; failed+=("cppcheck"); return; }
    cmake --build "$dir" --target admiral -j "$jobs" >>"$log" 2>&1 ||
        { echo "  FAILED: cppcheck (see $log)"; failed+=("cppcheck"); return; }
    "$src/scripts/static_analysis_control.sh" "$dir" >>"$log" 2>&1 &&
        { echo "  OK"; ((pass++)); rm -rf "$dir"; } ||
        { echo "  FAILED: positive control (see $log)"; failed+=("cppcheck"); }
}

arms=("$@")
((${#arms[@]})) || arms=(isa compilers catalog cxx17 sanitize valgrind tidy cppcheck)
for arm in "${arms[@]}"; do
    if [[ $(type -t "arm_$arm") != function ]]; then
        echo "=== $arm: no such arm"; failed+=("$arm-unknown"); continue
    fi
    "arm_$arm"
done

echo
echo "validate: $pass passed, ${#failed[@]} failed"
((pass > 0)) || { echo "validate: no arm reported success"; exit 1; }
((${#failed[@]} == 0)) || { printf 'failed: %s\n' "${failed[@]}"; exit 1; }
