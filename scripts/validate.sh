#!/usr/bin/env bash
# One entry point for "does admiral build clean and pass everywhere".
#
# Every arm is a real configure + build + ctest, and each asserts against
# compile_commands.json that the flags it asked for arrived. That assertion is the
# point, because CMake accepts an unknown -D name in silence.
#
# usage: scripts/validate.sh [arm...]        (default: every arm)
#   isa         x86-64 / -v2 / -v3 / -v4 at Release
#   compilers   $CXX_LIST (default "g++ clang++") at x86-64-v3; the baseline is the isa arm
#   catalog     a non-default codelet catalog (extra size 66 -> Rader's codelet inner)
#   cxx17       the C++17 compatibility arm (ADM_CXX_STANDARD=17), tests on
#   sanitize    address+undefined, then thread
#   valgrind    memcheck over the test binaries, no AVX-512, no fast-math
#   tidy        clang-tidy over the two non-template source files
# env: ADM_VALIDATE_JOBS (16), ADM_VALIDATE_OUT (build/validate), CXX_LIST

set -uo pipefail

# A NINJA_STATUS from the caller's shell can carry a placeholder this ninja rejects,
# which aborts the build with a fatal error that has nothing to do with the code.
unset NINJA_STATUS

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${ADM_VALIDATE_OUT:-$src/build/validate}
jobs=${ADM_VALIDATE_JOBS:-16}
read -ra cxx_list <<<"${CXX_LIST:-g++ clang++}"
mkdir -p "$out"

# gcc writes its intermediate .s to TMPDIR, and a sanitizer TU here spills gigabytes
# apiece. Default it next to the build tree so a full /tmp cannot kill the run.
export TMPDIR=${TMPDIR:-$out/tmp}
mkdir -p "$TMPDIR"

pass=0
declare -a failed=()

# checks are "want:<pat>" / "not:<pat>" over the compile database. Both directions
# matter: the -march is in force only if nothing appended a second one after it.
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

# run_arm <name> <check-spec...> -- <cmake args...>
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
    # Arms are ~1G of build tree each; keep a failed one for the post-mortem.
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
    # Compiler per sanitizer and a pinned ISA, like CI. Peak compiler RSS tracks SIMD
    # width, so unpinning the ISA multiplies the per-TU footprint several-fold. That is
    # the reason for the -j cap below (per_tu=3 GB). This arm excludes gcc ASan on
    # purpose, the one combination that does not fit a small machine.
    #
    # `local jobs` is deliberate. Bash scopes dynamically, so it shadows the global for
    # run_arm as well.
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
        # -O1, not Debug's -O0: xsimd's immediate arguments must fold to constants; at
        # -O0 a single test target does not finish in reasonable time.

        # clang 22 exceeds its default constexpr step budget on make_rader_bhat<97, float,
        # *>; the flag is clang-only, so g++'s arm must not see it. The Catch2 __COUNTER__
        # suppression lives in CompilerWarnings.cmake, which applies after -Wpedantic.
        local cxxflags=-O1
        [[ $cxx == clang++ ]] && cxxflags="-O1 -fconstexpr-steps=100000000"
        run_arm "san-${san//+/-}" "want:-fsanitize=${san/+/,}" "want:-O1" \
            "want:-march=x86-64-v2" "not:-march=native" -- \
            -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Debug \
            -DADM_SANITIZER="$san" -DADM_TARGET_ARCH=x86-64-v2 -DCMAKE_CXX_FLAGS="$cxxflags" \
            -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
    done
}

# The C++17 compatibility arm. The flag assertion is the same tripwire as the ISA
# arms: ADM_CXX_STANDARD drives CMAKE_CXX_STANDARD, and a missed threading of it
# would compile -std=c++20 quietly otherwise. Codegen parity with the C++20 build
# is checked per release (object .text diff), not here.
arm_cxx17() {
    run_arm cxx17 "want:-std=c++17" "want:-march=x86-64-v3" "not:std=c++20" -- \
        -DCMAKE_BUILD_TYPE=Release -DADM_CXX_STANDARD=17 -DADM_TARGET_ARCH=x86-64-v3 \
        -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
}

# The codelet catalog is four cache variables (MIN_N/MAX_N/EXTRA/EXCLUDE) and nothing else
# here builds a non-default one, so a knob the docs invite users to turn had no test.
# Adding 66 makes p=67 the first Rader prime whose p-1 is a catalog member, which is the
# only way rader_inner_kind::codelet is reachable. See the catalog test in
# test_route_forced.cpp, which SKIPs without such a size.
#
# The flag check is on the generated TU rather than on a -D name: cmake accepts an unknown
# -D in silence, but src/codelet_66.cpp only exists once the catalog has grown, and its
# presence is what makes the test find its witness instead of skipping.
arm_catalog() {
    run_arm catalog "want:codelet_66" "want:-march=x86-64-v3" -- \
        -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH=x86-64-v3 \
        -DADM_CODELET_EXTRA_SIZES="120;66" \
        -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF
}

# Valgrind cannot decode AVX-512, so this arm must be x86-64-v2. A native build would
# report SIGILL, not bugs. fast-math off so the numeric assertions mean what they say.
# This arm does not use ctest -T memcheck. That needs include(CTest), and running the
# binaries directly keeps the exit status attributable to valgrind rather than to ctest.
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
        # ~[ulp]: those cases spend all their time in an O(N^2) long-double reference,
        # which valgrind turns into minutes of pure arithmetic it has nothing to say about.
        # test_ulp is [ulp] end to end, so the filter leaves it with nothing to run and
        # Catch2 exits 4, which reads as a valgrind error. Skip the binary, not the tag.
        [[ $(basename "$bin") == test_ulp ]] && continue
        echo "--- $(basename "$bin")" >>"$log"
        valgrind --error-exitcode=1 --errors-for-leak-kinds=definite \
            --leak-check=full "$bin" '~[ulp]' >>"$log" 2>&1 || { echo "  ERRORS: $(basename "$bin")"; rc=1; }
    done
    ((rc == 0)) && { echo "  OK"; ((pass++)); rm -rf "$dir"; } || failed+=("valgrind")
}

# Run clang-tidy against the non-template source files.
# Uses a fresh clang++ configure (tests off, so no ctest, just compile_commands.json).
# Assertions: compile_commands uses clang++, .clang-tidy has HeaderFilterRegex.
# The HeaderFilterRegex check is required: without it, every xsimd/poet/catch2 header
# fires hundreds of diagnostics and the arm is useless as a regression gate.
arm_tidy() {
    local dir=$out/tidy log=$out/tidy.log rc=0
    echo "=== tidy"
    if ! command -v clang-tidy >/dev/null; then
        echo "  skipped: clang-tidy not on PATH"; return
    fi
    rm -rf "$dir"; : >"$log"
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release \
        -DADM_TARGET_ARCH=x86-64-v3 -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=OFF \
        >>"$log" 2>&1 || { echo "  FAILED: configure (see $log)"; failed+=("tidy"); return; }
    check_flags "$dir" "want:clang++" "want:-march=x86-64-v3" || { failed+=("tidy"); return; }
    cmake --build "$dir" --target admiral -j "$jobs" >>"$log" 2>&1 ||
        { echo "  FAILED: build (see $log)"; failed+=("tidy"); return; }
    grep -q 'HeaderFilterRegex' "$src/.clang-tidy" ||
        { echo "  MISSING: HeaderFilterRegex in .clang-tidy"; failed+=("tidy"); return; }
    # Scope to the two non-template implementation files. The extern-template
    # instantiation TUs (inst_*.cpp) are thin wrappers; running tidy on them
    # would take O(hours) parsing the same template specialisations repeatedly.
    # HeaderFilterRegex '.*admiral.*' keeps every admiral/ header reachable from
    # these TUs under the checks.
    clang-tidy -p "$dir" --quiet \
        "$src/src/c_api.cpp" "$src/src/fftw_compat.cpp" >>"$log" 2>&1 || rc=1
    ((rc == 0)) && { echo "  OK"; ((pass++)); rm -rf "$dir"; } ||
        { echo "  FAILED: clang-tidy (see $log)"; failed+=("tidy"); }
}

# Quoted "${@:-a b c}" would expand the default as ONE word, and an unknown arm name
# would otherwise pass silently, so both cases fail below.
arms=("$@")
((${#arms[@]})) || arms=(isa compilers catalog cxx17 sanitize valgrind tidy)
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
