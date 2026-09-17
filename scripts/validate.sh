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
declare -a skipped=()

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
        step ctest ctest --test-dir "$dir" -j "$jobs" --output-on-failure --timeout 900 ||
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
            echo "=== compiler-$cxx: skipped, not on PATH"; skipped+=("compiler-$cxx"); continue
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
        command -v "$cxx" >/dev/null || { echo "=== san-$san: skipped, $cxx not on PATH"; skipped+=("san-$san"); continue; }

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

# Bit-identity digest: builds the tree twice at x86-64-v3 Release -- shipped and strict
# (ADM_USE_FAST_MATH=OFF) -- runs test/digest's positional case sweep on both, and diffs
# each against the committed goldens in test/digest/golden with scripts/check_digest.sh.
# Two invariants on top: a strict-arm mover must also be a shipped-arm mover (the strict
# build never moves a case alone; cases a strict build skips do not participate), and
# the --ulp positive control must move exactly the one case test/digest/case_list.hpp
# names (kUlpControlCase, 700). A landing that legitimately changes bits regoldens in
# its own commit (test/digest/README.md); a machinery-only landing predicts zero movers.
arm_digest() {
    local gen=$out/digest log=$out/digest.log rc=0 sub
    echo "=== digest"
    rm -rf "$gen"; mkdir -p "$gen"; : >"$log"
    local ulp_case
    ulp_case=$(sed -n 's/.*kUlpControlCase = \([0-9][0-9]*\).*/\1/p' \
        "$src/test/digest/case_list.hpp")
    [[ $ulp_case =~ ^[0-9]+$ ]] || { echo \
        "  FAILED: kUlpControlCase unreadable in test/digest/case_list.hpp"; failed+=("digest"); return; }
    for sub in shipped strict; do
        local dir=$gen/build-$sub
        local -a cfg=(-DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH=x86-64-v3
                      -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF)
        local -a checks=("want:-march=x86-64-v3" "not:-march=native")
        if [[ $sub == strict ]]; then
            cfg+=(-DADM_USE_FAST_MATH=OFF)
            checks+=("not:-ffast-math")
        else
            checks+=("want:-ffast-math")
        fi
        echo "--- $sub: configure/build" >>"$log"
        cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            "${cfg[@]}" >>"$log" 2>&1 &&
            check_flags "$dir" "${checks[@]}" &&
            { [[ $sub == shipped ]] ||
              grep -q '^ADM_USE_FAST_MATH:BOOL=OFF' "$dir/CMakeCache.txt"; } &&
            cmake --build "$dir" --target admiral_digest -j "$jobs" >>"$log" 2>&1 ||
            { echo "  FAILED: $sub configure/build (see $log)"; failed+=("digest"); return; }
        # cat puts stdout in the 4096-blksize class: a direct 1 MiB-blksize NFS file
        # sink scales the stdio first heap alloc, which shifts the Bluestein/Rader
        # buffers' alignment class and flips prime>11 1-D cases (9 spurious movers,
        # 2026-09-15). pipefail (line 3) keeps the binary's rc through the pipe.
        if "$dir/test/admiral_digest" 2>>"$log" | cat >"$gen/$sub.txt"; then
            echo "--- $sub: $(wc -l <"$gen/$sub.txt") cases" >>"$log"
        else
            echo "  FAILED: $sub run or self-check (see $log)"; failed+=("digest"); return
        fi
        : >"$gen/pred-$sub.txt"   # this landing's reachability prediction: no movers
        "$src/scripts/check_digest.sh" "$src/test/digest/golden/$sub.txt" "$gen/$sub.txt" \
            --expect-movers "$gen/pred-$sub.txt" --movers-out "$gen/$sub.movers" 2>&1 |
            tee -a "$log" || rc=1
    done
    echo "--- invariant: strict movers subset of shipped movers" >>"$log"
    local only_strict
    only_strict=$(comm -23 <(LC_ALL=C sort "$gen/strict.movers") \
                           <(LC_ALL=C sort "$gen/shipped.movers"))
    if [[ -n $only_strict ]]; then
        echo "check_digest: FAIL: strict-only movers $(echo "$only_strict" | tr '\n' ' ')" |
            tee -a "$log"; rc=1
    fi
    echo "--- control: --ulp $ulp_case moves exactly one case" >>"$log"
    # Same sink rule as the sweep runs above: stdout through a pipe, never an NFS file.
    "$gen/build-shipped/test/admiral_digest" --ulp "$ulp_case" 2>>"$log" |
        cat >"$gen/shipped-ulp.txt" &&
        printf '%s\n' "$ulp_case" >"$gen/pred-ulp.txt" &&
        "$src/scripts/check_digest.sh" "$gen/shipped.txt" "$gen/shipped-ulp.txt" \
            --expect-movers "$gen/pred-ulp.txt" 2>&1 | tee -a "$log" || rc=1
    if ((rc == 0)); then
        { echo "  digest: zero movers vs golden in both arms; strict movers subset of shipped"; \
          echo "  digest: --ulp $ulp_case control moved exactly case $ulp_case"; } >>"$log"
        echo "  OK"; ((pass++))
        rm -rf "$gen"/build-shipped "$gen"/build-strict
    else
        echo "  FAILED: digest mismatch (see $log)"; failed+=("digest")
    fi
}

arm_valgrind() {
    local dir=$out/valgrind log=$out/valgrind.log rc=0 vrc=0
    echo "=== valgrind"
    command -v valgrind >/dev/null || { echo "  skipped: valgrind not on PATH"; skipped+=(valgrind); return; }
    rm -rf "$dir"; : >"$log"
    cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo -DADM_TARGET_ARCH=x86-64-v2 \
        -DADM_USE_FAST_MATH=OFF -DADM_BUILD_TESTS=ON -DADM_BUILD_BENCHMARKS=OFF >>"$log" 2>&1 &&
        check_flags "$dir" "want:-march=x86-64-v2" "not:-march=native" "not:-ffast-math" &&
        cmake --build "$dir" -j "$jobs" >>"$log" 2>&1 || {
            echo "  FAILED: configure or build (see $log)"; failed+=("valgrind"); return; }
    # WILL_FAIL binaries (the poison twins in test/CMakeLists.txt) exit nonzero by design,
    # and this loop runs the binaries directly, where ctest's WILL_FAIL inversion never
    # applies. The exclusion set must come from ctest's own test metadata.
    local wf_json wf_names='' ok=1
    wf_json=$(ctest --test-dir "$dir" --show-only=json-v1 2>>"$log") || ok=0
    if ((ok == 1)); then
        local wf_expr='.tests[] | select(any(.properties[]?; .name == "WILL_FAIL" and'
        wf_expr+=' .value == true)) | .name'
        if command -v jq >/dev/null; then
            wf_names=$(jq -r "$wf_expr" <<<"$wf_json") || ok=0
        elif command -v python3 >/dev/null; then
            wf_names=$(python3 -c '
import json, sys
for t in json.load(sys.stdin)["tests"]:
    if any(p.get("name") == "WILL_FAIL" and p.get("value") is True
           for p in t.get("properties", [])):
        print(t["name"])' <<<"$wf_json") || ok=0
        else
            echo "  no jq and no python3 on PATH for ctest test metadata" >>"$log"
            ok=0
        fi
    fi
    ((ok == 1)) || { echo "  FAILED: ctest WILL_FAIL metadata (see $log)"; failed+=("valgrind"); return; }
    # A WILL_FAIL test's name is its binary's basename (add_test(NAME test_x COMMAND test_x)).
    local -A wffail=()
    local t
    for t in $wf_names; do wffail[$t]=1; done
    if ((${#wffail[@]} > 0)); then
        local -a wf_sorted
        mapfile -t wf_sorted < <(printf '%s\n' "${!wffail[@]}" | LC_ALL=C sort)
        echo "  WILL_FAIL excluded (ctest metadata): ${wf_sorted[*]}" | tee -a "$log"
    fi
    for bin in "$dir"/test/test_*; do
        [[ -x $bin && ! -d $bin ]] || continue
        [[ $(basename "$bin") == test_ulp ]] && continue
        [[ $(basename "$bin") == test_long_double ]] && continue
        # test_alloc replaces global operator new/delete; valgrind redirects the same symbols
        # and the counter misreads (6f130b4). CI's valgrind job skips it the same way.
        [[ $(basename "$bin") == test_alloc ]] && continue
        [[ -n ${wffail[$(basename "$bin")]:-} ]] &&
            { echo "  skip (WILL_FAIL): $(basename "$bin")" | tee -a "$log"; continue; }
        echo "--- $(basename "$bin")" >>"$log"
        # This loop walks the binaries directly, so ctest's --timeout never reaches them.
        # Valgrind serialises threads and costs 50-100x, so a deadlocked binary never returns.
        # Slowest measured binary is test_threads at 642 s (ccmlin075, 22 binaries, 2026-09-12).
        timeout 1800 valgrind --error-exitcode=1 --errors-for-leak-kinds=definite \
            --leak-check=full "$bin" '~[ulp]' '~[longdouble]' >>"$log" 2>&1
        vrc=$?
        ((vrc == 124)) && echo "  TIMEOUT after 1800s: $(basename "$bin")" >>"$log"
        ((vrc == 0)) || { echo "  ERRORS: $(basename "$bin")$( ((vrc == 124)) && printf ' (timeout)')"; rc=1; }
    done
    ((rc == 0)) && { echo "  OK"; ((pass++)); rm -rf "$dir"; } || failed+=("valgrind")
}

# clang-tidy runs as a compiler launcher, so it sees every TU of the library, not a hand-picked
# pair, and a finding fails the build. The same wiring runs in CI's static-analysis job.
arm_tidy() {
    local dir=$out/tidy log=$out/tidy.log
    echo "=== tidy"
    if ! command -v clang-tidy >/dev/null; then
        echo "  skipped: clang-tidy not on PATH"; skipped+=(tidy); return
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
        echo "  skipped: cppcheck not on PATH"; skipped+=(cppcheck); return
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

# Constants-with-receipt coupling (WI-0b): a commit touching a hand-fit- or probe-fit
# constants header must name its derivation receipt. The scan is
# scripts/receipt_coupling_check.sh over ${ADM_VALIDATE_BASE:-HEAD~1}..HEAD (an empty
# range passes trivially and says so); the control runs first, so a scan that cannot
# fail cannot report green.
arm_receipt-coupling() {
    local range=${ADM_VALIDATE_BASE:-HEAD~1}..HEAD
    local log=$out/receipt-coupling.log v
    echo "=== receipt-coupling ($range)"
    command -v git >/dev/null || { echo "  skipped: git not on PATH"; skipped+=(receipt-coupling); return; }
    : >"$log"
    "$src/scripts/receipt_coupling_control.sh" >>"$log" 2>&1 ||
        { echo "  FAILED: positive control (see $log)"; failed+=(receipt-coupling); return; }
    "$src/scripts/receipt_coupling_check.sh" "$src" "$range" >>"$log" 2>&1
    v=$?
    case $v in
        0) tail -1 "$log"; echo "  OK"; ((pass++)) ;;
        2) if [[ -z ${ADM_VALIDATE_BASE:-} ]]; then
               echo "  skipped: range base HEAD~1 is not a commit here (shallow checkout?);"
               echo "           set ADM_VALIDATE_BASE to a resolvable base to run the scan"
               skipped+=(receipt-coupling)
           else
               echo "  FAILED: ADM_VALIDATE_BASE=$ADM_VALIDATE_BASE does not resolve (see $log)"
               failed+=(receipt-coupling)
           fi ;;
        *) echo "  FAILED: marker-less constants commit in $range (see $log)"
           failed+=(receipt-coupling) ;;
    esac
}

# WI-0c-ii granule admission A/B. Builds the tree four times (x86-64-v4/v3 crossed with
# ADM_GRANULE_ADMIT ON/OFF; Release, tests off, benchmarks on, the bench_granule_ab target
# only), asserts the knob arrive per build via check_flags plus the CMake cache and a
# symbol-level control (ON libadmiral_internal instantiates the granule drivers, OFF none).
# Timing is interleaved: six rounds of on/off/ctl invocations per ISA with rotated start
# order (ctl is the ON binary again -- the self-pair identity control), one pinned cpu on
# a core whose sibling stays idle, min-of-R on ~2 ms bursts inside each invocation, and a
# per-cell best-case reduction (min across rounds; powersave pstate wander moves whole
# invocations, so the resolution floor is the on/ctl best-case spread, with the max
# round-pair spread printed alongside as round-max). Per (isa, cell) the arm prints
# WIN/LOSS/FLAT with the ON/OFF ratio against 2x that floor and FAILS iff an admitted
# cell LOSES to OFF beyond it; wins inside the floor are reported FLAT, never failed.
# Quiet-box contract: verdicts are acceptance-grade only on a quiet box; under load the
# identity floor spreads and every verdict degrades to informational -- the arm PRINTS a
# line naming that condition whenever a cell's floor spread exceeds 10%, never silently
# passing on it.
arm_granule-admission() {
    local root=$out/granule-admission log=$out/granule-admission.log rc=0
    echo "=== granule-admission"
    command -v taskset >/dev/null || { echo "  skipped: taskset not on PATH"; skipped+=(granule-admission); return; }
    command -v nm >/dev/null || { echo "  skipped: nm not on PATH"; skipped+=(granule-admission); return; }
    rm -rf "$root"; mkdir -p "$root"; : >"$log"

    # Pin one logical cpu on the highest-numbered core that has a second context; the
    # sibling stays idle because exactly one timed invocation runs at a time.
    local pin
    pin=$(lscpu -e=CPU,CORE 2>/dev/null |
        awk 'NR>1 {seen[$2]=seen[$2]" "$1; if ($2+0 > max) max=$2+0}
             END {n=split(seen[max], a, " "); if (n >= 2) print a[1]+0}')
    [[ -n ${pin:-} ]] || { echo "  skipped: no two-context core in lscpu output"; skipped+=(granule-admission); return; }
    echo "--- pin: cpu $pin (sibling idle)" >>"$log"

    local isa st dir ons offs
    for isa in x86-64-v4 x86-64-v3; do
        for st in on off; do
            dir=$root/build-$isa-$st
            local -a checks=("want:-march=$isa" "not:-march=native")
            if [[ $st == on ]]; then checks+=("not:-DADM_GRANULE_ADMIT_OFF"); else checks+=("want:-DADM_GRANULE_ADMIT_OFF"); fi
            echo "--- build $isa-$st" >>"$log"
            cmake -S "$src" -B "$dir" -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                -DCMAKE_BUILD_TYPE=Release -DADM_TARGET_ARCH="$isa" \
                -DADM_BUILD_TESTS=OFF -DADM_BUILD_BENCHMARKS=ON \
                -DADM_GRANULE_ADMIT="${st^^}" >>"$log" 2>&1 &&
                check_flags "$dir" "${checks[@]}" &&
                grep -q "^ADM_GRANULE_ADMIT:BOOL=${st^^}$" "$dir/CMakeCache.txt" &&
                cmake --build "$dir" --target bench_granule_ab -j "$jobs" >>"$log" 2>&1 &&
                [[ -x $dir/benchmark/bench_granule_ab ]] ||
                { echo "  FAILED: build $isa-$st (see $log)"; failed+=(granule-admission); return; }
        done
        # Knob mechanization control: ON instantiates the granule drivers, OFF drops them.
        ons=$(nm --defined-only "$root/build-$isa-on/src/libadmiral_internal.so" 2>/dev/null | grep -c "granule_apply_rows_oopILj\|granule_col_applyILj" || true)
        offs=$(nm --defined-only "$root/build-$isa-off/src/libadmiral_internal.so" 2>/dev/null | grep -c "granule_apply_rows_oopILj\|granule_col_applyILj" || true)
        echo "--- $isa: granule driver symbols on=$ons off=$offs" >>"$log"
        (( ons > 0 && offs == 0 )) ||
            { echo "  FAILED: $isa knob symbol control (on=$ons off=$offs)"; failed+=(granule-admission); return; }
    done

    # Timing: six rounds of on/off/ctl per ISA, start order rotated across rounds.
    # ctl re-runs the ON binary: the per-round on/ctl spread is the identity floor.
    # The 4x4x4/8x8x8 cells are the fast3d cube payloads (WI-1a); they are judged by the
    # same WIN/LOSS/FLAT rule as the rows/cols squares.
    local cells="12x12:f32 12x12:f64 16x16:f32 24x24:f32 24x24:f64 4x4x4:f32 4x4x4:f64 8x8x8:f32 8x8x8:f64 9x9:f64 10x10:f32 10x10:f64 11x11:f64 13x13:f32 13x13:f64 14x14:f32 14x14:f64 15x15:f32 15x15:f64"
    local r order arm bin tsv
    for isa in x86-64-v4 x86-64-v3; do
        tsv=$root/runs-$isa.tsv; : >"$tsv"
        for r in 1 2 3 4 5 6; do
            case $((r % 3)) in
                1) order="on off ctl" ;;
                2) order="off ctl on" ;;
                0) order="ctl on off" ;;
            esac
            echo "# round $r" >>"$tsv"
            for arm in $order; do
                bin=$root/build-$isa-on/benchmark/bench_granule_ab
                [[ $arm == off ]] && bin=$root/build-$isa-off/benchmark/bench_granule_ab
                # --ramp-ms 200: intel_pstate powersave idles the pinned core near base
                # clock; the spin puts it at boost before the first measured burst. With
                # no ramp the on/ctl identity floor reads +-50% on this box.
                taskset -c "$pin" "$bin" --tag "$arm" --reps 9 --ramp-ms 200 >>"$tsv" 2>>"$log" ||
                    { echo "  FAILED: run $isa round $r arm $arm (see $log)"; failed+=(granule-admission); return; }
            done
        done
        # Candidate cells: leaf-level granule-vs-SoA inside the ON binary, informational
        # transfer evidence for the built-but-unadmitted N. No verdict rides on these.
        taskset -c "$pin" "$root/build-$isa-on/benchmark/bench_granule_ab" \
            --candidate --tag on --reps 9 --ramp-ms 200 >"$root/candidates-$isa.tsv" 2>>"$log" ||
            { echo "  FAILED: candidate run $isa (see $log)"; failed+=(granule-admission); return; }
    done

    local v
    for isa in x86-64-v4 x86-64-v3; do
        v=${isa#x86-64-}
        awk -F'\t' -v isa="$v" -v cells="$cells" '
            /^# round/ { sub(/^# round /, ""); r = $0 + 0; next }
            $1 != "ab" { next }
            {
                cell = prec = tag = mode = ""; ns = ""
                for (i = 2; i <= NF; ++i) {
                    split($i, kv, "=")
                    if (kv[1] == "cell") cell = kv[2]
                    else if (kv[1] == "prec") prec = kv[2]
                    else if (kv[1] == "mode") mode = kv[2]
                    else if (kv[1] == "tag") tag = kv[2]
                    else if (kv[1] == "ns_min") ns = kv[2] + 0.0
                }
                if (mode != "plan" || cell == "" || ns == "") next
                key = cell ":" prec
                hit[tag SUBSEP key SUBSEP r] = ns
                if (mins[tag SUBSEP key] == "" || ns < mins[tag SUBSEP key]) mins[tag SUBSEP key] = ns
                seen[key] = 1
            }
            END {
                nc = split(cells, cl, " ")
                losses = 0; missing = 0
                for (j = 1; j <= nc; ++j) {
                    key = cl[j]
                    if (!(key in seen)) { printf "MISSING\t%s\tcell=%s\n", isa, key; missing++; continue }
                    on = mins["on" SUBSEP key]; off = mins["off" SUBSEP key]; ctl = mins["ctl" SUBSEP key]
                    # The control is distance of the on/ctl BEST cases from 1.0: one bad
                    # round inflates a max-spread floor, it never moves the minima.
                    spread = (on > ctl ? on - ctl : ctl - on) / (on < ctl ? on : ctl)
                    floor = 2 * spread
                    rmax = 0
                    for (rr = 1; rr <= 6; ++rr) {
                        a = hit["on" SUBSEP key SUBSEP rr]; c = hit["ctl" SUBSEP key SUBSEP rr]
                        if (a == "" || c == "") { missing++; continue }
                        s = (a > c ? a - c : c - a) / (a < c ? a : c)
                        if (s > rmax) rmax = s
                    }
                    lo = (on < off) ? on : off
                    d = (off - on) / lo
                    verdict = (d > floor) ? "WIN" : (d < -floor) ? "LOSS" : "FLAT"
                    if (verdict == "LOSS") losses++
                    printf "%s\t%s\tcell=%s\ton=%.2f\toff=%.2f\tctl=%.2f\tctl-floor=%.2f%%\t2x-floor=%.2f%%\tround-max=%.2f%%\tratio_off/on=%.4f\n",
                           isa, verdict, key, on, off, ctl, spread * 100, floor * 100, rmax * 100, off / on
                    if (spread > 0.10)
                        printf "# %s cell=%s: control spread %.2f%% exceeds 10%% -- verdicts are acceptance-grade only on a quiet box; under load the identity floor spreads and every verdict degrades to informational\n",
                               isa, key, spread * 100
                }
                exit(losses > 0 || missing > 0)
            }' "$root/runs-$isa.tsv" | tee "$root/verdicts-$isa.txt"
        ((PIPESTATUS[0] == 0)) || rc=1
        # Candidate transfer rows: granule vs SoA leaf, per axis, informational.
        awk -F'\t' -v isa="$v" '
            $1 != "ab" { next }
            {
                cell = prec = mode = arm = ""; ns = ""
                for (i = 2; i <= NF; ++i) {
                    split($i, kv, "=")
                    if (kv[1] == "cell") cell = kv[2]
                    else if (kv[1] == "prec") prec = kv[2]
                    else if (kv[1] == "mode") mode = kv[2]
                    else if (kv[1] == "arm") arm = kv[2]
                    else if (kv[1] == "ns_min") ns = kv[2] + 0.0
                }
                if (mode ~ /^leaf/ && arm != "" && ns != "") {
                    key = cell ":" prec ":" mode
                    if (best[key SUBSEP arm] == "" || ns < best[key SUBSEP arm]) best[key SUBSEP arm] = ns
                    keys[key] = 1
                }
            }
            END {
                n = 0
                for (key in keys) {
                    g = best[key SUBSEP "granule"]; s = best[key SUBSEP "soa"]
                    if (g == "" || s == "") continue
                    printf "%s\tCAND\tcell=%s\tgranule=%.2f\tsoa=%.2f\tratio_soa/granule=%.4f\n",
                           isa, key, g, s, s / g
                    n++
                }
                exit(n == 0)
            }' "$root/candidates-$isa.tsv" | sort -t$'\t' -k3 | tee "$root/transfer-$isa.txt" >/dev/null
        ((${PIPESTATUS[0]} == 0 && ${PIPESTATUS[1]} == 0)) || rc=1
    done
    if ((rc == 0)); then
        echo "  OK"; ((pass++))
        rm -rf "$root"/build-*
    else
        echo "  FAILED: admitted cell lost to ADM_GRANULE_ADMIT=OFF beyond the floor (see $log)"
        failed+=(granule-admission)
    fi
}

arms=("$@")
((${#arms[@]})) || arms=(isa compilers catalog digest cxx17 sanitize valgrind tidy cppcheck receipt-coupling granule-admission)
for arm in "${arms[@]}"; do
    if [[ $(type -t "arm_$arm") != function ]]; then
        echo "=== $arm: no such arm"; failed+=("$arm-unknown"); continue
    fi
    "arm_$arm"
done

echo
echo "validate: $pass passed, ${#failed[@]} failed, ${#skipped[@]} skipped"
((${#skipped[@]} == 0)) || printf 'skipped: %s\n' "${skipped[*]}"
((pass > 0)) || { echo "validate: no arm reported success"; exit 1; }
((${#failed[@]} == 0)) || { printf 'failed: %s\n' "${failed[@]}"; exit 1; }
