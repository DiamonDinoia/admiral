#!/usr/bin/env bash
# D9 (G5), last gate: the source must not grow a foreign CPU-probe, loader or env-var route.
#
#   grep -rnE (cpuid|getauxval|__builtin_cpu_supports|target_clones|dlopen|getenv\("ADM) include src
# must not hit outside the two allowed sites, and
#   grep -rlE (xsimd::dispatch|available_architectures) include src
# must list src/api.cpp only. grep, not rg: the dispatch-x86 runner has no ripgrep.
#
# Allowed sites: src/scratch_alloc.cpp reads LD_PRELOAD (valgrind detection, stays);
# src/fftw_compat.cpp's dlopen(RTLD_LAZY|RTLD_NOLOAD) route came in with port/integrate and its
# review.
#
# The first grep's allowlist is PINS: RELPATH:TEXT pairs. A hit is allowed only when its file
# AND its text match a pin, never by line number (review M1: a line-number filter hides a new
# token on that line of any file). Empty today: every first-grep token is a failure.
#
# The controls mode plants both probe strings at the end of src/scratch_alloc.cpp and a cpuid
# comment on line 80 of src/api.cpp. Both copies must fail.
set -u
mode=${1:-check}
SRC="$(cd "$(dirname "$0")/.." && pwd)"
PINS=""

run_checks() { # where
    local W=$1 bad=0 n
    local hits kept="" hit pin drop
    hits=$({ grep -rnE -e cpuid -e getauxval -e __builtin_cpu_supports -e target_clones -e 'getenv\("ADM' \
        "$W/include" "$W/src" 2>/dev/null || true; } | grep -v -e src/test || true)
    while IFS= read -r hit; do
        [ -n "$hit" ] || continue
        drop=0
        for pin in $PINS; do
            case $hit in "$W/${pin%%:*}:"*"${pin#*:}"*) drop=1 ;; esac
        done
        [ "$drop" = 0 ] && kept="$kept$hit
"
    done <<< "$hits"
    n=$(printf '%s' "$kept" | grep -c . || true)
    local fdl
    fdl=$(grep -rn -e dlopen "$W/src" "$W/include" 2>/dev/null | grep -v src/fftw_compat.cpp | wc -l || true)
    if [ "$n" = 0 ] && [ "$fdl" = 0 ]; then :; else
        bad=1
        [ "$n" = 0 ] || printf '%s\n' "$kept"
        [ "$fdl" = 0 ] || grep -rn -e dlopen "$W/src" "$W/include" 2>/dev/null | grep -v src/fftw_compat.cpp
    fi
    local xs
    xs=$(grep -rlE -e 'xsimd::dispatch' -e available_architectures "$W/include" "$W/src" 2>/dev/null || true)
    if [ "$xs" != "$W/src/api.cpp" ]; then bad=1; echo "  xsimd route in: $xs"; fi
    # fftw_compat dlopen is allowed only at the reviewed site (line may drift: pin the call shape).
    if grep -qF 'dlopen(lib.c_str(), RTLD_LAZY | RTLD_NOLOAD)' "$W/src/fftw_compat.cpp"; then :; else
        bad=1; echo "  fftw_compat.cpp dlopen route changed shape"
    fi
    return $bad
}

case $mode in
    check)
        echo "== D9/G5: no foreign probe, loader or getenv routes"
        if run_checks "$SRC"; then echo "PASS"; exit 0; else echo "FAIL"; exit 1; fi
        ;;
    controls)
        echo "== controls D9/G5: planted routes must fail, a line-80 token must fail"
        T=$(mktemp -d); T2=$(mktemp -d); trap 'rm -rf "$T" "$T2"' EXIT
        cp -r "$SRC/include" "$T/include"
        cp -r "$SRC/src" "$T/src"
        echo 'getauxval(0); xsimd::dispatch<void>();' >> "$T/src/scratch_alloc.cpp"
        # Break the file listing claim and the clean-hit claim at once.
        cp -r "$SRC/include" "$T2/include"
        cp -r "$SRC/src" "$T2/src"
        sed -i '80i // cpuid' "$T2/src/api.cpp"   # lands ON line 80 (M1 mutation)
        rc=0
        if run_checks "$T"; then echo "control: appended routes did not fire: FAIL"; rc=1
        else echo "control: appended routes fired"; fi
        if run_checks "$T2"; then echo "control: cpuid at src/api.cpp:80 did not fire: FAIL"; rc=1
        else echo "control: cpuid at src/api.cpp:80 fired"; fi
        [ "$rc" = 0 ] && { echo "controls fired: PASS"; exit 0; } || { echo "controls FAIL"; exit 1; }
        ;;
    *) echo "usage: check_g5.sh [check|controls]"; exit 2 ;;
esac
