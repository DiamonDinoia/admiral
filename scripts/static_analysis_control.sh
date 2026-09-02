#!/usr/bin/env bash
# Positive control for the static-analysis gates. Every analyser the build configured must stay
# SILENT on a clean file and REPORT on test/static_analysis/control.cpp. A clean run over the
# library says nothing on its own: a check that has never failed cannot be told apart from one
# that cannot fail, and the clean half attributes the failure to the defect rather than to a
# parse error.
#
# Usage: scripts/static_analysis_control.sh <build-dir>
# The build dir must be configured with -DADM_ENABLE_CLANG_TIDY=ON and/or -DADM_ENABLE_CPPCHECK=ON.
# The flags come from that build's static_analysis_cmd.txt, so they cannot drift from the build's.

set -uo pipefail

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dir=${1:?usage: static_analysis_control.sh <build-dir>}
cmds=$dir/static_analysis_cmd.txt
bad=$src/test/static_analysis/control.cpp
std=$(grep -o -- '-std=c++[0-9]*' "$dir/compile_commands.json" 2>/dev/null | head -1)
std=${std:--std=c++20}

[[ -s $cmds ]] || { echo "control: $cmds is empty; the build enabled no analyser"; exit 1; }

good=$(mktemp "${TMPDIR:-/tmp}/adm_sa_clean_XXXXXX.cpp")
trap 'rm -f "$good"' EXIT
printf '#include <vector>\nint adm_clean(const std::vector<int>& v) { return static_cast<int>(v.size()); }\n' >"$good"

run() {  # run <tool-kind> <cmd> <file>
    case $1 in
    clang-tidy) (cd "$src" && eval "$2 \"$3\" -- $std") ;;
    cppcheck)   eval "$2 \"$3\"" ;;
    *)          return 125 ;;
    esac
} >/dev/null 2>&1 </dev/null

# A check listed in .clang-tidy but absent from `--list-checks` is a check that is not running.
# That is not hypothetical: `Checks: >` folds the whole list into ONE line, `#` then runs to the
# end of that line, and the entry after each comment block gets glued to the comment text into a
# junk glob. Nine checks were silently off that way. The YAML-list form is what makes the comments
# real comments, and this loop is what proves it stayed that way.
check_roster() {  # check_roster <clang-tidy-binary>
    local tidy=$1 miss=0 want have
    have=$("$tidy" --config-file="$src/.clang-tidy" --list-checks 2>/dev/null | tail -n +2 | tr -d ' ')
    while read -r want; do
        grep -qxF "$want" <<<"$have" || { echo "control: .clang-tidy lists $want but clang-tidy does not run it"; miss=1; }
    done < <(sed -n "s/^  - '\([a-z][a-z0-9.*-]*\)'.*/\1/p" "$src/.clang-tidy")
    return $miss
}

rc=0
while read -r cmd; do
    [[ -n $cmd ]] || continue
    read -r bin _ <<<"$cmd"
    case $(basename "$bin") in
    *clang-tidy*) kind=clang-tidy ;;
    *cppcheck*)   kind=cppcheck ;;
    *)            echo "control: unknown analyser $bin"; rc=1; continue ;;
    esac
    if [[ $kind == clang-tidy ]]; then
        check_roster "$bin" || rc=1
    fi
    run "$kind" "$cmd" "$good"
    if (($? != 0)); then
        echo "control: $kind reports on a CLEAN file; its verdict on the library is meaningless"
        rc=1
        continue
    fi
    run "$kind" "$cmd" "$bad"
    if (($? == 0)); then
        echo "control: $kind is SILENT on the known-bad file; the gate is dead"
        rc=1
    else
        echo "control: $kind silent on clean, reports on known-bad"
    fi
done <"$cmds"
exit $rc
