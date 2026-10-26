# shellcheck shell=bash
# Check helpers for the test drivers. A driver sources this file, runs checks, then calls finish.
# A positive control is a check through not. The command must exit non-zero on a known bad input.
fail=0

# check <claim> <command...>: prints PASS when the command exits 0, else FAIL. The command output
# follows the claim.
check() {
    local claim=$1 out rc=0
    shift
    out=$("$@" 2>&1) || rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "PASS: $claim${out:+: $out}"
    else
        echo "FAIL: $claim (status $rc)${out:+: $out}"
        fail=1
    fi
}

# not <command...>: exits 0 when the command exits non-zero.
not() { ! "$@"; }

# bound <file> <program> <symbol> <library regex>: <file> holds LD_DEBUG=bindings output. The first
# line of <program> for <symbol> names a library path that matches the regex.
bound() {
    awk -v f="binding file $2 " -v s="\`$3'" 'index($0, f) && index($0, s) { print; exit }' "$1" |
        grep -Eq " to [^ ]*$4 \["
}

finish() {
    if [ "$fail" -ne 0 ]; then
        echo "RESULT: FAIL"
        exit 1
    fi
    echo "RESULT: PASS"
}
