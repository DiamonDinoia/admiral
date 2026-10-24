#!/usr/bin/env bash
# Positive controls for scripts/check_readme.sh. Fixture markers resolve against this directory.
set -uo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
chk=$here/../../scripts/check_readme.sh
export ADM_README_ROOT=$here
fail=0
# expect NAME OUTCOME REASON-PATTERN CLAIM
expect() {
    local out rc
    out=$(bash "$chk" "$here/$1.md" 2>&1); rc=$?
    if [[ $2 == pass && $rc -eq 0 && $out == *"$3"* ]] || [[ $2 == fail && $rc -ne 0 && $out == *"$3"* ]]; then
        echo "ok   $1.md: $4"
    else
        echo "FAIL $1.md: expected $2 ($3), got exit=$rc: $out"; fail=1
    fi
}
expect good pass "1 blocks checked against 1 files" "verbatim block with indentation change passes"
expect crlf pass "1 blocks checked against 1 files" "CRLF README against LF source passes"
expect nested pass "1 blocks checked against 1 files" "four-backtick block holding a three-backtick pair passes"
expect tilde_good pass "1 blocks checked against 1 files" "marked tilde block is checked and passes"
expect nonl_readme pass "1 blocks checked against 1 files" "README without trailing newline passes"
expect nonl_src pass "1 blocks checked against 1 files" "source without trailing newline passes"
expect spaces pass "1 blocks checked against 1 files" "marker path with spaces passes"
expect drifted fail "text not found" "one changed character is caught"
expect laterdiff fail "text not found" "a differing later line is caught"
expect good_drifted fail "text not found" "a drifted block next to a good one is caught"
expect unmarked fail "no marker" "block without marker is caught"
expect tilde_unmarked fail "no marker" "unmarked tilde block is caught"
expect blankgap fail "marker without a fence" "blank line between marker and fence is caught"
expect twomarkers fail "marker without a fence" "two markers in a row are caught"
expect marker_text fail "marker without a fence" "marker before text is caught"
expect marker_eof fail "marker without a fence" "marker at end of file is caught"
expect missing fail "file missing" "marker naming an absent file is caught"
expect dotdot fail "leaves the root" "marker with .. component is rejected"
expect absolute fail "leaves the root" "marker with a leading slash is rejected"
expect empty_src fail "empty source" "empty source file is not matched by a blank block"
expect blankonly fail "empty block" "block of blank lines is rejected"
expect unterminated fail "unterminated" "fence without a closer is caught"
expect closer_info fail "unterminated" "closer with info string does not close the block"
expect inline_not_fence fail "cannot parse" "fence-like line with backticks in the info string fails"
expect indented4 fail "cannot parse" "fence indented 4 spaces fails"
expect bq fail "cannot parse" "blockquote fence fails"
expect bqtight fail "cannot parse" "blockquote fence without space fails"
expect bqtilde fail "cannot parse" "blockquote tilde fence fails"
expect list fail "cannot parse" "list item fence fails"
expect empty fail "no fenced blocks" "zero blocks is a failure"
# README that cannot be read: a directory, and a file with mode 000 (skipped as root).
out=$(bash "$chk" "$here" 2>&1); rc=$?
if [[ $rc -ne 0 && $out == check_readme:*"cannot read"* ]]; then echo "ok   directory README: fails with a check_readme message"; else echo "FAIL directory README: exit=$rc: $out"; fail=1; fi
tmp=$(mktemp -d); : > "$tmp/r.md"; chmod 000 "$tmp/r.md"
if [[ -r $tmp/r.md ]]; then echo "skip unreadable README: file stays readable (root)"
else
    out=$(bash "$chk" "$tmp/r.md" 2>&1); rc=$?
    if [[ $rc -ne 0 && $out == check_readme:*"cannot read"* ]]; then echo "ok   unreadable README: fails with a check_readme message"; else echo "FAIL unreadable README: exit=$rc: $out"; fail=1; fi
fi
chmod 600 "$tmp/r.md"; rm -rf "$tmp"
exit $fail
