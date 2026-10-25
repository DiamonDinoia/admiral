#!/usr/bin/env bash
# Positive controls for scripts/check_readme.sh. Fixture markers resolve against this directory.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=test/check.sh
. "$here/../check.sh"
chk=$here/../../scripts/check_readme.sh
export ADM_README_ROOT=$here
# says <pass|fail> <text> <readme>: check_readme gives the status that the first argument names,
# and its output holds the text.
says() {
    local out rc=0
    out=$(bash "$chk" "$3" 2>&1) || rc=$?
    case $1:$rc in pass:0 | fail:[1-9]*) [[ $out == *"$2"* ]] && return ;; esac
    echo "status $rc: $out"
    return 1
}
# expect <fixture> <pass|fail> <text> <claim>
expect() { check "$1.md: $4" says "$2" "$3" "$here/$1.md"; }
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
check "directory README: fails with a check_readme message" \
    says fail "check_readme: cannot read" "$here"
tmp=$(mktemp -d); : > "$tmp/r.md"; chmod 000 "$tmp/r.md"
if [[ -r $tmp/r.md ]]; then
    echo "SKIP: unreadable README: file stays readable (root)"
else
    check "unreadable README: fails with a check_readme message" \
        says fail "check_readme: cannot read" "$tmp/r.md"
fi
chmod 600 "$tmp/r.md"; rm -rf "$tmp"
finish
