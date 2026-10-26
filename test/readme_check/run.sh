#!/usr/bin/env bash
# Positive controls for scripts/check_readme.sh. Fixture markers name the tag "sample" in test/readme_check/source.txt.
set -u
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=test/check.sh
. "$here/../check.sh"
# says <pass|fail> <text> <fixture>: check_readme gives the status that the first argument names,
# and its output holds the text.
says() {
    local out rc=0
    out=$(bash "$here/../../scripts/check_readme.sh" "$here/$3.md" 2>&1) || rc=$?
    case $1:$rc in pass:0 | fail:[1-9]*) [[ $out == *"$2"* ]] && return ;; esac
    echo "status $rc: $out"
    return 1
}
check "good.md: a block with another indentation matches" says pass "1 of 1 blocks checked" good
check "drifted.md: one changed character is caught" says fail "text differs" drifted
check "unmarked.md: a block without a marker is caught" says fail "no marker" unmarked
check "dangling.md: a marker before a blank line is caught" says fail "dangling.md:1: marker" dangling
check "dangling.md: a marker at the end of the file is caught" says fail "dangling.md:6: marker" dangling
check "missing.md: a tag that the source lacks is caught" says fail "no tagged lines nope" missing
check "empty.md: a README without blocks fails" says fail "no fenced blocks" empty
finish
