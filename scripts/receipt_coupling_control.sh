#!/usr/bin/env bash
# Positive control for scripts/receipt_coupling_check.sh. On a scratch clone of HEAD it
# fabricates commits and asserts the scan: FAILS a commit that edits a constants header
# without "receipt:" (naming sha and path), passes one that carries the marker in any
# case, passes one that touches only a non-constants path, and treats an empty range as
# a trivial pass. A gate that has never failed cannot be told apart from one that
# cannot fail; this script is what proves the check bites. Exit non-zero on any
# surprise. Runtime is a handful of mktemp-local git operations, well under 30 s.

set -uo pipefail

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
check=$src/scripts/receipt_coupling_check.sh
tmp=$(mktemp -d "${TMPDIR:-/tmp}/adm_receipt_control_XXXXXX")
trap 'rm -rf "$tmp"' EXIT
repo=$tmp/repo

rc=0
fail() { echo "receipt-coupling-control: $*" >&2; rc=1; }

git clone --quiet --no-checkout "$src" "$repo" || { fail "clone of $src failed"; exit 1; }
head_sha=$(git -C "$src" rev-parse HEAD)
git -C "$repo" checkout --quiet "$head_sha" || { fail "checkout of $head_sha failed"; exit 1; }
gitc=(git -C "$repo" -c user.name=receipt-control -c user.email=receipt-control@localhost)
cst=include/admiral/detail/granule_codelet.hpp

# 1: constants edit without the marker. The scan must fail and name both sha and path.
printf '\n// control scratch edit\n' >>"$repo/$cst"
"${gitc[@]}" add "$cst" >/dev/null
"${gitc[@]}" commit --quiet -m "test: control scratch edit without the marker" ||
    fail "scratch commit 1 failed"
sha1=$(git -C "$repo" rev-parse HEAD)
out=$("$check" "$repo" HEAD~1..HEAD 2>&1)
v=$?
((v != 0)) || fail "a marker-less constants edit PASSED the scan"
grep -q "${sha1:0:12}" <<<"$out" || fail "the failure did not name the sha"
grep -q "$cst" <<<"$out" || fail "the failure did not name the path"

# 2: constants edit carrying the marker in a non-canonical case. The scan must pass.
printf '\n// control scratch edit, marked\n' >>"$repo/$cst"
"${gitc[@]}" add "$cst" >/dev/null
"${gitc[@]}" commit --quiet -m "test: control scratch edit with the marker" \
    -m "Receipt: a scratch commit for the receipt-coupling positive control." ||
    fail "scratch commit 2 failed"
"$check" "$repo" HEAD~1..HEAD >/dev/null ||
    fail "a marked constants edit FAILED the scan"

# 3: non-constants path without the marker. The scan must pass.
echo x >"$repo/CONTROL_SCRATCH.md"
"${gitc[@]}" add CONTROL_SCRATCH.md >/dev/null
"${gitc[@]}" commit --quiet -m "test: control scratch edit outside the constants set" ||
    fail "scratch commit 3 failed"
"$check" "$repo" HEAD~1..HEAD >/dev/null ||
    fail "a non-constants edit FAILED the scan"

# 4: empty range passes trivially and says so.
out=$("$check" "$repo" HEAD..HEAD) || fail "an empty range FAILED"
grep -qi "empty" <<<"$out" || fail "the empty-range pass did not say so"

((rc == 0)) && echo "receipt-coupling-control: OK — the scan fails a marker-less" \
    "constants edit (naming sha+path), passes a marked one, passes a non-constants" \
    "edit, and treats an empty range as a trivial pass"
exit $rc
