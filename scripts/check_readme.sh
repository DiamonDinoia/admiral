#!/usr/bin/env bash
# Usage: check_readme.sh [README]
# Each fenced block in README must follow a line "<!-- from: PATH -->".
# The block body must appear in PATH as consecutive lines, ignoring leading whitespace.
# PATH resolves against ADM_README_ROOT (default: the repo root).
# The checker owns the README format, not CommonMark. A fence at column 0 to 3 is a block.
# Any other fence-like line (blockquote, list item, deep indent) fails.
# The marker rule rejects a leading / and a .. component as typo checks. It does not resolve symlinks.

set -uo pipefail

src=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
root=${ADM_README_ROOT:-$src}
readme=${1:-$src/README.md}

[[ -f $readme && -r $readme ]] || { echo "check_readme: cannot read README: $readme" >&2; exit 2; }

awk -v root="$root" '
function strip(s) { sub(/\r$/, "", s); sub(/^[ \t]+/, "", s); return s }
function bad(line, why) { printf "%s:%d: %s\n", FILENAME, line, why; fails++ }
function load(path,    n, l, r) {
    if (path in nlines) return 1
    if ((r = getline l < path) < 0) return 0
    n = 0
    while (r > 0) { src[path, ++n] = strip(l); r = getline l < path }
    close(path)
    nlines[path] = n
    return 1
}
function check(    path, i, j, n, ok, text) {
    if (marker == "") return bad(start, "no marker")
    if (marker ~ /^\// || marker ~ /(^|\/)\.\.(\/|$)/) return bad(start, "marker path leaves the root: " marker)
    path = root "/" marker
    if (!load(path)) return bad(start, "file missing: " marker)
    n = nlines[path]
    if (n == 0) return bad(start, "empty source: " marker)
    for (i = 1; i <= blen; i++) text = text body[i]
    if (text == "") return bad(start, "empty block")
    for (i = 1; i + blen - 1 <= n; i++) {
        ok = 1
        for (j = 1; j <= blen && ok; j++) if (src[path, i + j - 1] != body[j]) ok = 0
        if (ok) { used[marker] = 1; checked++; return }
    }
    bad(start, "text not found in " marker)
}
{ line = $0; sub(/\r$/, "", line) }
inblock {
    if (match(line, /^( |  |   )?(```+|~~~+)[ \t]*$/)) {
        run = line; gsub(/[ \t]/, "", run)
        if (substr(run, 1, 1) == fch && length(run) >= flen) { inblock = 0; check(); next }
    }
    body[++blen] = strip(line); next
}
match(line, /^( |  |   )?(```+|~~~+)/) {
    run = substr(line, RSTART, RLENGTH); sub(/^ +/, "", run)
    if (!(substr(run, 1, 1) == "`" && substr(line, RSTART + RLENGTH) ~ /`/)) {
        inblock = 1; fch = substr(run, 1, 1); flen = length(run)
        start = FNR; blen = 0; blocks++
        marker = pending; pending = ""
        next
    }
}
{
    if (pending != "") { bad(markline, "marker without a fence"); pending = "" }
    if (line ~ /^<!-- from: .+ -->$/) {
        pending = line; markline = FNR
        sub(/^<!-- from: /, "", pending); sub(/ -->$/, "", pending)
    } else if (line ~ /^[ >]*([-*+]|[0-9]+[.)])?[ >]*(```|~~~)/) bad(FNR, "fence the checker cannot parse")
}
END {
    if (inblock) bad(start, "unterminated block")
    if (pending != "") bad(markline, "marker without a fence")
    if (blocks == 0) { print FILENAME ": no fenced blocks"; exit 1 }
    if (fails) exit 1
    for (f in used) m++
    printf "check_readme: %d blocks checked against %d files\n", checked, m
}' "$readme"
