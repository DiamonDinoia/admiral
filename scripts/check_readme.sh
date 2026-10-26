#!/usr/bin/env bash
# Usage: check_readme.sh [README]
# Each fenced block in README follows a line "<!-- from: PATH#NAME -->". PATH is relative to the
# repo root. The block must equal the lines of PATH between "[doc:NAME]" and "[doc:NAME-end]",
# ignoring leading whitespace. The docs pull the same lines in with literalinclude.
# Any line with a fence marker opens a block, so an unmarked or unusual fence fails.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
awk '
function strip(s) { sub(/^[ \t]+/, "", s); return s }
function bad(line, why) { printf "%s:%d: %s\n", FILENAME, line, why }
function check(    i, n, l, on, m) {
    if (marker == "") return bad(start, "no marker")
    split(marker, m, "#")
    while ((getline l < m[1]) > 0) {
        if (index(l, "[doc:" m[2] "-end]")) on = 0
        if (on) src[++n] = strip(l)
        if (index(l, "[doc:" m[2] "]")) on = 1
    }
    close(m[1])
    if (n == 0) return bad(start, "no tagged lines " m[2] " in " m[1])
    for (i = 1; i <= n && i <= blen && src[i] == body[i]; i++) ;
    if (i > n && n == blen) { checked++; return }
    bad(start, "text differs from " marker)
}
inblock {
    if ($0 ~ /^[ \t>]*(```|~~~)+[ \t]*$/) { inblock = 0; check() } else body[++blen] = strip($0)
    next
}
/```|~~~/ { inblock = 1; blen = 0; start = FNR; marker = pending; pending = ""; blocks++; next }
{
    if (pending != "") bad(markline, "marker without a fence")
    pending = ""
    if ($0 ~ /^<!-- from: .+ -->$/) {
        pending = $0; markline = FNR
        sub(/^<!-- from: /, "", pending); sub(/ -->$/, "", pending)
    }
}
END {
    if (pending != "") bad(markline, "marker without a fence")
    if (blocks == 0) { print FILENAME ": no fenced blocks"; exit 1 }
    printf "check_readme: %d of %d blocks checked\n", checked, blocks
    exit checked != blocks
}' "${1:-README.md}"
