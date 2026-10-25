#!/usr/bin/env bash
# Usage: check_readme.sh [README]
# Each fenced block in README follows a line "<!-- from: PATH -->". PATH is relative to the repo
# root. The block lines must appear in PATH as consecutive lines, ignoring leading whitespace.
# Any line with a fence marker opens a block, so an unmarked or unusual fence fails.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
awk '
function strip(s) { sub(/^[ \t]+/, "", s); return s }
function bad(line, why) { printf "%s:%d: %s\n", FILENAME, line, why }
function check(    i, j, n, l) {
    if (marker == "") return bad(start, "no marker")
    while ((getline l < marker) > 0) src[++n] = strip(l)
    close(marker)
    for (i = 1; blen > 0 && i + blen - 1 <= n; i++) {
        for (j = 1; j <= blen && src[i + j - 1] == body[j]; j++) ;
        if (j > blen) { checked++; return }
    }
    bad(start, "text not found in " marker)
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
