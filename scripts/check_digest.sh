#!/usr/bin/env bash
# check_digest.sh [--expect-movers FILE] [--movers-out FILE] GOLDEN ACTUAL
#
# Text-compares two digest files (one "id tag name hash" line per case; see
# test/digest/digest.cpp). coreutils + sed only: comm, sort, join, cut, wc.
#
#   * ids in ACTUAL that GOLDEN does not know are a hard error (a case was added without
#     regoldening);
#   * ids only in GOLDEN are skipped, not compared -- a build that leaves a case
#     undefined does not participate in that case (strict builds may skip);
#   * any id whose tag or name drifted is a structural error: the list renumbered, which
#     silently re-keys the whole golden (test/digest/case_list.hpp states the rule);
#   * the remaining ids compare by hash; the differing ids are the "movers".
#
# Exit 0 iff the mover set equals --expect-movers (default: empty). The mover ids, one
# per line, go to --movers-out if given.
set -euo pipefail
export LC_ALL=C

expect=/dev/null
movers_out=
declare -a positional=()
while (($#)); do
    case $1 in
        --expect-movers) expect=$2; shift 2 ;;
        --movers-out) movers_out=$2; shift 2 ;;
        --*) echo "check_digest: unknown option $1" >&2; exit 2 ;;
        *) positional+=("$1"); shift ;;
    esac
done
((${#positional[@]} == 2)) || { echo \
    "usage: check_digest.sh [--expect-movers FILE] [--movers-out FILE] GOLDEN ACTUAL" >&2; exit 2; }
golden=${positional[0]} actual=${positional[1]}

tmp=$(mktemp -d "${TMPDIR:-/tmp}/check_digest.XXXXXX")
trap 'rm -rf "$tmp"' EXIT

# Canonical shape: "<id> <tag> <name> <hash>", one space per gap, sorted lexically on id
# (consistent collation is what join/comm demand; ids are never numeric-sorted inside).
norm() { sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]\+/ /g' "$1" | sort -t' ' -k1,1; }
norm "$golden" >"$tmp/golden"; norm "$actual" >"$tmp/actual"

cut -d' ' -f1 "$tmp/golden" >"$tmp/golden.ids"; cut -d' ' -f1 "$tmp/actual" >"$tmp/actual.ids"

unknown=$(comm -13 "$tmp/golden.ids" "$tmp/actual.ids")
if [[ -n $unknown ]]; then
    echo "check_digest: FAIL: ids in the actual digest that the golden does not know" \
        "(a case landed without regoldening): $(echo "$unknown" | sort -n | tr '\n' ' ')" >&2
    exit 1
fi
nskipped=$(comm -23 "$tmp/golden.ids" "$tmp/actual.ids" | wc -l)

# Structural check: same id, different tag or name => renumbering, not a bit change.
join -t' ' -j1 -o '0 1.2 1.3 2.2 2.3' "$tmp/golden" "$tmp/actual" >"$tmp/names"
drift=
while read -r id gtag gname atag aname; do
    [[ $gtag == "$atag" && $gname == "$aname" ]] || drift+="$id($gtag $gname->$atag $aname) "
done <"$tmp/names"
if [[ -n $drift ]]; then
    echo "check_digest: FAIL: case ids renumbered ($drift): new cases append after the" \
        "f64 block only (test/digest/case_list.hpp)" >&2
    exit 1
fi

join -t' ' -j1 -o '0 1.4 2.4' "$tmp/golden" "$tmp/actual" >"$tmp/hashes"
: >"$tmp/movers"
while read -r id ghash ahash; do
    [[ $ghash == "$ahash" ]] || echo "$id" >>"$tmp/movers"
done <"$tmp/hashes"
sort -o "$tmp/movers" "$tmp/movers"   # lexical, the collation comm below expects
[[ -z $movers_out ]] || sort -n "$tmp/movers" >"$movers_out"

ncompared=$(wc -l <"$tmp/hashes")
nmovers=$(wc -l <"$tmp/movers")
echo "check_digest: $ncompared cases compared, $nskipped golden-only (skipped), $nmovers movers"
if [[ -s $tmp/movers ]]; then
    echo "movers: $(join -t' ' -j1 -o '1.1 2.2 2.3' "$tmp/movers" "$tmp/names" |
        sort -n | tr '\n' ' ' | cut -c1-400)"
fi

sort "$expect" >"$tmp/expect"   # lexical, matching $tmp/movers
unexpected=$(comm -23 "$tmp/movers" "$tmp/expect")
unfired=$(comm -13 "$tmp/movers" "$tmp/expect")
if [[ -n $unexpected || -n $unfired ]]; then
    echo "check_digest: FAIL: mover set != expectation" >&2
    [[ -z $unexpected ]] || echo "  unexpected movers: $(echo "$unexpected" | tr '\n' ' ')" >&2
    [[ -z $unfired ]] || echo "  expected but did not move: $(echo "$unfired" | tr '\n' ' ')" >&2
    exit 1
fi
exit 0
