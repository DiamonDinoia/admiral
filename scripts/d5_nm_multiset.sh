#!/usr/bin/env bash
# D5 (a): OFF keeps master's objects. Compares the nm --defined-only -S output of the engine
# code objects of two single-ISA builds object by object: the OFF build must carry master's
# code verbatim, inst_api_* included, because the seam compiles out.
#
# usage: d5_nm_multiset.sh MASTER_BUILD OFF_BUILD
# Prints per-object equal/moved counts; exit 1 when any engine object differs.
set -u
M=${1:?master build} O=${2:?OFF build}
NM=${NM:-nm}
# Text identity per object: compare the full nm -S sorted symbol list, not only totals, so a
# size-preserving symbol rename still counts as a mover.
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0; moved=0; equal=0
for o in "$M"/src/CMakeFiles/admiral_engine.dir/*.o; do
    b=$(basename "$o")
    p="$O/src/CMakeFiles/admiral_engine.dir/$b"
    if [ ! -f "$p" ]; then echo "MISSING in OFF: $b"; fail=1; continue; fi
    "$NM" --defined-only -S "$o" | sort > "$T/m.txt"
    "$NM" --defined-only -S "$p" | sort > "$T/o.txt"
    if cmp -s "$T/m.txt" "$T/o.txt"; then equal=$((equal + 1)); else moved=$((moved + 1)); fail=1; fi
done
extra=$(for o in "$O"/src/CMakeFiles/admiral_engine.dir/*.o; do b=$(basename "$o"); [ -f "$M/src/CMakeFiles/admiral_engine.dir/$b" ] || echo "$b"; done | wc -l)
echo "engine objects: $equal equal, $moved moved, $extra extra-in-OFF"
[ "$moved" = 0 ] && [ "$extra" = 0 ] || fail=1
exit $fail
