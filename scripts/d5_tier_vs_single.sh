#!/usr/bin/env bash
# D5 (b): each renamed tier object against the branch's single-ISA build at that tier, the tier
# prefix removed, inst_api_* excluded (the seam TUs differ by design): the size multisets must
# be equal. Control: the same compare against the NEXT tier's single-ISA build must differ,
# so the compare can fire.
#
# usage: d5_tier_vs_single.sh BUILD SINGLE_TIER_BUILD OTHER_TIER_BUILD v3 [v4]
#   BUILD                      the ADM_DISPATCH=ON build holding tier_vN/
#   SINGLE_TIER_BUILD          single-ISA build at this tier's -march
#   OTHER_TIER_BUILD           single-ISA build at a different -march (the control arm)
#   v3 [v4]                    which tier(s) to compare
set -u
B=${1:?ON build}
S=${2:?single-ISA at tier}
C=${3:?single-ISA control}
shift 3
NM=${NM:-nm}; TR=${TR:-tr}; SED=${SED:-sed}
fail=0
# Seam surface discovered 2026-10-27. Two designed differences, both one-directional:
# - include/admiral/errors.hpp under ADM_DISPATCH gives the error hierarchy a key function, so
#   vtable/typeinfo/dtors are emitted once at the floor (src/errors.cpp); master's form emits
#   them as vague linkage in every TU that includes errors.hpp. Filtered out of both sides.
ERRS='7admiral(5error|10size_error|14internal_error|17unsupported_error)'
# - The six plan-machinery TUs are the only allowed movers, per tier. The seam is the cause,
#   not per-tier statics (route_memo statics exist identically in the single-ISA build):
#   -DADM_DISPATCH changes errors.hpp (the key function above) and -DADM_STATIC_DEFINE empties
#   the ADM_API attribute. The isolated probe proves it: inst_plan_f.cpp compiled with the
#   single-ISA-v3 flags moves its size multiset from the two -Ds alone. Throw paths stop
#   inlining the error dtors, gcc's per-TU inline budget shifts, and with it a handful of glue
#   instantiations (nd_runtime_plan ctor, nd_apply_axis, STL/variant weak copies, std::thread
#   ctors), plus constant-pool and switch-table numbering (.LC*, CSWTCH*). The FFT math does
#   not move: codelet_*, inst_col_* and inst_dif_* show 0 movers in this same compare, and the
#   sampled FP-arithmetic streams of the six (v3+v4, x86 review S2) are identical or differ
#   only by block placement and register allocation. Scalar pricing ops (vcvt/vdivsd) drift by
#   1-2 static copies; result bits are D5 (c)'s job, route choice is a perf matter.
ALLOWED_MOVERS="inst_nd_d.cpp.o inst_nd_f.cpp.o inst_plan_d.cpp.o inst_plan_f.cpp.o inst_real_d.cpp.o inst_real_f.cpp.o"
sizes_of() { # object -> newline table: every defined symbol's size, sorted numerically
    # bsd format: value size type name. $4 nonempty selects the lines that carry a size;
    # the size itself is $2.
    "$NM" --defined-only -S "$1" | grep -vE "$ERRS" | awk '$4!=""{print $2}' | sort -n
}
counterpart() { # single-ISA object holding $1: the engine or the codelets of the base build
    for d in admiral_engine.dir admiral_codelets.dir; do
        [ -f "$2/src/CMakeFiles/$d/$1" ] && { printf '%s\n' "$2/src/CMakeFiles/$d/$1"; return 0; }
    done
    return 1
}
for v in "$@"; do
    pfx="admiral_${v}_"
    moved=0; equal=0; nls=0; ctlmoved=0; unexpected=""
    for ro in "$B/src/tier_$v/"*.o; do
        b=$(basename "$ro")
        case $b in inst_api_*.o|codelet_dispatch.cpp.o) continue ;; esac  # D5 (b) excludes the seam TUs; codelet_dispatch is branch-only
        so=$(counterpart "$b" "$S") || { echo "single-ISA build misses $b"; fail=1; continue; }
        ms=$(sizes_of "$so")
        # The control compare must be able to fire: same object from a build at another tier
        # flag differs in the size multiset.
        co=$(counterpart "$b" "$C") || co=
        if [ -n "$co" ] && [ "$(sizes_of "$co")" != "$ms" ]; then ctlmoved=$((ctlmoved + 1)); fi
        # Sizes only (the rename changes names, by design): the multisets must be equal.
        if [ "$(sizes_of "$ro")" = "$ms" ]; then equal=$((equal + 1));
        else
            moved=$((moved + 1))
            case " $ALLOWED_MOVERS " in *" $b "*) ;; *) unexpected="$unexpected $b" ;; esac
        fi
    done
    echo "tier $v: $equal equal, $moved moved (allowed: plan-machinery seam TUs, unexpected:${unexpected:-none}) control: $ctlmoved differ vs other-tier build"
    [ -z "$unexpected" ] || { echo "FAIL: unexpected movers in tier $v:$unexpected"; fail=1; }
    [ "$ctlmoved" -gt 0 ] || { echo "FAIL: control: no object of tier $v differs against the control build; compare cannot fire"; fail=1; }
done
exit $fail
