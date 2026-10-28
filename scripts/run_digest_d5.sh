#!/usr/bin/env bash
# D5 (c) and (d): the al64 digest driver across the single-ISA levels of the merge base and
# the tier runs of the dispatched library on one host with AVX-512.
#
# Builds three drivers natively (x86-64 baseline codegen):
#   drv_single: merge-base headers + public API; run against each single-ISA libadmiral L(1..4)
#   drv_disp:   branch headers + public API; run against the dispatched lib (picks v4 here)
#   drv_tier:   branch headers + test/tier_select.cpp linked to libadmiral_internal; ADM_TEST_TIER
# comparisons through scripts/check_digest.sh:
#   tier_v1 vs single_v1, tier_v3 vs single_v3, tier_v4 vs single_v4 and disp vs single_v4: 0 movers
#   controls: tier_v3 vs single_v4 movers; tier_v1 vs single_v3 movers; tier_v3 --ulp 700: 1 mover
#   (d): the OFF build at x86-64-v3 vs the base at x86-64-v3: 0 movers
#
# usage: run_digest_d5.sh DYN_WORKTREE BASE_WORKTREE DYN_BUILD BASE_BUILDS_X86_64 BASE_BUILDS_V3 BASE_BUILDS_V4
set -u
DYN=${1:?} BASE=${2:?} DBUILD=${3:?} SV1=${4:?} SV3=${5:?} SV4=${6:?}
CHK=$DYN/scripts/check_digest.sh
X="-isystem /home/mbarbone/localcache/cpm/xsimd/59f31c8a81dd43b8197c75f54e86e75979f505df/include -isystem /home/mbarbone/localcache/cpm/poet/ff3c9ca8409e1f5dc3a6e64edde538bd964e4d3f/include"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
CXX="g++ -O3 -DNDEBUG -std=c++20 -I$DYN/test/digest"
$CXX -I$BASE/include $DYN/test/digest/digest.cpp -L$SV1/src -ladmiral -o $T/drv_single || exit 1
$CXX -I$DYN/include $DYN/test/digest/digest.cpp -L$DBUILD/src -ladmiral -o $T/drv_disp || exit 1
$CXX -DADM_DISPATCH -DADM_STATIC_DEFINE -I$DYN/src -I$DYN/include $X \
    $DYN/test/digest/digest.cpp $DYN/test/tier_select.cpp -L$DBUILD/src -ladmiral_internal \
    -o $T/drv_tier || exit 1
fail=0
run() { # out libdir tier binary [args]
    local out=$T/$1 lib=$2 tier=$3; shift 3
    LD_LIBRARY_PATH=$lib ADM_TEST_TIER=$tier "$@" > $out.txt 2> $out.err; local rc=$?
    echo "run $1: rc=$rc tier=${tier:-unset} $(grep -h 'engine arch' $out.err 2>/dev/null)"
    ((rc == 0)) || fail=1
    return 0
}
run single_v1 $SV1/src "" $T/drv_single
run single_v3 $SV3/src "" $T/drv_single
run single_v4 $SV4/src "" $T/drv_single
run disp_native $DBUILD/src "" $T/drv_disp
for t in v1 v3 v4; do run tier_$t $DBUILD/src $t $T/drv_tier; done
run tier_v3_ulp700 $DBUILD/src v3 $T/drv_tier --ulp 700
# (d) OFF at v3: the OFF build's digest driver against the base v3 driver
$CXX -I$DYN/include $DYN/test/digest/digest.cpp -L$DYN/build-off-v3/src -ladmiral -o $T/drv_off_v3 || exit 1
run off_v3 $DYN/build-off-v3/src "" $T/drv_off_v3
cmp_pair() { # golden actual expect
    $CHK $T/$1.txt $T/$2.txt > $T/$2.vs.$1.cmp 2>&1
    local s; s=$(grep -o '[0-9]* cases compared, [0-9]* golden-only (skipped), [0-9]* movers' $T/$2.vs.$1.cmp)
    local nc ns nm; read -r nc _ _ ns _ _ nm _ <<< "$s"
    local v=FAIL
    if [ -n "$s" ] && ((nc > 0 && ns == 0)); then
        case $3 in +) ((nm > 0)) && v=PASS ;; *) ((nm == $3)) && v=PASS ;; esac
    fi
    [ $v = PASS ] || fail=1
    echo "$2 against $1: ${s:-no summary}; expect $3 -> $v"
}
cmp_pair single_v1 tier_v1 0
cmp_pair single_v3 tier_v3 0
cmp_pair single_v4 tier_v4 0
cmp_pair single_v4 disp_native 0
cmp_pair single_v3 off_v3 0
cmp_pair single_v4 tier_v3 +
cmp_pair single_v3 tier_v1 +
cmp_pair single_v3 tier_v3_ulp700 1
echo "verdict: $([ $fail = 0 ] && echo PASS || echo FAIL)"
exit $fail
