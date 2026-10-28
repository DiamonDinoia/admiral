#!/usr/bin/env bash
# D2: the dispatched x86-64 library's checks.
#
# usage: check_dispatch.sh BUILD [OFF_BUILD] [CLAUSES...]
#          BUILD       an ADM_DISPATCH=ON build tree (compile_commands.json present)
#          OFF_BUILD   an ADM_DISPATCH=OFF build tree at x86-64 of the same source (check (d))
#          CLAUSES     subset of a b c d e f g h i, default: all that this platform supports
#        check_dispatch.sh controls BUILD [CLAUSES...]
#          runs each check's committed positive control and fails when a control cannot fire.
#
# Env: XED64 (check e; skip when absent), NM/OBJDUMP/AR default to llvm-nm/llvm-objdump or
# readelf/ar on PATH. XED also needs the SDE kit. (g) needs python3.
#
# Clauses (fails when unmet):
#  (a) a global that a renamed tier object defines is also defined by the floor, the shared TUs
#      or the other tier (tier_entry excepted), over the objects and over the members of each
#      .a; or a COMDAT group signature of a tier object lacks the tier prefix (ELF); or a
#      member of a .a is not byte-equal to an object of floor.list or a renamed tier object
#  (b) a tier object defines _ZTI, _ZTS or _ZTV of a type in admiral outside detail
#  (c) a library holds other than 9 tier_entry call operators, or no name with a tier's prefix
#  (d) the exports of a surface library differ from the OFF build, or name tier_entry
#  (e) an object holds an ISA set outside its class's allow-list; the floor holds other than 3
#      xgetbv; a decode error; an empty class (scripts/check_xed.py)
#  (f) the undefined names of a surface .so or admiral_internal name an admiral symbol, a
#      tier_entry or a tier prefix (ELF)
#  (g) floor TU carries a tier flag, tier TU lacks its tier flag, or a col or inst_api_ld TU of
#      any tier lacks its numerics pin, in compile_commands.json
#  (h) a tier object holds an init section, or a test object other than tier_select.cpp.o has
#      init code that names an admiral symbol
#  (i) a renamed COFF object changes the value or the storage class of @feat.00
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

mode=check
[ "${1:-}" = controls ] && { mode=controls; shift; }
BUILD="${1:?usage: check_dispatch.sh [controls] BUILD [OFF_BUILD] [CLAUSES...]}"; shift || true
BUILD=$(cd "$BUILD" && pwd)   # absolute: the archive audit extracts inside temp dirs
OFF=""
if [ $# -ge 1 ] && [ -f "$1/compile_commands.json" ]; then OFF=$(cd "$1" && pwd); shift; fi
CLAUSES="$*"
# A word outside a..i is a typo or a wrong OFF_BUILD path: it would run chk_<word>, fail the
# shell lookup and still exit 0 (x86 review S1). Fail loud, run nothing.
for c in $CLAUSES; do
    case $c in a|b|c|d|e|f|g|h|i) ;; *)
        echo "check_dispatch.sh: '$c' is not a clause name (a..i); wrong OFF_BUILD path?" >&2
        exit 1 ;;
    esac
done
S=$BUILD/src
NM=${NM:-llvm-nm}
OBJDUMP=${OBJDUMP:-llvm-objdump}
command -v "$OBJDUMP" >/dev/null || OBJDUMP=readelf
if [ "$(uname -s)" = Linux ] && command -v readelf >/dev/null; then READELF=readelf; else READELF=$OBJDUMP; fi
AR=${AR:-ar}   # GNU ar: llvm-ar lacks the xN instance modifier the .a member audit needs
command -v "$AR" >/dev/null || AR=llvm-ar
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
say() { echo "  $*"; }
bad() { echo "FAIL: $*"; fail=1; }
need_list() { [ -s "$1" ] || bad "$1 missing or empty"; }
have() { [ -z "$CLAUSES" ] || [[ " $CLAUSES " == *" $1 "* ]]; }
os=$(uname -s)

tier_prefix() { echo "admiral_$1_"; }
tier_objs() { ls "$S/tier_$1/"*"$OBJ" 2>/dev/null; }        # renamed objects
tier_raw() { grep -v '^$' "$S/tier_$1.list" 2>/dev/null; } # pre-rename objects per the build
floor_objs() { grep -v '^$' "$S/floor.list" 2>/dev/null; }
OBJ=$([ "$os" = Linux -o "$os" = Darwin ] && echo .o || echo .obj)
gdef() { "$NM" --defined-only --extern-only --format=just-symbols "$@" | grep -v -e ':$' -e '^$' | sort -u; }

##### (a) isolation over objects and archive members; group signatures (ELF)
chk_a() {
  echo "== (a) isolation: no tier global is defined by the floor, the shared TUs or the other tier"
  local SHARED FL
  SHARED=$(floor_objs | grep -e '/api.cpp' -e '/errors.cpp' -e '/scratch_alloc.cpp' -e '/large_route_probe.cpp')
  FL=$(floor_objs | grep -v -e '/api.cpp' -e '/errors.cpp' -e '/scratch_alloc.cpp' -e '/large_route_probe.cpp')
  gdef $FL > $T/floor.g; gdef $SHARED > $T/shared.g
  [ -s $T/floor.g ] || bad "floor census is empty"
  local v o n u
  for v in v3 v4; do
    need_list "$S/tier_$v.list"
    gdef $(tier_objs $v) > $T/ren_$v.g
    n=$(wc -l < $T/ren_$v.g); [ "$n" -gt 0 ] || bad "tier $v: no renamed objects or no symbols"
    o=$([ $v = v3 ] && echo v4 || echo v3)
    for side in floor shared $o; do
      comm -12 $T/ren_$v.g $T/$side.g 2>/dev/null | grep -v tier_entry > $T/iso.$v.$side || true
      n=$(wc -l < $T/iso.$v.$side)
      say "$v vs $side: $n shared names (tier_entry excluded)"
      [ "$n" -eq 0 ] || { head -3 $T/iso.$v.$side | c++filt | sed 's/^/    /'; bad "tier $v shares a global with $side"; }
    done
    u=$(grep -v tier_entry $T/ren_$v.g | grep -vc "^$(tier_prefix $v)" || true)
    say "$v: $(grep -c tier_entry $T/ren_$v.g) tier_entry names, $(grep -vc tier_entry $T/ren_$v.g) other globals, unprefixed: $u"
    [ "$u" -eq 0 ] || bad "tier $v defines a global without the $(tier_prefix $v) prefix"
    if [ "$os" = Linux ] && [ "$READELF" = readelf ]; then
      readelf -gW $(tier_objs $v) 2>/dev/null | grep -o "COMDAT group section \[[ 0-9]*\] [\`'][^']*' \[[^]]*\]" |
        sed 's/.*\[\(.*\)\]$/\1/' | sort -u > $T/grp_$v.g
      n=$(wc -l < $T/grp_$v.g); u=$(grep -v -e "^$(tier_prefix $v)" -e tier_entry $T/grp_$v.g | wc -l)
      say "$v: $n COMDAT group signatures, $u without the tier prefix"
      [ "$n" -gt 0 ] || bad "tier $v: group census broken"
      [ "$u" -eq 0 ] || bad "tier $v: a COMDAT group signature lacks the prefix"
    elif [ "$os" = Linux ]; then
      bad "tier $v: no readelf for the COMDAT group census"
    fi
  done
  if [ "$os" = Linux -o "$os" = Darwin ]; then
    for A in "$S/libadmiral.a"; do
      echo "-- (a) members of $(basename $A)"
      rm -rf $T/ar; mkdir -p $T/ar
      (cd $T/ar && "$AR" t "$A" | sort | uniq -c | while read -r c m; do
        for ((i = 1; i <= c; i++)); do mkdir -p "d$i"; (cd "d$i" && "$AR" xN "$i" "$A" "$m"); done
      done)
      local M; M=$(ls $T/ar/d*/*.o 2>/dev/null)
      n=$(echo "$M" | grep -c .)
      say "members extracted: $n; archive lists $("$AR" t "$A" | wc -l)"
      [ "$n" -eq "$("$AR" t "$A" | wc -l)" ] || bad "archive extraction count differs"
      { sha256sum $FL $SHARED $(tier_objs v3) $(tier_objs v4) | awk '{print $1}'; } | sort > $T/listed.sha
      sha256sum $M | awk '{print $1}' | sort > $T/members.sha
      n=$(comm -23 $T/members.sha $T/listed.sha | wc -l)
      say "members not byte-equal to a listed object: $n"
      [ "$n" -eq 0 ] || bad "an archive member is byte-foreign: neither floor nor renamed tier"
    done
  fi
}
ctl_a() {
  echo "== controls (a): the pre-rename objects must share names with the floor and each other"
  local SHARED FL nc n v o
  SHARED=$(floor_objs | grep -e '/api.cpp' -e '/errors.cpp' -e '/scratch_alloc.cpp' -e '/large_route_probe.cpp')
  FL=$(floor_objs | grep -v -e '/api.cpp' -e '/errors.cpp' -e '/scratch_alloc.cpp' -e '/large_route_probe.cpp')
  gdef $FL > $T/floor.g
  [ -s $T/floor.g ] || bad "floor census is empty"
  for v in v3 v4; do
    gdef $(tier_objs $v) > $T/ren_$v.g
    gdef $(tier_raw $v) > $T/raw_$v.g
    n=$(comm -12 $T/raw_$v.g $T/floor.g | grep -vc tier_entry); say "raw $v vs floor: $n shared"
    [ "$n" -gt 0 ] || bad "control: raw tier $v shares nothing with the floor; isolation cannot fire"
    o=$([ $v = v3 ] && echo v4 || echo v3)
    gdef $(tier_raw $o) > $T/raw_$o.g
    n=$(comm -12 $T/raw_$v.g $T/raw_$o.g | grep -vc tier_entry); say "raw $v vs raw $o: $n shared"
    [ "$n" -gt 0 ] || bad "control: raw tiers share nothing; isolation cannot fire"
  done
  if [ "$os" = Linux ]; then
    for v in v3 v4; do
      readelf -gW $(tier_raw $v) 2>/dev/null | grep -o "COMDAT group section \[[ 0-9]*\] [\`'][^']*' \[[^]]*\]" |
        sed 's/.*\[\(.*\)\]$/\1/' | sort -u > $T/grpraw_$v.g
      nc=$(grep -v -e "^$(tier_prefix $v)" -e tier_entry $T/grpraw_$v.g | wc -l)
      say "raw $v group signatures without the prefix: $nc"
      [ "$nc" -gt 0 ] || bad "control: raw $v groups are all prefixed; the group clause cannot fire"
      { sha256sum $FL $SHARED $(tier_objs v3) $(tier_objs v4) | awk '{print $1}'; } | sort > $T/listed.sha
      nc=$(sha256sum $(tier_raw $v) | awk '{print $1}' | sort | comm -12 - $T/listed.sha | wc -l)
      say "raw $v objects byte-equal to a shipped object: $nc (0: a raw object in the archive fires)"
      [ "$nc" -eq 0 ] || bad "control: a raw $v object is byte-equal to a listed object; byte-equality cannot fire"
    done
  fi
}

##### (b) no tier object defines _ZTI/_ZTS/_ZTV of a type in admiral outside detail
chk_b() {
  echo "== (b) error-class typeinfo out of tiers"
  local v hits
  for v in v3 v4; do
    hits=$(gdef $(tier_objs $v) | grep -E '^_ZT[ISV]N7admiral' | grep -v '^_ZT[ISV]N7admiral6detail' || true)
    n=$(echo "$hits" | grep -c . || true); [ -z "$hits" ] && n=0
    say "$v: $n _ZTI/_ZTS/_ZTV of admiral types outside detail"
    [ "$n" -eq 0 ] || { echo "$hits" | c++filt | head -3 | sed 's/^/    /'; bad "tier $v defines error typeinfo"; }
  done
  gdef $(floor_objs | grep /errors.cpp) | grep -q '^_ZTIN7admiral5errorE$' ||
    bad "errors.cpp.o does not define _ZTIN7admiral5errorE (the key function)"
  say "errors.cpp.o defines _ZTIN7admiral5errorE"
}
# Compile a control fixture with the exact flags of one tier TU (first v3 entry in
# compile_commands.json), substituting the source and output paths: the tier flags, include
# dirs and defines ride along verbatim. Prints nothing on success; returns 1 on failure.
control_build() { # source out
  python3 - "$BUILD" "$SRC" > $T/control_build.cmd <<'EOF'
import json, sys, shlex
build = sys.argv[1]
row = None
for e in json.load(open(build + "/compile_commands.json")):
    if "/admiral_engine_v3.dir/" in e.get("output", ""):
        row = e
        break
assert row, "no v3 TU row"
args = shlex.split(row["command"])
out, skip = [], False
for a in args[1:]:
    if skip:
        skip = False
        continue
    if a in ("-c", "-o", "-MF", "-MT"):
        skip = True          # drop the flag and its operand; the control sets its own -c/-o
        continue
    if a == "-MD":
        continue             # standalone dep-generation flag, no operand
    if a == row["file"] or a.endswith(".cpp"):
        continue             # the tier TU's own source
    out.append(a)
print(" ".join(shlex.quote(x) for x in out))
EOF
  local cc; cc=$(command -v "${CXX:-c++}" || command -v g++)
  "$cc" $(cat $T/control_build.cmd) -c "$1" -o "$2" 2> $T/control_build.log ||
    { head -5 $T/control_build.log; echo "FAIL: control build of $1 failed"; return 1; }
  return 0
}

ctl_b() {
  echo "== controls (b): inline_error.cpp (inline thrown class) compiled as a tier TU must fire"
  control_build "$SRC/test/static_analysis/inline_error.cpp" $T/inline_error$OBJ || { fail=1; return 0; }
  local n
  n=$(gdef $T/inline_error$OBJ | grep -cE '^_ZT[ISV]N7admiral12inline_errorE' || true)
  say "control object defines $n _ZTI/_ZTS/_ZTV of admiral::inline_error"
  [ "$n" -gt 0 ] || bad "control: (b) cannot fire"
}

##### (c) 9 tier_entry call operators per library; tier prefixes present
count_ops() { # library -> number of tier_entry call operators (not the operator's ::ops static)
  nm --defined-only "$1" | grep tier_entry | c++filt | grep 'tier_entry::operator()<' |
    grep -cv '::ops'
}
chk_c() {
  echo "== (c) tier_entry call operators and tier prefixes per library"
  local l ext
  for l in admiral admiral_c admiral_fftw admiral_internal; do
    for ext in so a; do
      [ -f "$S/lib${l}.${ext}" ] || continue
      local n v3n v4n
      n=$(count_ops "$S/lib${l}.${ext}")
      v3n=$(nm --defined-only "$S/lib${l}.${ext}" | grep -c "$(tier_prefix v3)" || true)
      v4n=$(nm --defined-only "$S/lib${l}.${ext}" | grep -c "$(tier_prefix v4)" || true)
      say "lib${l}.${ext}: $n tier_entry call operators, $v3n v3-prefixed, $v4n v4-prefixed names"
      { [ "$n" -eq 9 ] && [ "$v3n" -gt 0 ] && [ "$v4n" -gt 0 ]; } ||
        bad "lib${l}.${ext}: wrong tier census"
    done
  done
}
ctl_c() {
  echo "== controls (c): the v3 tier alone must read as other than 9"
  need_list "$S/tier_v3.list"
  local n
  n=$( { tier_raw v3; } | xargs "$AR" qc $T/v3_only.a 2>/dev/null && count_ops $T/v3_only.a)
  say "operators counted over the v3 tier alone: $n (must differ from 9, so a dropped arch fires)"
  [ "$n" != 9 ] || bad "control: (c) cannot fire on a partial tier set"
}

##### (d) exports equal the OFF build; no tier_entry or tier prefix exported
surface_exports() {
  if [ "$os" = Linux ]; then nm -D --defined-only "$1" | awk '{print $NF}' | sort -u
  else "$NM" --defined-only --extern-only --format=just-symbols "$1" | grep -v -e '^$' -e ':$' | sort -u; fi
}
chk_d() {
  echo "== (d) surface exports equal the OFF build"
  [ -n "$OFF" ] || { echo "-- (d) SKIPPED: no OFF build given"; return 0; }
  local l on off
  for l in admiral admiral_c admiral_fftw; do
    on="$S/lib${l}.so"; off="$OFF/src/lib${l}.so"
    [ -f "$on" ] || continue
    if [ ! -f "$off" ]; then off="$OFF/src/lib${l}.dylib"; on="$S/lib${l}.dylib"; fi
    surface_exports "$on" > $T/exp_on.$l
    surface_exports "$off" > $T/exp_off.$l
    n=$(comm -3 $T/exp_off.$l $T/exp_on.$l | wc -l)
    te=$(grep -c -e tier_entry -e "$(tier_prefix v3)" -e "$(tier_prefix v4)" $T/exp_on.$l || true)
    say "lib${l}: $(wc -l < $T/exp_on.$l) exports, $n differ from OFF, $te tier-like"
    { [ "$n" -eq 0 ] && [ "$te" -eq 0 ]; } ||
      { comm -3 $T/exp_off.$l $T/exp_on.$l | head -5 | sed 's/^/    /'; bad "lib${l} exports differ from OFF or name a tier"; }
    grep -q -e "$(tier_prefix v3)" -e "$(tier_prefix v4)" "$SRC/src/${l}.map" 2>/dev/null &&
      bad "${l}.map globs a tier prefix" || true
  done
}
ctl_d() {
  echo "== controls (d): an adm_v3_-style export added to the ON list must fire"
  [ -n "$OFF" ] || { echo "-- controls (d) SKIPPED: no OFF build given"; return 0; }
  { cat $T/exp_on.admiral_c 2>/dev/null || surface_exports "$S/libadmiral_c.so"; echo "adm_v3_control_probe"; } | sort -u > $T/exp_doctored
  n=$(surface_exports "$OFF/src/libadmiral_c.so" > $T/exp_off.c; comm -3 $T/exp_off.c $T/exp_doctored | wc -l)
  say "doctored export list differs by $n (the R4 leak: adm_* catches adm_v3_*)"
  [ "$n" -gt 0 ] || bad "control: the (d) comparator cannot fire"
}

##### (e) XED ISA-set census
chk_e() {
  echo "== (e) XED ISA sets per class"
  [ -n "${XED64:-}" ] || { echo "-- (e) SKIPPED: XED64 unset"; return 0; }
  command -v as >/dev/null || bad "as missing (control probe needs it)"
  as "$SRC/test/static_analysis/xed_probe.s" -o $T/xed_probe.o || bad "control probe fails to assemble"
  python3 -I "$SRC/scripts/check_xed.py" "$XED64" $T/xed_probe.o \
    floor="$S/floor.list" v3="$S/tier_v3.list" v4="$S/tier_v4.list" || bad "XED census failed"
}
ctl_e() {
  echo "== controls (e): built into check_xed.py (v3 under floor list, v4 under v3 list, probe)"
  [ -n "${XED64:-}" ] || { echo "-- controls (e) SKIPPED: XED64 unset"; return 0; }
}

##### (f) no undefined admiral/tier names in the shipped .so's (ELF)
chk_f() {
  echo "== (f) undefined names of each surface .so and admiral_internal"
  [ "$os" = Linux ] || { echo "-- (f) SKIPPED: not ELF"; return 0; }
  local l n
  for l in admiral admiral_c admiral_fftw admiral_internal; do
    n=$(nm -D --undefined-only "$S/lib${l}.so" | grep -c -e 'admiral' -e tier_entry || true)
    say "lib${l}.so: $n undefined admiral/tier names"
    [ "$n" -eq 0 ] || { nm -D --undefined-only "$S/lib${l}.so" | grep -e admiral -e tier_entry | head -3; bad "lib${l}.so has undefined admiral names"; }
  done
}
ctl_f() {
  echo "== controls (f): a .so from the floor objects alone must leave 6 tier names undefined"
  [ "$os" = Linux ] || { echo "-- controls (f) SKIPPED: not ELF"; return 0; }
  local n
  { c++ -shared $(floor_objs) -o $T/floor_only.so 2>$T/floor_only.link && [ -f $T/floor_only.so ]; } ||
    { head -3 $T/floor_only.link; bad "control link of the floor-only .so failed"; }
  n=$(nm -D --undefined-only $T/floor_only.so | grep -c tier_entry)
  say "floor-only .so undefined tier_entry names: $n (6 expected: v3 and v4 f/d/ld)"
  [ "$n" -eq 6 ] || bad "control: the (f) census reads $n, not 6"
}

##### (g) compile_commands.json flags audit
g_py() { python3 - "$@" <<'EOF'
import json, sys, re
build, ccj, colre, ldre = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
tierflag = {"floor": re.compile(r"-march=x86-64-v[34]\b|/arch:AVX(2|512)\b", re.I),
            "v3": re.compile(r"(-march=x86-64-v3\b|/arch:AVX2\b)", re.I),
            "v4": re.compile(r"(-march=x86-64-v4\b|/arch:AVX512\b)", re.I)}
bad = mid = 0
for e in ccj:
    out, cmd = e.get("output", ""), e["command"]
    cls = None
    for tag, cls_dir in (("v3", "admiral_engine_v3.dir"), ("v4", "admiral_engine_v4.dir")):
        if "/%s/" % cls_dir in out or out.endswith(cls_dir + ".o"):
            cls = tag
    if any(x in out for x in ("admiral_engine.dir/", "admiral_codelets.dir/")):
        cls = "floor"
    if cls is None:
        continue
    base = out.rsplit("/", 1)[-1]
    if cls == "floor":
        if tierflag["floor"].search(cmd):
            print(f"FAIL: floor TU carries a tier flag: {base}"); bad += 1
        continue
    if not tierflag[cls].search(cmd):
        print(f"FAIL: {cls} TU lacks its tier flag: {base}"); bad += 1
    if re.match(r"inst_col_.*\.cpp", base) and not re.search(colre, cmd):
        print(f"FAIL: {cls} col TU {base} lacks the col pin"); bad += 1
    if base == "inst_api_ld.cpp.o" or base == "inst_api_ld.cpp.obj":
        if ldre != "skip" and not (re.search(ldre, cmd) and re.search(r"-ffp-contract=off", cmd)):
            print(f"FAIL: {cls} inst_api_ld lacks the strict-FP set"); bad += 1
    mid += 1
print(f"flag audit: {mid} engine/codelet TUs checked")
sys.exit(1 if bad or mid == 0 else 0)
EOF
}
chk_g() {
  echo "== (g) flags per class from compile_commands.json"
  local colre ldre
  if [ "$os" = Linux -o "$os" = Darwin ]; then colre='-ffp-contract=on'; ldre='-fno-fast-math'
  else colre='/fp:precise'; ldre='skip'; fi  # msvc: col pin is /fp:precise; no strict-FP flags
  g_py "$BUILD" "$BUILD/compile_commands.json" "$colre" "$ldre" || bad "(g) flag audit failed"
}
ctl_g() {
  echo "== controls (g): a doctored compile_commands.json (floor TU given a tier flag) must fail"
  python3 - "$BUILD/compile_commands.json" "$T/ccj_doctored.json" <<'EOF'
import json, sys
ccj = json.load(open(sys.argv[1]))
hit = 0
for e in ccj:
    if "/admiral_engine.dir/" in e.get("output", ""):
        e["command"] += " -march=x86-64-v3"; hit += 1; break
json.dump(ccj, open(sys.argv[2], "w"))
assert hit
EOF
  local colre='-ffp-contract=on' ldre='-fno-fast-math'
  [ "$os" = Linux -o "$os" = Darwin ] || { colre='/fp:precise'; ldre='skip'; }
  if g_py "$BUILD" "$T/ccj_doctored.json" "$colre" "$ldre" > $T/ctl_g.out 2>&1; then
    cat $T/ctl_g.out; bad "control: (g) accepted a floor TU with a tier flag"
  else
    say "doctored json rejected: $(grep -c '^FAIL' $T/ctl_g.out) failures"
  fi
}

##### (h) no init sections in tier objects; init code naming admiral only in tier_select.cpp.o
chk_h() {
  echo "== (h) init sections"
  [ "$os" = Linux -o "$os" = Darwin ] || { echo "-- (h) SKIPPED: objdump dialect unknown here"; return 0; }
  # Section-name match without regex escapes: prefix families plus two exact names.
  sec_is_init() {
    case "$1" in
      __mod_init_func|__init_offsets) return 0 ;;
      .init_array|.init_array.*|.ctors|.ctors.*|.preinit_array|.preinit_array.*) return 0 ;;
    esac
    [ "${1#.CRT\$XC}" != "$1" ]
  }
  local n=0 v o sec hit
  for v in v3 v4; do
    for o in $(tier_objs $v); do
      hit=1
      for sec in $($OBJDUMP -h "$o" | awk '{print $2}'); do
        sec_is_init "$sec" && { hit=0; break; }
      done
      [ "$hit" -eq 0 ] && { echo "FAIL: init section in tier object $o"; n=$((n + 1)); }
    done
  done
  say "tier objects scanned: $(( $(tier_objs v3 | wc -l) + $(tier_objs v4 | wc -l) )), with an init section: $n"
  [ "$n" -eq 0 ] || bad "a tier object holds an init section"
  if [ "$os" = Linux ]; then
    find "$BUILD/test/CMakeFiles" -name "*$OBJ" -exec python3 -I "$SRC/scripts/init_refs.py" {} + || bad "(h) test objects init refs failed"
  fi
}
ctl_h() {
  echo "== controls (h): init_plan.cpp (namespace-scope plan) as a tier object must fire"
  [ "$os" = Linux -o "$os" = Darwin ] || { echo "-- controls (h) SKIPPED"; return 0; }
  control_build "$SRC/test/static_analysis/init_plan.cpp" $T/init_plan$OBJ || { fail=1; return 0; }
  local sec hit n
  hit=1
  for sec in $($OBJDUMP -h $T/init_plan$OBJ | awk '{print $2}'); do
    case "$sec" in
      __mod_init_func|__init_offsets|.init_array|.init_array.*|.ctors|.ctors.*|.preinit_array|.preinit_array.*) hit=0; break ;;
    esac
    [ "${sec#.CRT\$XC}" != "$sec" ] && { hit=0; break; }
  done
  if [ "$hit" -eq 0 ]; then
    say "control object holds an init section: the tier-side detector fires"
  else
    bad "control: init_plan.cpp has no init section; the tier-side (h) cannot fire"
  fi
  if [ "$os" = Linux ]; then
    if python3 -I "$SRC/scripts/init_refs.py" $T/init_plan$OBJ > $T/ctl_h.out 2>&1; then
      cat $T/ctl_h.out; bad "control: init_refs.py passed on init_plan.o; the test-side (h) cannot fire"
    else
      say "init_refs.py fails on init_plan.o as required: $(tail -2 $T/ctl_h.out | head -1)"
    fi
  fi
}

##### (i) @feat.00 (COFF)
chk_i() {
  echo "== (i) @feat.00 value and class unchanged by the rename (COFF)"
  [ "$os" = Linux -o "$os" = Darwin ] && { echo "-- (i) SKIPPED: not COFF"; return 0; }
  local v raw ren
  for v in v3 v4; do
    n=0
    for o in $(tier_raw $v); do b=$(basename "$o"); r="$S/tier_$v/$b"
      rv=$("$NM" "$o" | awk '$3=="@feat.00"{print $2}')
      [ -n "$rv" ] || continue
      sv=$(llvm-readobj --symbols "$o" | awk '/Name: @feat.00/{f=1} f&&/StorageClass/{print $2;exit}')
      rv2=$(llvm-readobj --symbols "$o" | awk '/Name: @feat.00/{f=1} f&&/Value/{print $2;exit}')
      sv2=$(llvm-readobj --symbols "$r" | awk '/Name: @feat.00/{f=1} f&&/StorageClass/{print $2;exit}')
      rv3=$(llvm-readobj --symbols "$r" | awk '/Name: @feat.00/{f=1} f&&/Value/{print $2;exit}')
      { [ "$sv" = "$sv2" ] && [ "$rv2" = "$rv3" ]; } || bad "@feat.00 changed: $o ($sv/$rv2) -> $r ($sv2/$rv3)"
      n=$((n + 1))
    done
    say "$v: $n objects carried @feat.00 unchanged"
  done
}

case $mode in
  check)   for c in ${CLAUSES:-a b c d e f g h i}; do chk_$c; done ;;
  controls)
    # Controls run after the main checks' caches; do the census first like chk_a/b do.
    # No ctl_i: clause (i) has no committed control.
    for c in ${CLAUSES:-a b c d e f g h}; do ctl_$c; done ;;
esac
echo "$mode verdict: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
