#!/usr/bin/env bash

# compile_time_census.sh -- per-TU compile-time census and clang -ftime-trace attribution.
#
# METHOD
#   table BUILD_DIR
#     Parse BUILD_DIR/.ninja_log (format v4/v5: start_ms, end_ms, mtime, output, hash).
#     Duplicate entries for one output are rebuilds; the LAST entry wins, matching
#     ninja's own log semantics. Per-TU wall is (end - start). Times are honest only
#     when the build ran serially (-j1) under flock on the quiet box; at -jN each
#     entry includes scheduling interference and the table is attribution-grade no
#     longer. Emits group aggregates (n, sum, max) and the per-TU table sorted by
#     seconds, plus a metadata header (host, date, compiler, -march) parsed back out
#     of compile_commands.json so a table can never be silently attached to the
#     wrong build.
#   trace FILE_OR_DIR [...]
#     Parse clang -ftime-trace JSON (one per object, next to the .o). For each TU:
#     total / frontend / backend split from the "Total *" events, template
#     instantiation mass (Instantiate{Function,Class,Variable} events; all inside
#     the frontend window), a per-family instantiation census over args.detail
#     (counts + summed microseconds), and the top instantiation details so a family
#     verdict never rests on a regex alone.
#   control BUILD_DIR
#     Print the measurement-control block the receipts quote: ccache resolution,
#     CMAKE compiler-launcher cache entries, and a `ninja -t commands` grep proving
#     no launcher sits between ninja and the compile. CCACHE_DISABLE=1 is expected
#     in the environment of every measured build; this script checks, not assumes.
#
# Groups (matched on the object basename, first hit wins): codelet_<digit>*,
# inst_col_*, inst_dif_thunks_*, inst_dif_*, inst_gt_/inst_plan_/inst_real_/
# inst_nd_/inst_api_*, path component benchmark/*, path component test/*, else lib.
#
# Controls a caller must still provide: serial builds under flock -s on the quiet
# lock, nice -n19 ionice -c3, NINJA_STATUS unset, fresh build dir per arm, reps
# min-of (never mean) with the control arm first and last.
#
# Exit 0 on success; nonzero on malformed input, missing log, or control failure.

set -euo pipefail

prog=${0##*/}

usage() {
    sed -n '2,33p' "$0"
    exit "${1:-2}"
}

warn() { printf '%s\n' "$prog: $*" >&2; }

# ---------------------------------------------------------------- table mode

ninja_meta() { # build_dir -> key=value lines for the header
    local dir=$1 cxx march
    cxx=$(python3 - "$dir/compile_commands.json" <<'EOF'
import json, sys
try:
    db = json.load(open(sys.argv[1]))
except (OSError, ValueError):
    sys.exit(0)
for ent in db:
    cmd = ent.get("command") or ""
    if cmd:
        print(cmd.split()[0])
        break
EOF
)
    march=$(grep -o -- '-march=[^ "]*' "$dir/compile_commands.json" 2>/dev/null | sort -u |
            head -3 | paste -sd, -)
    printf 'cxx=%s\nmarch=%s\n' "${cxx:-unknown}" "${march:-unknown}"
}

table() {
    local dir=$1 top=$2
    [[ -f $dir/.ninja_log ]] || { warn "no .ninja_log in $dir"; return 1; }
    local ver entries
    ver=$(sed -n 's/# ninja log v//p' "$dir/.ninja_log" | head -1)
    entries=$(grep -cv '^#' "$dir/.ninja_log")
    {
        printf '# %s table  build=%s\n' "$prog" "$dir"
        printf '# host=%s date=%s log_version=%s log_entries=%s\n' \
            "$(hostname -s)" "$(date +%F)" "${ver:-?}" "$entries"
        ninja_meta "$dir" | sed 's/^/# /'
    }
    python3 - "$dir/.ninja_log" "$top" <<'EOF'
import re, sys

path, top = sys.argv[1], sys.argv[2]
top = None if top == "all" else int(top)

# Parse: last entry per output wins (rebuilds append).
last, order = {}, []
with open(path) as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            continue
        try:
            start, end = float(f[0]), float(f[1])
        except ValueError:
            continue
        out = f[3]
        if out not in last:
            order.append(out)
        last[out] = (end - start) / 1000.0


def group(obj):
    leaf = obj.rsplit("/", 1)[-1]
    top = obj.split("/", 1)[0]  # ninja log paths are build-relative: test/, benchmark/...
    if top == "benchmark" or leaf.startswith("bench_"):
        return "bench"
    if top == "test":
        return "test"
    if re.match(r"codelet_\d", leaf):
        return "codelet"
    if leaf.startswith("inst_col_"):
        return "inst_col"
    if leaf.startswith("inst_dif_thunks_"):
        return "inst_dif_thunks"
    if leaf.startswith("inst_dif_"):
        return "inst_dif"
    if re.match(r"inst_(gt|plan|real|nd|api)_", leaf):
        return "inst_other"
    return "lib"


rows = [(dur, group(o), o) for o, dur in last.items()]
rows.sort(reverse=True)
total = sum(r[0] for r in rows)

agg = {}
for dur, grp, obj in rows:
    n, s, mx, arg = agg.get(grp, (0, 0.0, 0.0, ""))
    agg[grp] = (n + 1, s + dur, (dur if dur > mx else mx), (obj if dur > mx else arg))

print("group\tn\tsum_s\tshare_pct\tmax_s\targmax")
for grp in sorted(agg, key=lambda g: -agg[g][1]):
    n, s, mx, arg = agg[grp]
    print(f"{grp}\t{n}\t{s:.1f}\t{100*s/total:.1f}\t{mx:.1f}\t{arg.rsplit('/',1)[-1]}")
print(f"TOTAL\t{len(rows)}\t{total:.1f}\t100.0\t{rows[0][0]:.1f}\t{rows[0][2].rsplit('/',1)[-1]}")
print()
print("rank\tseconds\tshare_pct\tcum_pct\tgroup\tobject")
cum = 0.0
for rank, (dur, grp, obj) in enumerate(rows[:top], 1):
    cum += dur
    print(f"{rank}\t{dur:.1f}\t{100*dur/total:.2f}\t{100*cum/total:.1f}\t{grp}\t{obj}")
EOF
}

# ---------------------------------------------------------------- trace mode

trace() {
    [[ $# -ge 1 ]] || usage
    python3 - "$@" <<'EOF'
import glob, json, os, re, sys

FAMILIES = [  # (key, regex on args.detail), first hit wins
    ("granule", r"\bgranule_"),
    ("codelet_apply", r"codelet_apply"),
    ("col_codelet", r"col_codelet"),
    ("kernel_batched", r"kernel_batched"),
    ("dif_col", r"dif_col"),
    ("iterative_dif", r"iterative_dif|dif_build|dif_passes"),
    ("vpass", r"\bvpass"),
    ("butterfly", r"butterfly|radix_"),
    ("ct_twiddle", r"ct_sincos|twiddle"),
    ("poet", r"static_for|poet::"),
    ("xsimd", r"xsimd::"),
]
INST_EVENTS = {"InstantiateFunction", "InstantiateClass", "InstantiateVariable"}


def load_events(path):
    with open(path) as fh:
        doc = json.load(fh)
    return doc["traceEvents"] if isinstance(doc, dict) else doc


def union_us(intervals):
    """Total microseconds covered by at least one interval. Instantiation and
    parse events nest (a function instantiation opens class instantiations), so
    raw summing counts nested windows several times over; the union is the
    honest 'time with such an event on the stack'."""
    total = end = 0
    for ts, dur in sorted(intervals):
        te = ts + dur
        if ts > end:
            total += dur          # window disjoint from the open one
            end = te
        elif te > end:
            total += te - end     # window extends the open one
            end = te
    return total


def analyse(path):
    ev = load_events(path)
    tot = fe = be = 0.0
    inst_iv, parse_iv = [], []
    fam_iv = {}
    fam_n = {}
    per_detail = {}
    parse_detail = {}
    for e in ev:
        if e.get("ph") != "X":
            continue
        name, dur, ts = e.get("name", ""), e.get("dur", 0), e.get("ts", 0)
        flat = name.replace(" ", "")
        if flat == "TotalExecuteCompiler":
            tot = dur
        elif flat == "TotalFrontend":
            fe = dur
        elif flat == "TotalBackend":
            be = dur
        elif name in INST_EVENTS:
            iv = (ts, dur)
            inst_iv.append(iv)
            detail = (e.get("args") or {}).get("detail", "?")
            for key, pat in FAMILIES:
                if re.search(pat, detail):
                    fam_iv.setdefault(key, []).append(iv)
                    fam_n[key] = fam_n.get(key, 0) + 1
                    break
            c, lst = per_detail.get(detail, (0, []))
            lst.append(iv)
            per_detail[detail] = (c + 1, lst)
        elif name.startswith("Parse"):
            parse_iv.append((ts, dur))
            detail = (e.get("args") or {}).get("detail", "?")
            c, s = parse_detail.get(detail, (0, 0.0))
            parse_detail[detail] = (c + 1, s + dur)

    def rollup(d, limit):
        scored = [(union_us(lst), c, det) for det, (c, lst) in d.items()]
        scored.sort(reverse=True)
        return scored[:limit]

    return {
        "events": len(ev), "total": tot, "fe": fe, "be": be,
        "inst_union": union_us(inst_iv), "n_inst": len(inst_iv),
        "parse_union": union_us(parse_iv),
        "fam": {k: (fam_n[k], union_us(v)) for k, v in fam_iv.items()},
        "top": rollup(per_detail, 15),
        "top_parse": sorted(((s, c, d) for d, (c, s) in parse_detail.items()),
                            reverse=True)[:10],
    }


paths = []
for arg in sys.argv[1:]:
    paths += sorted(glob.glob(os.path.join(arg, "**", "*.json"), recursive=True)) \
        if os.path.isdir(arg) else [arg]
paths = [p for p in paths if os.path.isfile(p)]
if not paths:
    sys.exit("compile_time_census.sh: no trace JSON found")

grand = 0.0
for path in paths:
    a = analyse(path)
    grand += a["total"]
    tot = a["total"] or 1
    fe = a["fe"] or 1
    print(f"# file={path}")
    print(f"# total={a['total']/1e6:.1f}s fe={a['fe']/1e6:.1f}s be={a['be']/1e6:.1f}s "
          f"inst_union={a['inst_union']/1e6:.2f}s parse_union={a['parse_union']/1e6:.2f}s "
          f"n_inst={a['n_inst']} events={a['events']}")
    print("total_s\tfe_pct\tbe_pct\tinst_pct_of_fe\tparse_pct_of_fe\tfe_resid_pct")
    # Parse events wrap instantiation events mid-parse; unions remove nesting.
    print(f"{a['total']/1e6:.1f}\t{100*a['fe']/tot:.1f}\t{100*a['be']/tot:.1f}"
          f"\t{100*a['inst_union']/fe:.1f}\t{100*a['parse_union']/fe:.1f}"
          f"\t{100*(a['fe']-max(a['inst_union'], a['parse_union']))/fe:.1f}")
    print("family\tcount\tunion_s\tshare_of_inst_union_pct")
    for key, _ in FAMILIES:
        if key in a["fam"]:
            c, s = a["fam"][key]
            print(f"{key}\t{c}\t{s/1e6:.2f}"
                  f"\t{100*s/a['inst_union'] if a['inst_union'] else 0:.2f}")
    print(f"<unmatched>\t{a['n_inst'] - sum(c for c,_ in a['fam'].values())}\t-\t-")
    print("top_instantiation_details (count union_s detail)")
    for s, c, detail in a["top"]:
        print(f"{c}\t{s/1e6:.2f}\t{detail[:110]}")
    print("top_parse_details (count sum_s detail)")
    for s, c, detail in a["top_parse"]:
        print(f"{c}\t{s/1e6:.2f}\t{detail[:110]}")
    print()
print(f"# grand_total_s={grand/1e6:.1f} across {len(paths)} traces")
EOF
}

# -------------------------------------------------------------- control mode

control() {
    local dir=$1 rc=0
    printf '# %s control  build=%s  host=%s  date=%s\n' \
        "$prog" "$dir" "$(hostname -s)" "$(date +%F)"
    if command -v ccache >/dev/null; then
        printf 'ccache_on_path=%s\n' "$(command -v ccache)"
        ccache -s | sed 's/^/ccache_stat: /'
    else
        printf 'ccache_on_path=NONE\n'
    fi
    printf 'CCACHE_DISABLE=%s\n' "${CCACHE_DISABLE:-unset}"
    local launches
    launches=$(grep -c 'COMPILER_LAUNCHER' "$dir/CMakeCache.txt" 2>/dev/null || true)
    printf 'cmake_launcher_cache_entries=%s\n' "$launches"
    if [[ -n ${2:-} ]]; then :; fi
    local hits
    hits=$(ninja -C "$dir" -t commands all 2>/dev/null | grep -c ccache || true)
    printf 'ninja_commands_ccache_hits=%s\n' "$hits"
    if [[ ${CCACHE_DISABLE:-0} != 1 || $hits != 0 ]]; then
        warn "control FAILED: cache could poison measured times (CCACHE_DISABLE=${CCACHE_DISABLE:-unset}, hits=$hits)"
        rc=1
    fi
    return $rc
}

# --------------------------------------------------------------------- main

[[ $# -ge 1 ]] || usage
mode=$1; shift
case $mode in
    table)
        top=40
        while [[ ${1:-} == --* ]]; do
            case $1 in
                --top) top=$2; shift 2 ;;
                --all) top=all; shift ;;
                *) usage ;;
            esac
        done
        [[ $# -eq 1 ]] || usage
        table "$1" "$top"
        ;;
    trace)  trace "$@" ;;
    control)
        [[ $# -eq 1 ]] || usage
        control "$1"
        ;;
    *) usage ;;
esac
