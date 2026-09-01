"""Interleaved A/B wall-clock compare of two admiral_benchmark binaries.

Runner timing drifts, so each cell runs back-to-back inside one window, round order
rotates AB/BA, and the answer is the min per binary. Ratios claim direction only
unless the spread columns are small.

usage: ci_perf_ab.py BIN_A BIN_B [--rounds=6] [--grid=ci|full] [--json=path]

Grid lines are N,prec,iters picked for ~20-100 ms per arm-call. The bench's per-call
buffer copy compresses ratios toward 1 at large N. Both binaries must offer
--size=N --iters=M --prec=f32|f64 (built with ADM_BUILD_BENCHMARKS=ON).
"""
import argparse, json, re, subprocess, sys
import math

GRID_CI = [
    (64, "f64", 400000), (64, "f32", 400000),
    (1024, "f64", 100000), (1024, "f32", 100000),
    (8192, "f64", 20000), (65536, "f64", 3000),
    (1260, "f64", 100000), (4970, "f32", 8000), (21538, "f32", 8000),
    (194000, "f64", 600), (1048576, "f64", 40), (4194304, "f32", 15),
]
GRID_FULL = GRID_CI + [
    (16, "f64", 400000), (128, "f32", 200000), (4096, "f64", 50000),
    (100000, "f64", 1000), (100000, "f32", 2000), (122220, "f64", 600),
    (113876, "f64", 600), (524287, "f64", 600), (1048575, "f64", 400),
    (1048575, "f32", 300), (2097152, "f64", 30), (2097152, "f32", 30),
    (4194304, "f64", 15), (8388608, "f64", 15),
]

def run_cell(binary, n, prec, iters):
    out = subprocess.run(
        [binary, f"--size={n}", f"--iters={iters}", f"--prec={prec}"],
        check=True, capture_output=True, text=True).stdout
    return float(re.search(r"per_call_us=([\d.]+)", out).group(1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin_a")
    ap.add_argument("bin_b")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--grid", choices=["ci", "full"], default="ci")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cells = GRID_CI if args.grid == "ci" else GRID_FULL
    acc = {(n, p): {"A": [], "B": []} for n, p, _ in cells}
    for r in range(args.rounds):
        order = [("A", args.bin_a), ("B", args.bin_b)] if r % 2 == 0 else [("B", args.bin_b), ("A", args.bin_a)]
        for arm, binary in order:
            for n, p, iters in cells:
                acc[n, p][arm].append(run_cell(binary, n, p, iters))

    rows, results = [], {}
    print(f"{'N':>9} {'prec':>4} {'min_A_us':>10} {'min_B_us':>10} {'A/B':>7} {'sprA%':>6} {'sprB%':>6}")
    for (n, p), d in acc.items():
        a, b = min(d["A"]), min(d["B"])
        sa, sb = (max(d["A"]) / a - 1) * 100, (max(d["B"]) / b - 1) * 100
        results[f"{n},{p}"] = {"A": a, "B": b, "ratio": a / b, "spreadA": sa, "spreadB": sb}
        rows.append(f"{n:>9} {p:>4} {a:>10.3f} {b:>10.3f} {a/b:>7.3f} {sa:>6.1f} {sb:>6.1f}")
    print("\n".join(rows))
    geo = math.exp(sum(math.log(v["ratio"]) for v in results.values()) / len(results))
    print(f"geomean A/B = {geo:.4f} over {len(results)} cells")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"geomean": geo, "cells": results}, f, indent=1)
    return 0

if __name__ == "__main__":
    sys.exit(main())
