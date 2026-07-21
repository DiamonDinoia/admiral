# WS7 P0 loss ledger — 2026-07-08, HEAD 4219bc7, w5-3435X

Sources: bench-results/ratio_table.txt (1D, fresh), bench-results/ws7_p0_nd_matrix.txt
(ND, fresh mt binary built 22:41 tonight — the 2026-07-05 stale receipts are dead).
Ratios = yafft/ref, <1 wins. ND at NT>1 is wall-clock (serial-anchor caveat: flag,
don't gate; role-swapped tools arbitrate). MT rt ratios historically bogus — fwd is
the MT signal.

## vs ducc0 — 1D single-thread (5 losses + 1 marginal, 114/120 fwd wins)

| N     | prec | ISA | fwd  | rt   | phase |
|-------|------|-----|------|------|-------|
| 15120 | f64  | v4  | 1.70 | 1.51 | P4 (good_thomas gather / route A/B) |
| 15120 | f64  | v3  | 1.43 | 1.27 | P4 |
| 16384 | f64  | v4  | 1.42 | 1.05 | P3/P4 (r16 spill chiplet; terminal via P1 tool) |
| 32768 | f64  | v3  | 1.18 | 1.12 | P2 (v3 cost-cell refit) |
| 8192  | f64  | v3  | 1.10 | 1.00 | P2 |
| 15120 | f32  | v3  | 1.08 | 0.84 | P2/P4 |

## vs ducc0 — ND matrix (fresh; the stale 1T 2D f64 1.38–1.63 losses largely evaporated)

1T (cycle-true):
| shape | prec | fwd | rt | note |
|-------|------|-----|----|------|
| 2048x2048 | f64 | **1.101** | 0.936 | only real 1T ND loss left (32MB/side band) |
| 512x512   | f64 | **1.029** | 0.464 | marginal, err 3.2% — re-arbitrate before work |

All other 1T cells win: 1024² f64 0.71, 256³ f64 0.70, 128³ f64 0.74, 1024x1024x64
f64 0.62; f32 2D/3D 0.51–0.90.

MT — the structural residual is **512² at intermediate/high threads** (yafft
non-monotone in NT while ducc0 keeps scaling):

| shape | prec | NT | fwd | rt | yafft fwd us trend |
|-------|------|----|-----|----|--------------------|
| 512x512 | f64 | 4  | **1.356** | 0.61 | 1T 2237 → 2T 780 → 4T 989 → 8T 976 → 16T 144 |
| 512x512 | f64 | 8  | **1.868** | 0.65 | (16T=144us looks bimodal-fast; treat with 16384-style suspicion) |
| 512x512 | f32 | 8  | **1.775** | 1.59 | 1T 657 → 2T 388 → 4T 421 → 8T 575 → 16T 737 |
| 512x512 | f32 | 16 | **2.637** | 2.48 | f32 512² gets SLOWER with threads beyond 2T |

Every other MT cell wins, mostly 0.1–0.6. Diagnosis hypothesis (P6): no thread-count
cap for small working sets (512² = 2–4MB); ducc0 caps/limits parallelism where the
work is too small, yafft pays full spin-pool fan-out + memory contention.

## vs FFTW

ND: yafft beats FFTW in every cell except 16T f32 512² (1.678) — same P6 residual.
1D (from ratio_table FFTW columns, unchanged this pass): v2 small/mid N 256–2048
both prec (1.1–1.5), v3 f32 2048/4096 (1.30/1.36), f64 1024/2048 (~1.18), 60/120
f32, 15120. Named reason: FFTW genfft split-radix codelets — P5 track (a).

## P0 verdicts feeding the plan

1. ND driver work (old P6 "1T 2D f64 driver audit") shrinks to: 2048² f64 1T fwd
   1.10 + 512² f64 1T marginal. The big fish is the **512² MT scaling defect**.
2. P2 (v3 refit) unchanged: 8192/32768 f64 v3 + 15120 f32 v3 confirmed fresh.
3. P3/P4 (16384/15120 f64) unchanged, biggest 1D losses.
4. 16T f64 512² fwd 144us is 6.8x faster than 8T — bimodality; any 512² receipts
   need repeated runs before gating (yafft-16384-bimodal discipline applies to ND).

## P2 addendum (2026-07-09, ws7_p2_v3_receipts.txt + ws7_p2_v3_terminals.txt)

v3 refit REFUTED by receipts — no cost-cell change warranted:
- Routes: 8192/32768 f64 + 15120 f32 all already route iterative_dif, tie forced-DP.
- Chains: DP picks beat or tie every candidate (32768 DP beats all-8 robustly 0.78;
  15120 f32 DP beats all four candidates robustly, best challenger 0.88).
- Terminals (+64/+32) at v3 f64: ties/losses — the 16384-W8 lever doesn't transfer.
- 15120 f32 v3: forced-dif A/ducc = 0.69 in-process (twice) — the table's 1.08 is
  cross-process artifact; NOT a real loss.
- 8192/32768 f64 v3: chain-, route-, and terminal-insensitive; A/ducc anchors swing
  0.85–1.33 across runs. Genuine residual (if any) is kernel-level (r8 W=4 spill
  cell 6.10) — v3 has no 16/32 radices to restage, so P3's register-budget work
  is the only remaining lever family.

## P3 addendum (2026-07-09, commit 9dcdda5)

Two-sweep restage SHIPPED for register-ceiling DIF butterflies (r16/r32 at 32
regs, r8 at 16 regs). Production receipts: native 16384 f64 -10.3% (now ~0.87
vs ducc0 — the 1.42 headline loss is CLOSED), 8192/32768/65536 f64 -3~4%; v3
f32 4096 -14.7%, f64 16384 -6.9%, 65536 -4.0%. Post-fix re-audits
(ws7_p3_postfix_audits.txt): 4096 all-8 override and 16384 [16,16]+64 terminal
both still robustly justified — kept. No ranking flips → no cost-cell refit.
Remaining 1D ledger: 15120 f64 v4/v3 (P4), 8192/32768 f64 v3 residuals if any
(kernel-level, partially addressed by this restage — re-measure at next table
regen).

## P4 addendum (2026-07-09)

15120 f64 status after restage: fwd 1.29–1.38 (was 1.70), rt now WINS 0.76–0.93.
Receipts (ws7_p4_15120_*.txt):
- Route: default is iterative_dif; forced-dif ties (RABDIF 1.022). good_thomas is
  a small-N catalog route (N<=60, factor set {2,3,4,5,8}) — structurally
  unavailable at 15120; the "GT gather port-5 fix" lever is moot for this size.
- Chains: CHAIN-INSENSITIVE. First battery showed apparent robust wins
  ([2,3,3,3,5,7,8] 1.34, [4,3,3,3,5,7,4] 1.28) but every rematch in a fresh
  process ties — 15120 has the same bimodal cross-process swing as 16384
  (anchors 0.84–1.79 across runs). Only repeated role-swapped ties are real.
- Kernel decomposition: BASECOST 304k cyc == sum of isolated pass costs
  (20.1 cyc/elem for [5,3,3,3,16,7]) — pure compute, no cache effect.
  Normalized cyc/elem/log2(r): r16 0.90, r8 1.14, r7 1.38, r4 1.61, r5 1.62,
  r3 1.87. The three r3 passes are 44% of total cost.
- Radix-9 chains rejected by the harness (not in the radix set; generic odd
  path fails verify at 9).

VERDICT: 15120 f64 residual (~1.33 fwd) is odd-radix kernel compute quality,
not route/chain — folds into P5 (from-scratch kernels: radix-9/merged-3·3,
better odd-radix butterflies). P4 closed.

## P5 addendum (2026-07-09): merged radix-9/15 passes

The P4 verdict's lever, executed. radix_sym_dft is IP-generic for composite odd
IP; admitting 9 (=3·3) and 15 (=3·5) as single merged passes into the wide
(32-reg) radix set + DP cost tables halves the odd-radix sweep count. (The old
"fails verify at 9" note predates dispatch integration — VERIFY now PASSes all
sizes, both precisions.) Isolated cells: r9 4.0 cyc/elem (≈1.26/log2, vs 2×r3
1.87) −34%; r15 5.2 (−16% vs r3+r5). Receipts ws7_p5_r9r15_cells.txt.

In-chain role-swapped receipts (ws7_p5_r9r15_chain_ab{,2}.txt):
- 15120 f64 [7,9,16,15]: 0.78 vs old chain, then 0.90 vs battery-1 winner —
  cumulative ~0.70x; A/ducc now ~0.93–1.03 (was 1.33). Headline loss closed
  within serial swing.
- 15120 f32 0.73, 6561 f64/f32 ~0.5 ([9,9,9,9]), 360 f64 0.88.
- L1-resident sizes exposed the isolated-vs-in-chain trap again: r9.last
  (f64 2.35→4.80, f32 1.30→2.35) and f32 r15.vec (1.90→2.72) re-fitted
  in-chain; feasible regions in the twiddles.hpp comments.
- Known model ceiling: 720 f64 keeps [9,5,16] (1.06 vs old, still 0.70 vs
  ducc0) — flipping it needs r9.vec>5.80 which kills the 360 win exactly.

v3 (16-reg) admission REFUTED by probe (ws7_p5_v3_r9r15_cells.txt): r9 vec
6.25 vs 2xr3 4.55 (+37%), r15 6.33 vs r3+r5 6.50 (tie, below gate). Structural:
sym-DFT pair sums/diffs alone are 18 live batches at IP=9 — exceeds the 16-reg
file however the sweeps are sliced, so no restage variant can fix it. 15120
f64 v3 (1.43) keeps that named reason; its remaining lever is small-radix
quality at 16 regs (P5 split-radix track (a)). Odd-restage at v4 also closed
without prototyping: the sym-DFT is FMA-port-bound, spills ride idle store
ports (butterfly.hpp header, simdref+llvm-mca receipts) — r15's ~160 spills
are priced and paid where it still wins in-chain.

## P6 addendum (2026-07-09): ND/MT closure

Truth pass at HEAD (bench-results/mt_scaling.txt, then ws7_p6_mt_scaling_fixed
.txt): the only REAL defect was MT col-tile granularity at small shapes. Fixed
by a one-line parallel balance cap in nd_col_block (>= ~4 tiles per worker;
binds only when inner/(4*nt) < the byte-budget tile count). Receipts
ws7_p6_colbt_{ab,confirm}.txt: 512x512 4T f64 1.32->0.76, 8T f64 1.55->0.53,
8T f32 1.51->0.51, 16T f32 2.23->0.27 vs ducc0; large shapes and 1T untouched
(2048x2048 16T 0.13/0.12 unchanged). Post-fix full matrix (ws7_p6_mt_scaling_
fixed.txt): ALL 50 cells win, worst 0.98 (2048x2048 f64 1T).

The apparent 1T 2D f64 losses (1.38-1.62 in both the stale P0 rows and the
first fresh sweep) were MEASUREMENT POLLUTION: the first sweep's 512x512 f64
1T fwd was 3989us where the yafft-only perf driver measures 1228us; fresh
repeats give 0.76-0.94 vs ducc0 with no 1T-relevant code change. 1T ND was
never losing at HEAD.

Refuted along the way (all at 512x512 f64 1T, scratch_perf_driver receipts,
probes reverted): smaller col radices (r8 +26%, r4 +60% — pass count rules);
ducc-style copy-through-contiguous tiles (+35%); per-arm pointer hoisting
(tie — the lea/mov storm in the profile was sample skid on hoisted-pointer
reloads, the pass is L3-latency-bound at IPC 0.92); per-arm SW prefetch (tie);
fat tiles at 1T beyond Bt=80 (worse). The col driver is locally optimal on
every knob short of a four-step/lane-packed redesign (P5 track (d), not
currently needed — the matrix is green).

## FFTW standing post-WS7 (2026-07-09, rt3 receipts)

vs ducc0 the campaign is green (117/120 1D, 50/50 ND/MT). vs FFTW, 1D fwd
losses remaining (>1.03): dominated by v2/SSE small-mid pow2 256-4096
(1.2-1.4 both prec) and tiny smooth 60/120 (worst: 120 f32 v4 2.48, 60 f32
v3 1.91 — lane waste at W>=8 against N<=120 persists even after the shipped
width-adaptive last pass a3440c8). v3 mid pow2 1.1-1.2. ND: yafft beats FFTW
in every cell (ws7_p6 receipts).

Named reason, all cells: FFTW genfft split-radix/conjugate-pair codelets —
~25% fewer real multiplies than classical radices in the compute-bound band,
which is exactly where these sizes live. Open idea: P5 track (a) split-radix
chiplet (also the only remaining lever for 15120 f64 v3 1.43 vs ducc0 — the
16-reg register file excludes merged odd radices structurally). This is the
sole open perf item leaving WS7.

## WS8 P0 addendum (2026-07-09): FFTW gap audit — premise corrected

Full receipts: bench-results/ws8_p0_dag_audit/ (census.csv, census_report.txt,
fftw_plans.txt, routes.txt, dp_chains.txt, README.md).

The standing named reason "FFTW genfft split-radix codelets — ~25% fewer real
multiplies" is REFUTED at 256–4096: per-transform FLOP census (perf-delta,
fp_arith_inst_retired.*) shows yafft within ±6% of MEASURE-planned FFTW at
every pow2 loss cell, often lower. FFTW's own MEASURE plans here are not
split-radix at plan level — 1–2 twiddled radix-8/16/32 levels over big
straight-line leaves (n1fv/n2fv_32/64). P3 split-radix track: CLOSED for the
pow2 band by the plan's own stop-gate. This FFTW build has no AVX-512 codelets.

Replacement named reasons (measured):
- v2 256–4096: instruction volume — 3.1–3.9x FFTW instr/pt at IPC 4.3
  (retire-limited); 2x is the SSE-vs-AVX yardstick handicap, ~1.5–1.9x is real
  overhead. Plus 7.6% cycles in __memmove (extra OOP-path copy).
- v3 mid pow2: 1.4–2.1x instr/pt + 10.5% __memmove.
- v4 vs MEASURE-FFTW: store-bound — topdown 52% memory-bound (store 25.4%,
  L1 9.7%, L2 12.7%) vs FFTW 14%; fewer instructions than FFTW but IPC 1.6.
  Receipt-grade FFTWAB 4096 f64 fwd=1.139.
- 60/120: real FLOP deficit 1.05–1.38x AND structure: 120 f32 v4 spends 64.8%
  in the merged r15 last pass; 60 f32 v3 runs a 4-pass 3-2-2-5 chain (38% in
  r2 passes). 60 is NOT in the codelet catalog at HEAD (cost 501 vs dif 337);
  v4 60 routes good_thomas, v3 f32 256/512 route four_step_batched (MISMATCH
  flags stand).
- Yardstick note: rt3 FFTW columns (ESTIMATE, cross-process) understate FFTW;
  vs MEASURE in-process the v2 cells are 2.3–3.6x, and v4 loses 1.2–2.0x at
  256–4096. Receipt grade for WS8 = FFT_BENCH_FFTW_MEASURE=1 --fftw-ab only.

## WS8 close-out (2026-07-10): P1-P4 shipped, acceptance receipts at HEAD 290009b

Shipped this workstream: copy-free OOP + good_thomas aliasing split (P1a,
8c747cf/8c0e1ec), tiny-N DIT codelets + BASECOST admission (P4 band),
row-split last pass + col-major L1-resident passes on 16-reg ISAs (P1b/P1c,
f46d0a7/938fe6f), 120 f64 W=8 measured order (4c08221), middle-pass fusion
at 4096 (16-reg) + f64-only at 2048 (2dc380f/6f47781), dif_pass_last tail
elimination: measured 3-way remainder (chiplet / overlap / outlined scalar)
+ constant-mask prefix stores via xsimd 633553e pin (730d57b), BASECOST
epoch refresh (290009b).

Acceptance receipts (FFT_BENCH_FFTW_MEASURE=1 --fftw-ab, 9 rounds, at HEAD):
bench-results/ws8_p6_fftw_ab_{v2,v3,v4}.txt; rt3 tables rt_{v2,v3,v4}_t{1,2}.txt.

Standing vs MEASURE-FFTW (fwd ratios; rt generally better):
- v4: outright wins 360 f64 0.79, 2520 f64 0.92, 720 f32 0.62; within 20%
  on 9 more cells incl 4096 f64 1.07 (was 1.2-2.0x band at P0) and 512 f64
  1.12. Worst v4 cell now 120 f32 1.74 (was 2.48 at P0).
- v3: best cells 360 f32 1.08 (rt 0.94), 720 f32 1.08 (rt 0.96), 360 f64
  1.14; the mid-pow2 band sits 1.4-1.7 (instruction volume, 16-reg).
- v2: 1.5-2.5 across the board — the SSE-vs-AVX yardstick handicap dominates
  (FFTW's build uses AVX2 codelets regardless of our -march=v2 binary);
  ducc0 remains the fair per-ISA opponent for v2 (rt3 tables).
- 60/120 remain the largest structural losses on all ISAs (real FLOP deficit
  + chain structure, see WS8 P0 addendum). Open lever: genfft-style
  straight-line leaves beyond the current catalog radices.
