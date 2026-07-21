# WS8 P0 — on-box DAG gap audit (2026-07-09, HEAD beaca52, w5-3435X ccmlin075)

Method: per-transform counts via perf-stat delta — run each driver at I and 2I
iterations, subtract, divide by I; planning/startup cancels exactly, counts are
retired-instruction counts (contention-immune). FFTW planned FFTW_MEASURE with
wisdom reuse (`scratch_ws8_fftw_drv.c`); yafft production `fft::plan` OOP
forward (`scratch_ws8_yafft_drv.cpp`), per-ISA drivers linked against each
build's libfft_codelets.a with the bench's exact flags. Both sides use the
same allocation style as the production bench (std::vector for yafft,
fftw_malloc for FFTW). Files: census.csv, census_report.txt, fftw_plans.txt,
fftw_codelet_mix.txt, routes.txt, dp_chains.txt, prof_*/.

## Headline verdicts

### 1. The "~25–30% multiply deficit" premise is REFUTED at 256–4096

flops/point (FMA counted as 2, exact from fp_arith_inst_retired.*):
yafft is within ±6% of FFTW at every 256–4096 cell, both precisions — often
LOWER (e.g. 4096 f32 v2/v3 0.94x, v4 2048 f64 0.96x). The DAG multiply economy
of genfft is fully matched by the existing twiddle-elision + fused-emit
butterflies. Per the plan gate: **P3 split-radix is dead on arrival for the
pow2 band** — the deficit it would remove does not exist.

Only 60/120 have a real FLOP deficit: 1.05–1.38x (worst: 120 v4 both prec
~1.37 — the 15-8/8-15 good_thomas-flavored chains do more arithmetic than
FFTW's t2fv_10 × n2fv_12 DIT).

### 2. FFTW's chosen plans are NOT split-radix at decomposition level

MEASURE plans on this box (fftw_plans.txt): 1–2 twiddled radix-8/16/32 levels
over one big straight-line leaf (n1fv/n2fv_32/64), e.g. 4096 f64 =
t2fv_8 → t2fv_8 → n1fv_64; 2048 f64 = t1fv_32 → n2fv_64. The twiddle economy
is "few twiddled levels + huge untwiddled leaf", not conjugate-pair recursion.
(Leaf codelets internally use genfft's DAG, but the plan-level lever FFTW
actually uses here is leaf size, which yafft's r16/r32 wide passes already
approximate at v4.) Also: this FFTW build has NO AVX-512 codelet set
(sse2/avx/avx2/avx2_128 only) — FFTW's yardstick is 256-bit even vs v4.

### 3. Where the cycles actually go (census cyc/pt + topdown, 4096 f64)

| side | instr/pt | cyc/pt | IPC | topdown |
|---|---|---|---|---|
| fftw (MEASURE) | 22.1 | 6.9 | 3.2 | 55% retiring, 42% BE-bound (28% core, 14% mem) |
| yafft v4 | 13.9 | 8.5 | 1.6 | 29% retiring, 69% BE-bound (17% core, **52% mem**) |
| yafft v2 | 68.9 | 16.2 | 4.3 | instruction-volume-bound |

v4 memory-bound split at 4096 f64: **store_bound 25.4%** (fftw 7.3%),
l1_bound 9.7% (1.2%), l2_bound 12.7% (6.7%). yafft v4 executes FEWER
instructions than FFTW and stalls instead — the strided scatter-stores of the
iterative DIF passes are the bottleneck, not the DAG and not instruction count.

v2 loses on sheer instruction volume: 3.1–3.9x FFTW's instr/pt, of which 2x is
the SSE-vs-AVX width handicap (fixed-CPUID yardstick), leaving ~1.5–1.9x
genuine overhead (loads/stores/shuffles/loop) at IPC already 4.3 — near the
retire ceiling, so shaving instructions is the only lever there.

v3 mid pow2: 1.4–2.1x instr/pt overhead at IPC ~2.3–3 — mixed volume+stall.

Tiny N (60/120): FLOP deficit 1.1–1.4x PLUS 2–5.5x instruction overhead
(worst: 120 f32 v4 55 instr/pt vs fftw 10 — structure/dispatch, confirming the
lane-waste diagnosis; note v4 60 f32 has LOW instr/pt 8.9 via good_thomas yet
still ~1.4x cyc — small-N losses are per-pass overhead, not arithmetic).

### 4. MEASURE-FFTW is a much harder yardstick than the ledger's

The rt3 table's FFTW columns (--compare, FFTW_ESTIMATE, cross-process) said
v2 loses 1.2–1.4; against MEASURE-planned FFTW in-process the same cells are
2.3–3.6x, and v4 (which "wins" in the rt3 table) loses 1.2–2.0x at 256–4096.
The campaign target "beat FFTW" needs an explicit yardstick decision:
receipt-grade comparisons must be --fftw-ab with FFT_BENCH_FFTW_MEASURE=1
(standing discipline) — the rt3 FFTW columns understate FFTW badly.

### 5. Hot-symbol attribution (perf record, cycles:u, per archetype)

- v2 4096 f64: 65% mid radix-4 passes + 14% first + 12% last + **7.6%
  __memmove** — an extra full-array copy in the workspace execute path that
  FFTW does not pay.
- v3 4096 f32: 45% mid r8 passes, 24% first, 19% last, **10.5% __memmove**.
- v4 120 f32 (worst tiny, 2.48): **64.8% in the merged radix-15 LAST pass**,
  30.5% first r8 pass. The [8,15] chain's r15 last pass is the whole loss —
  lane waste + its ~160 priced spills at a size where FFTW runs t2fv_10 ×
  n2fv_12 at 10 instr/pt.
- v3 60 f32 (1.91): chain 3-2-2-5 = FOUR passes for N=60; 38% in the r2
  passes alone, 12% un-inlined dispatch (iterative_dif_execute_ws). FFTW does
  it in one twiddled level + leaf. Pass-count/structure, not arithmetic.

Cross-check vs receipt-grade tool: FFT_BENCH_FFTW_MEASURE=1 --fftw-ab
--sizes=4096 --prec=f64 on v4 → FFTWAB fwd=1.139 (spread 20.9%), census cyc
ratio 1.23 — consistent; the census methodology is validated by the
role-swapped tool.

## Route/chain receipts (routes.txt, dp_chains.txt)

All loss cells route iterative_dif except: v3 f32 256/512 → four_step_batched
(MISMATCH vs model; the model itself prefers iterative_dif) and v4 60 →
good_thomas (MISMATCH). Plan-doc corrections: 60 is NOT in the codelet catalog
at HEAD (codelet cost 501 > dif 337 at every ISA); 120 routes iterative_dif
everywhere.

DP chains at the loss cells: v2 4096 f64 = 4-4-4-4-4-4 (6 twiddled sweeps vs
FFTW's 2), v3 adds a leading r2; v4 = 16-16-16 / 16-16 / 32-32 (2–3 sweeps —
already FFTW-like level counts).

## Verdict-triage finalization (fairness rule 2)

| Prior verdict | Status after P0 |
|---|---|
| genfft fused-twiddle t1_8 tie (155H) | CONFIRMED here by flop parity — twiddle fusion is not the gap |
| Late-load scheduling 17% worse (155H) | Carry over; nothing in P0 contradicts |
| radix-16 spills at 16 regs | Register arithmetic — binds |
| ws7-p5-survey "split-radix demoted under SIMD" (paper) | Superseded by stronger result: multiply parity means split-radix has no deficit to remove at 256–4096; P3 gate already failed |
| beat-fftw-audit-2026-07 band attribution (155H, pre-WS4–7) | REPLACED by this audit: v2=volume, v3=volume+stall, v4=store-bound, tiny=structure+flops |

## What this means for the campaign (P1/P4 refocus; P3 closed at pow2)

- P3 conjugate-pair split-radix: CLOSED for 256–4096 by the census gate
  (no multiply deficit to remove; the bottlenecks are stores and volume).
  60/120's 1.1–1.4x flop gap is a tiny-N DAG matter → P4's straight-line
  codelet track, not a pass-family rewrite.
- New P1 targets, evidence-backed:
  (a) __memmove at 7.6–10.5% of cycles in the v2/v3 OOP execute path — find
      and kill the extra full-array copy (pass-parity/workspace artifact).
      Cheapest confirmed lever in the audit.
  (b) v4 mid-pow2 store-bound: pass staging/write pattern (store_bound 25%,
      l1 9.7% — scatter stores of DIF sweeps). Restage/write-through-tile
      ideas, admission via the existing predicates.
  (c) v2/v3 instruction volume: shave loads/stores/shuffles in the 16-reg
      pass bodies (IPC 4.3 = retire-limited; every shaved instruction is
      wall-clock).
  (d) 60 f32 v3 chain 3-2-2-5: four passes for N=60 — chain-length cost at
      tiny N looks underpriced in the model (r2 passes 38% of cycles).
- P4 tiny-N: both structure (dispatch/lane waste) AND a modest flop deficit —
  a straight-line 60/120 DAG at the right width addresses both.
- Yardstick: adopt MEASURE-FFTW via --fftw-ab as the only receipt-grade FFTW
  comparison for WS8 (matches standing discipline; rt3 FFTW columns are
  survey-grade only).

## Post-P1(a) addendum (HEAD 8c0e1ec, 2026-07-09)

Census re-run after the copy-free OOP commits (8c747cf, 8c0e1ec):
census_post_p1a.csv / census_report_post_p1a.txt, same methodology
(perf-stat delta at I and 2I, taskset -c 0).

Gap deltas (cyc/pt vs MEASURE-FFTW, pre -> post):
- 60  f32 v4: 1.39 -> 1.05  CLOSED by P1(a); dropped from P4 targets
- 120 f32 v4: 4.29 -> 4.59  survives, worst cell (instr 5.4x) -> P4 #1
- 120 f64 v4: 1.99 -> 1.75  survives -> P4
- 60  f64 v4:  n/a -> 1.94  survives (good_thomas route) -> P4
- 4096 f64 v4: 1.23 -> 1.05 nearly closed
- v4 f64 mid-pow2 narrowed (256/512 -> 1.15); f32 2048 1.49 / 4096 1.39
  and f64 1024 1.40 remain -> P1(b)
- v2/v3 unchanged (1.8-3.1x retire-limited) -> P1(c)/P1(d)
