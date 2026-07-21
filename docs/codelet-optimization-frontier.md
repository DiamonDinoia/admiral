# Codelet optimization campaign (sizes 2–64) — measured frontier & do-not-retry ledger

Date: 2026-06 (Intel Core Ultra 7 155H, AVX2/16 ymm, pinned `taskset -c 0` P-core).

Method for every claim below: pair-build baseline vs candidate to side paths, then
**per-size-interleaved** paired `--compare` (`taskset -c 0`, cycle metric), ≥12–15 reps,
median of per-rep cand/base ratios (cancels thermal drift). Accuracy gate (`--verify`,
full catalog + decomposition) passed on every shipped change. Controls (unaffected sizes)
held within ±1.2% → that is the noise floor; ship gate = Δ ≥ +1% with no >1% regression.

> ⚠️ Methodology note that bit us: an all-baseline-then-all-candidate sweep (sizes walked
> once per binary per rep) injects a **size-position thermal bias** — unaffected sizes
> showed spurious ±6% swings. Always interleave base/cand **per size**, back-to-back.

## Shipped wins

### WS1 — f32 r=2 Rader-prime cofactor → scalar `rader_apply` (full ymm)
`cofactor_simd_profitable<T,R>` (`codelet.hpp`): the f32 R=2 Rader-prime batch uses a
4-lane SSE register with **2 idle lanes**. Routing those to the scalar `rader_apply<M>`
recursion (whose inner `kernel<M-1>` conv runs at full ymm width) is faster. Now gated by
the same low-half-mask guard (`2*R <= Wc → false`) the odd composites already use.

Measured (f32 fwd, cand/base): N=26 +20.5%, 34 +24.3%, 46 +12.8%, 58 +44.2%, 62 +18.4%;
nested via `kernel<46>`/`kernel<58>`: N=47 +12.3%, 59 +41.6%. f64 unaffected by construction.
**Exception N=38 (M=19): −8.4%** — shipped anyway (chosen tradeoff; still beats ducc0 0.91×;
no clean structural predictor separates M=19 from the winners {13,17,23,29,31}).

### WS2 — N=54 f32: precision-aware radix (the catalog's only loss, fixed)
`codelet_radix_for<T>` (`ct_math.hpp`), used at the `kernel<N,T>` leaf. N=54 f32 peels
**r=6** (M=9 odd → cofactor-SIMD over 6/8 ymm lanes): 1.11× (LOSE) → **0.51× (WIN)**,
+51%. f64 keeps r=2 (native batch 4 < 6, so r=6 would fall to scalar and regress −55%);
f64 unchanged (+0.1%). This exactly meets the plan's "flip f32 without moving f64" gate —
the optimum is precision-dependent, which a precision-agnostic split could not capture.

### WS4 — N=32: radix-8 leaf (despite spills)
`codelet_radix(32)=8` (M=4): N=32 f32 0.68→0.40 (+41%), f64 0.79→0.49 (+35%). **Ships a
kernel that emits ~2× instructions and ~2× stack spills** (91 vs 48 rsp stores; peak_live
26 > 16) — it violates the no-spill register-model heuristic but measures decisively
faster on this wide OoO core (the r=4 path was throughput-suboptimal). Accepted as a
measured exception to the heuristic. **N=64 (M=8) does NOT**: there r=8 is register-starved
and regresses ~25%, so 64 keeps the default r=4.

## Do-not-retry (measured, no win)

### WS3 — Rader gather/pack layout (N=47,53,59)
Buffer-alias (reuse the dead gathered `are/aim` as the inverse-DFT output, saving
2·L·sizeof(T) of frame) measured **perf-neutral**: N=47/53/59 within ±1.4% both precisions,
nothing clears +1%. The staging buffers are already L1-resident, so the frame saving buys
no cycles; the ~5.8% `ld_blocks.store_forward` on N=53 is not frame-pressure-driven. Since
(a) showed no signal, the larger packed-gather refactor (b) was not pursued. The serial
`fwd→mul→inv` chain is the floor and is algorithmic (out of scope). These sizes already
beat ducc0 2–5×. **Rader spill/layout micro-optimization is not worthwhile here.**

### WS5 — small-leaf latency floor (N=5,6,7,8,11)
Scalar single-call kernels (M≤2 / r > batch width), latency-bound on a short FFT DAG with
no extractable ILP. `radix_sym_dft` is already at the half-multiply optimum with U=1 proven
optimal (`butterfly.hpp` dif_pass_unroll note). llvm-mca: these run ≈5 IPC when *iterated*
(batched across many transforms — which already happens when they are sub-transforms of the
cofactor/iterative paths) but only 1.2–1.7 IPC for one isolated transform. **No single-call
change is viable**; the IPC headroom is only realizable by batching across transforms, which
the cofactor/iterative paths already do.

## Abandoned-approach guardrails (unchanged, do not re-litigate)
- per-(N,T) factor *ordering* — `codelet.hpp` (the codelet_radix fixpoint is already best).
- U>1 on radix-5/7 — `butterfly.hpp` dif_pass_unroll (U=1 is the throughput optimum on AVX2).
- `load_complex` AoS↔SoA boundary — perf-neutral; the boundary is not the bottleneck.

## Final catalog scoreboard (post-campaign, shipped build = WS1+WS2+WS4)

`taskset -c 0 fft_benchmark --compare --sizes=2..64 --prec=both` (cycle metric;
ratio = fft/ducc0, **< 1.0 means we win**). **0 / 63 sizes lose** on either precision
(fwd or rt) — N=54 f32 was the last remaining loss and WS2 fixed it. Avg fwd ratio:
**f64 0.452, f32 0.435** (min 0.055, max 0.900). The **Mpts/s** columns are forward-transform
throughput on this core = N / fft_fwd_µs (i.e. 10⁶ points/s; one warm pinned transform, not
ducc0). Rows changed by this campaign are tagged.

| N | f64 fwd | f64 rt | f64 Mpts/s | f32 fwd | f32 rt | f32 Mpts/s | |
|---|--------:|-------:|-----------:|--------:|-------:|-----------:|---|
| 2 | 0.055 | 0.055 | 417 | 0.061 | 0.053 | 377 | |
| 3 | 0.131 | 0.164 | 259 | 0.126 | 0.153 | 278 | |
| 4 | 0.077 | 0.069 | 563 | 0.077 | 0.075 | 563 | |
| 5 | 0.306 | 0.269 | 187 | 0.263 | 0.246 | 222 | |
| 6 | 0.204 | 0.234 | 308 | 0.227 | 0.244 | 278 | |
| 7 | 0.313 | 0.309 | 236 | 0.263 | 0.260 | 289 | |
| 8 | 0.262 | 0.278 | 362 | 0.142 | 0.183 | 667 | |
| 9 | 0.588 | 0.489 | 161 | 0.305 | 0.329 | 292 | |
| 10 | 0.280 | 0.290 | 346 | 0.424 | 0.362 | 238 | |
| 11 | 0.440 | 0.369 | 246 | 0.438 | 0.374 | 242 | |
| 12 | 0.281 | 0.283 | 418 | 0.272 | 0.289 | 441 | |
| 13 | 0.427 | 0.387 | 173 | 0.437 | 0.397 | 174 | |
| 14 | 0.329 | 0.323 | 391 | 0.478 | 0.394 | 275 | |
| 15 | 0.580 | 0.533 | 239 | 0.317 | 0.332 | 459 | |
| 16 | 0.341 | 0.333 | 449 | 0.364 | 0.346 | 422 | |
| 17 | 0.439 | 0.399 | 167 | 0.411 | 0.389 | 177 | |
| 18 | 0.416 | 0.394 | 354 | 0.618 | 0.535 | 256 | |
| 19 | 0.392 | 0.380 | 169 | 0.569 | 0.520 | 127 | |
| 20 | 0.436 | 0.390 | 405 | 0.343 | 0.356 | 529 | |
| 21 | 0.543 | 0.533 | 298 | 0.353 | 0.355 | 481 | |
| 22 | 0.457 | 0.438 | 348 | 0.566 | 0.515 | 283 | |
| 23 | 0.379 | 0.373 | 165 | 0.487 | 0.467 | 136 | |
| 24 | 0.740 | 0.665 | 270 | 0.799 | 0.727 | 260 | |
| 25 | 0.535 | 0.486 | 356 | 0.360 | 0.381 | 353 | |
| 26 | 0.451 | 0.402 | 257 | 0.622 | 0.576 | 179 | WS1 (f32 0.80→0.62) |
| 27 | 0.574 | 0.525 | 364 | 0.565 | 0.473 | 376 | |
| 28 | 0.443 | 0.397 | 447 | 0.371 | 0.345 | 565 | |
| 29 | 0.269 | 0.268 | 208 | 0.261 | 0.250 | 230 | |
| 30 | 0.514 | 0.476 | 399 | 0.641 | 0.600 | 346 | |
| 31 | 0.276 | 0.278 | 189 | 0.369 | 0.353 | 160 | |
| 32 | 0.359 | 0.340 | 658 | 0.388 | 0.339 | 644 | WS4 (r=8: f64 0.79→0.36, f32 0.68→0.39) |
| 33 | 0.584 | 0.542 | 356 | 0.439 | 0.443 | 454 | |
| 34 | 0.383 | 0.375 | 285 | 0.649 | 0.589 | 186 | WS1 (f32 0.87→0.65) |
| 35 | 0.688 | 0.629 | 321 | 0.346 | 0.374 | 453 | |
| 36 | 0.500 | 0.451 | 477 | 0.360 | 0.360 | 681 | |
| 37 | 0.222 | 0.219 | 221 | 0.209 | 0.200 | 257 | |
| 38 | 0.421 | 0.428 | 248 | 0.900 | 0.796 | 128 | WS1 (f32 0.86→0.90, accepted regression) |
| 39 | 0.459 | 0.450 | 286 | 0.384 | 0.375 | 356 | |
| 40 | 0.743 | 0.732 | 329 | 0.640 | 0.636 | 391 | |
| 41 | 0.300 | 0.288 | 153 | 0.316 | 0.297 | 167 | |
| 42 | 0.577 | 0.557 | 409 | 0.677 | 0.630 | 370 | |
| 43 | 0.231 | 0.231 | 192 | 0.288 | 0.283 | 167 | |
| 44 | 0.519 | 0.484 | 446 | 0.333 | 0.345 | 650 | |
| 45 | 0.665 | 0.627 | 403 | 0.544 | 0.549 | 502 | |
| 46 | 0.475 | 0.477 | 207 | 0.706 | 0.668 | 147 | WS1 (f32 0.84→0.71) |
| 47 | 0.403 | 0.437 | 101 | 0.659 | 0.649 | 69 | WS1 nested (f32 0.78→0.66) |
| 48 | 0.799 | 0.775 | 330 | 0.838 | 0.775 | 348 | |
| 49 | 0.664 | 0.639 | 351 | 0.455 | 0.500 | 459 | |
| 50 | 0.726 | 0.696 | 358 | 0.440 | 0.590 | 383 | |
| 51 | 0.422 | 0.431 | 303 | 0.356 | 0.356 | 386 | |
| 52 | 0.465 | 0.473 | 380 | 0.389 | 0.359 | 483 | |
| 53 | 0.227 | 0.221 | 169 | 0.197 | 0.198 | 213 | |
| 54 | 0.710 | 0.686 | 419 | 0.516 | 0.449 | 620 | WS2 (f32 1.11 LOSE→0.52, the last loss) |
| 55 | 0.734 | 0.727 | 315 | 0.307 | 0.348 | 618 | |
| 56 | 0.748 | 0.772 | 332 | 0.723 | 0.696 | 367 | |
| 57 | 0.451 | 0.462 | 268 | 0.390 | 0.404 | 332 | |
| 58 | 0.398 | 0.400 | 233 | 0.406 | 0.400 | 238 | WS1 (f32 0.76→0.41) |
| 59 | 0.293 | 0.303 | 110 | 0.350 | 0.341 | 110 | WS1 nested (f32 0.62→0.35) |
| 60 | 0.887 | 0.886 | 326 | 0.680 | 0.725 | 456 | |
| 61 | 0.214 | 0.214 | 151 | 0.235 | 0.233 | 155 | |
| 62 | 0.379 | 0.377 | 231 | 0.564 | 0.544 | 166 | WS1 (f32 0.72→0.56) |
| 63 | 0.628 | 0.624 | 409 | 0.565 | 0.565 | 512 | |
| 64 | 0.696 | 0.675 | 411 | 0.629 | 0.601 | 502 | (r=8 tried, −25%; kept r=4) |

## Spill-audit + measured-throughput campaign (2026-06-24)

Goal (user directive): generate optimal asm per compiled codelet, measure each
normalized by data size, and derive the optimal decomposition for any 1D N.

**Outcome:** the existing routing is already at/near the model-optimal decomposition
for every N. A measured per-codelet recalibration of the cost model was built and
A/B-tested — it **regressed** both precisions and was **reverted**. Shipped: the
two analysis tools (`--codelet-sweep`, `--decomp-report`) + this report. No routing
or codelet code changed; `kernel_should_noinline` left as-is.

### Phase 0 — ASM spill-map (AVX2, 16 YMM; `-fno-lto -O3 -march=native -ffast-math`)

Per-N vector-insn count, stack-spill movs (`vmov*` to/from `(%rsp|%rbp)`), and
measured cyc per butterfly (cyc/blf = cyc_per_call ÷ N). Both precisions; selected
rows (full 2–64 table in scratchpad `spill_table.md`):

| N | f64 vec | f64 spill | f64 cyc/blf | f32 vec | f32 spill | f32 cyc/blf |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 13 | 0 | 9.75 | 10 | 0 | 9.44 |
| 4 | 44 | 0 | 4.03 | 46 | 0 | 4.00 |
| 8 | 274 | 4 | 6.01 | 258 | 3 | 3.06 |
| 11 | 420 | 52 | 7.38 | 423 | 64 | 7.45 |
| 13 | 611 | 127 | 10.60 | 591 | 118 | 10.43 |
| 16 | 377 | 8 | 3.62 | 380 | 6 | 3.66 |
| 25 | 413 | 21 | 4.05 | 416 | 46 | 4.11 |
| 32 | 813 | 38 | 1.88 | 862 | 58 | 2.49 |
| 47 | 3111 | 991 | 11.89 | 1870 | 299 | 17.24 |
| 49 | 1093 | 93 | 3.25 | 981 | 350 | 2.62 |
| 53 | 2576 | 582 | 6.89 | 2199 | 492 | 5.46 |
| 59 | 2065 | 491 | 9.90 | 2754 | 241 | 10.39 |
| 64 | 641 | 32 | 2.60 | 610 | 24 | 2.08 |

**Findings:**
- **Spilling is radix-driven, not size-driven.** cyc/blf *falls* as N grows: the
  big composite leaves 32/49/64 are the *cheapest* per butterfly (1.9–3.3 cyc),
  the primes 47/53/59 the dearest (7–12 cyc) with the heaviest spill (47 spills
  ~1000 movs). The live set is `2·radix+10`; composite leaves stay
  register-resident via their iterative sub-passes, only *prime* radices (radix==N)
  overflow 16 YMM. **Directly answers the spilling question:** feeding a bigger
  *composite* codelet does NOT push the register file out to L1 — it's the prime
  radix that spills, and those sizes are already routed through Rader/Bluestein,
  never used as four-step leaves.
- **32-register lens (`-march=skylake-avx512`, codegen-only):** the heavy prime
  leaves shed most spills with 32 ZMM, confirming spills (not memory traffic) bind
  *those*. Irrelevant to deployment (155H has no AVX-512) and to routing — they're
  never leaves.
- **Action gate — `kernel_should_noinline` UNCHANGED.** The `2·radix+10 > 16` gate
  fires only at N=64 on AVX2, and no other catalog size both spills and is used as
  a leaf while the gate misses it. Re-dumping `.s` after a no-op confirmed nothing
  to change.

### Phase 1 — `--codelet-sweep` (shipped)

`benchmark/bench_fft.cpp --codelet-sweep [--prec][--reps][--inner][--no-ducc]`:
times `codelet_dispatch<T>` per catalog size 2–64, accuracy-gated, emits
cyc/call plus the **size-normalized** cyc/N and cyc/(N·log₂N). Self-contained
(no cost-model dependency). This is the per-codelet normalized-throughput
measurement the directive asked for.

### Phase 2 — recalibration TESTED & REJECTED; `--decomp-report` shipped

- Regenerated `codelet_cost_cyc` from measured f64 data and built a precision-aware
  variant (added an f32 table + a `kCostScale` cross-family re-anchor + templated
  cost model). The `--decomp-report` analysis showed HEAD already matches the
  model-optimal route for every N except 7 benign f32 pow2/smooth sizes
  (128/256/384/448/512/640/768) where the planner correctly prefers the shipped
  SIMD `four_step_batched` path the scalar model doesn't account for.
- **Per-size interleaved A/B (cycles, taskset -c 2, 14×9 reps) of the recalibrated
  model vs HEAD rejected it:**
  - f64 broadly *slower*: 613 +13.6%, 705 +9.8%, 915 +8.9%, 451/369 +5.8%,
    1681 +6.7%; only 1217 (−9.4%) and 984 (−2%) faster. Net negative, violates the
    no->1%-regression gate.
  - f32 *catastrophic at the boundary*: 169 (13²), 143 (11·13), 255 flipped
    four_step→Bluestein on a razor-thin model margin (e.g. 169: 14743 vs 14552) and
    ran **~5× slower**, while 451/705/984 flipped the other way for ~−45%. The
    deciding margins are **within measurement noise** — a "more accurate" table
    makes route *selection* worse, not better.
- **Verdict:** reverted the recalibration; HEAD's hand-tuned f64 cost model is the
  better router. `--decomp-report` (f64, the shipped model) kept as the
  "optimal decomposition for any N" tool. The f32 451/705/984 wins are real but
  fragile (same boundary that wins them loses 169 5×); capturing them would need
  hand-verified per-size measured overrides — not worth the bloat/layout risk per
  prior campaign traps. Left for a future measured-override pass if f32 small-
  composite throughput becomes a priority.
