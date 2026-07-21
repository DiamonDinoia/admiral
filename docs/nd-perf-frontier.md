# N-D FFT performance frontier — goal + ledger

Date: 2026-07-01. Hardware: Core Ultra 7 155H (Meteor Lake, AVX2, **no AVX-512**),
g++, single-thread, pinned `taskset -c 0`, nanobench `cpucycles` (frequency-
invariant), accuracy-gated. Same regime as the other frontier docs.

## Goal

N-D complex FFT **and** real FFT (r2c/c2r) at **parity-or-better vs ducc0 and
FFTW single-thread** on the target HW (cycle metric). Consistent with
`beat-fftw-plan.md`: **ducc0 is the beaten reference; FFTW is the ceiling.**

## Architecture recap (row–column, `nd_plan.hpp`)

An N-D c2c on a contiguous row-major tensor is a sequence of batched 1D
transforms, one per axis — not an N-D butterfly. The 1D chiplet/codelet layer is
reused verbatim; only the per-axis *addressing* is new.

- **Innermost (contiguous) axis** — each line is a contiguous `complex<T>[len]`,
  handled by the 1D `plan_impl` at **full parity** (this was never the gap).
- **Every outer (strided) axis** — a batched SIMD column DIF pass
  (`col_dif_execute_ws`) with the contiguous trailing block as the SIMD-lane
  batch; non-smooth axes fall back to a scalar gather→plan→scatter.

The inner-vs-outer split is the whole story. The diagnostic rectangle pair
isolates it: **16×256** (tiny outer axis) *wins* ~0.74, while **256×16** (big
outer/column axis) *loses* — the column pass was the gap.

## The lever: cache-block the column pass (WS3, SHIPPED)

Large pow2² / cubes are 8–33 MB — well past the 2 MB L2 (distinct from the 1D
"≤64K is L2-resident, don't block" verdict in `cache-blocking-frontier.md`). The
untiled column pass streamed the full `len·inner` slab through cache once **per
DIF pass**, thrashing where ducc0 tiles.

Fix (`nd_apply_axis`, `if (st.dif)` branch): tile the contiguous `inner`
dimension into `Bt`-wide column blocks; each block runs its whole first→middle→
last DIF chain while its working set (`len·Bt` data + 4 SoA scratch buffers) is
L2-resident. `axis_stride` stays `inner`; only `batch_count` and the slab base
shrink. Scratch drops from `len·inner` to `len·Bt`.

`Bt = nd_col_block<T>(len,inner)`: largest SIMD-width multiple whose working set
fits `kColBlockBytes` (512 KiB, a conservative fraction of L2 — a `constexpr`
**hardware-calibration knob**; `0` disables tiling), clamped to `[W, inner]`.

**asm gate** (asm-analysis): the hot `col_dif_execute_ws` kernel is untouched —
vectorization intact (packed ymm FMA/mul, masked tail), **0 new spills**; the
tiling wrapper adds only loop-counter arithmetic (0 ymm/GPR spills).

## Measured: c2c, baseline → post-blocking (ducc0 ratio, <1 = we win)

reps=9; ratios are fft/ducc0 on cycles. `us` = our forward, best epoch.

| shape | prec | ducc fwd (before→after) | ducc rt (before→after) | our fwd µs (before→after) |
|-------|------|-------------------------|------------------------|---------------------------|
| 256²      | f64 | 1.164 → **0.985** | 1.102 → **0.984** | 333 → 253 (−24%) |
| 512²      | f64 | 1.136 → **0.974** | 1.158 → **0.909** | 1820 → 1304 (−28%) |
| 1024²     | f64 | 1.018 → **0.987** | 1.390 → **1.03**  | 17406 → 7902 (−55%) |
| 64³       | f64 | 1.050 → **1.006** | 1.137 → 1.116     | 1675 → 1426 (−15%) |
| 128³      | f64 | 1.019 → **0.892** | 1.025 → **0.890** | 35159 → 15223 (−57%) |
| 256×16    | f64 | 1.043 → 0.906     | 0.988 → 1.043     | 18.6 → 15.6 (−16%) |
| 1024×64   | f64 | 1.209 → 1.118     | 1.146 → 1.099     | 491 → 369 (−25%) |
| 256²      | f32 | 1.013 → **0.887** | 0.975 → **0.852** | 173 → 150 (−13%) |
| 1024²     | f32 | 1.007 → **0.980** | 0.975 → **0.930** | 4229 → 4386 (noise) |
| 128³      | f32 | 1.013 → **0.932** | 0.977 → **0.866** | 16839 → 7872 (−53%) |
| 60²       | f32 | 1.190 → 1.214     | 1.241 → 1.252     | 12.2 → 11.7 |
| 256×16    | f32 | 1.218 → 1.265     | 1.231 → 1.199     | 12.6 → 12.8 |

Blocking is a broad win: **−15…−57% cycles** on the large pow2²/cubes, flipping
most from LOSE to parity-or-win vs ducc0. No regression survived re-measurement
(the f32 1024² +8% at reps=9 read 0.98 at reps=15 — noise on 1 M-point transforms,
err 2.5–4.8%). It benefits r2c's outer axes too.

## Honest residual

1. **Small-`inner` column axes (256×16, 60², f32)** stay ~1.0–1.27× ducc0. Their
   working set already fits L2, so blocking does nothing — the residual is the
   column pass being intrinsically slower than the row pass when the SIMD batch
   (`inner`) is small (few lanes, strided). Not a cache problem; **not chased**
   (bounded scope). Follow-up: SIMD-batch the small-`inner` / scalar-fallback path.
2. **FFTW ceiling.** FFTW ratios were captured with **FFTW_ESTIMATE** (MEASURE
   planning is intractably slow on 1024²/128³), so the reported fftw ratios are a
   *pessimistic* bound (real MEASURE FFTW is faster). Flip to FFTW_MEASURE for a
   fair ceiling once the shape set is frozen. The ducc0 result stands on its own.

## r2c / c2r (WS2, SHIPPED — correctness feature, honest perf residual)

Half-length trick (`real_fft.hpp`): even innermost N packs to N/2 complex, runs
an existing length-(N/2) `plan_impl`, then one length-N recombination pass →
N/2+1 half-spectrum. N-D = 1D r2c on the innermost axis then the c2c column
passes on the remaining axes (inherits the blocking win). Odd N → full-c2c
fallback (`ponytail:` correctness, not the perf target; bench sizes are even).
Correctness: l2 vs ducc0 r2c ≤ 2e-7 (f32) / 1e-15 (f64); r2c→c2r identity; 108/108 ctest.

**Residual (f64, reps=9): ~2.4–3.1× ducc0 forward, ~1.5–2.1× round-trip.** ducc0
has a fused, vectorized real-FFT; ours reuses the c2c engine plus a **scalar,
precision-agnostic recombination pass** — that scalar pass is the bottleneck and
the clear upgrade path (SIMD-batch the recombination over rows; the SoA principle).
**→ CLOSED in Phase 2 (WS-B), see below.**

⚠ **Measurement caveat:** the in-process `--compare-nd --r2c` **f32** cycle ratios
read implausibly high (13–19×) and unstable, while a clean plan-once/loop tight
measurement shows the engine at the normal f32<f64 scaling (256²: f32 594 µs <
f64 924 µs). Treat the in-process f32 r2c ratio as unreliable (a nanobench
in-process artifact, cf. `rdtscp-minofn-frequency-trap`), not an engine result.

## Measurement infra (WS1, SHIPPED)

`benchmark/bench_fft.cpp`: `--compare-nd[=RxCxD,…] [--r2c] [--prec] [--reps]
[--fail-on-lose]` — arbitrary rank vs ducc0-ND and (with `-DFFT_BENCH_FFTW`)
FFTW-ND, forward + round-trip, cycle-ratio. ducc0 (`ducc0_*_fft_nd`, `ducc0_r2c/
c2r_nd`) and FFTW (`fftw_c2c` general-rank `fftw_plan_dft`, `fftw_r2c`) wrappers
generalized from 1D/2D. `--compare-2d` is a thin alias.

## Go / no-go

- **GO (done):** column-pass cache blocking — broad c2c win, gate passed, shipped.
- **NO-GO (bounded, not retried here):** small-`inner` column SIMD-batching;
  r2c recombination vectorization; FFTW_MEASURE parity chase; AVX-512 / threads.
  Each is a real follow-up, deliberately out of this scope.
  *(Update: r2c recombination vectorization → SHIPPED in Phase 2, WS-B below.
  small-`inner` column → attributed + NO-GO, WS-C below.)*
- **do-not-retry:** blocking the *innermost/row* pass or small-N (≤64 KB) tensors —
  L2-resident, `cache-blocking-frontier.md` verdict holds; tiling there only adds
  loop overhead.

---

# Phase 2 — trustworthy A/B, batched real-FFT, small-inner attribution

Core Ultra 7 155H, AVX2, single-thread, `taskset -c 0`, nanobench `cpucycles`
(frequency-invariant), accuracy-gated. Ratios are **ours/competitor** — `<1.0`
means we win. FFTW on ESTIMATE (deferred to MEASURE-hardening; not this round).

## WS-A — trustworthy engine A/B (SHIPPED, the gate)

`--compare-nd [--r2c] --robust [--rounds=N]` (`bench_fft.cpp`
`compare_nd_robust` / `compare_nd_r2c_robust`, built on the `engine_ab_core`
generalization of `compare_factors_ab`): interleaved per-round timing,
**role-swap + `sqrt(mAB/mBA)`** first-touch/layout-bias cancellation, MAD spread
floor, cycle-invariant. **Mandatory identity control** (ours-vs-ours) read
**0.99–1.01 across every shape/precision** → harness trusted. This is the only
admissible gate for the numbers below; the old sequential per-engine `nb_measure`
(and its false 13–19× f32 r2c) is retired. Resolves the "measurement caveat" above.

## WS-B — batched real-FFT (SHIPPED, big win)

`real_fft_plan` even path rebuilt as a **row-batched, ISA-parametric** transform
(`real_fft.hpp`): pack W real rows into SIMD lanes (SoA), run the inner size-M
complex transform **once per W-row tile** via the lane-batched DIF multipass
(`vp::multipass_run`, reused from vecpass — no new codelets, no MAX_N growth), and
do the recombination butterfly **V-wide across rows** (split re/im twiddle ring,
scalar-broadcast per k). 1D (rows=1) and the `<W` row tail keep the scalar path.

**The killer bug was a per-tile `make_unique_for_overwrite` heap alloc** (one
malloc per W-row tile per axis — e.g. 64 mallocs/execute for 256²). Hoisting it to
one plan-owned, over-aligned `4*M` ping-pong block reused across all tiles
(`tile_scratch_`, allocated once in the ctor) is what unlocked the win — exactly
the trap in `[[vecpass-f32-and-noinit-scratch-shipped]]`.

r2c robust A/B, ours/ducc0, before (per-tile malloc) → after (hoisted), f64:

| shape   | fwd before→after | rt before→after |
|---------|------------------|-----------------|
| 256²    | 3.34 → **1.19**  | 2.02 → **0.98** |
| 512²    | 2.62 → **1.16**  | 1.57 → **0.98** |
| 1024²   | 2.28 → **1.19**  | 1.40 → **0.97** |
| 64³     | 2.47 → **0.97**  | 1.64 → **0.93** |
| 128³    | 2.09 → **1.00**  | 1.17 → **0.80** |
| 8⁴      | 1.53 → **0.57**  | —    → **0.57** |

**c2r (inverse) now beats ducc0 across essentially all dim>1 shapes (0.80–1.04);
r2c (forward) went from 2–3.7× to near-parity (0.97–1.24)** and beats FFTW on the
pow2 squares (256²/1024² f64 0.82–0.89, all f32 squares 0.70–0.81). f32 mirrors f64.
Correctness unchanged: l2 ≤ 2e-7 (f32) / 5e-16 (f64) vs ducc0, r2c→c2r identity,
**108/108 ctest** + new W-tail cases ({6,16},{14,24} — f64 W=4 & f32 W=8 tails).

**Honest residual — r2c forward ~1.15–1.24× ducc0 (pow2 squares).** *Not* the
scatter store: c2r does ~2M scalar boundary iterations vs r2c's M+1, yet c2r *wins*
— so boundary/scatter work is not the bottleneck. The gap is structural: ducc0
fuses the real-input optimization into the transform (no separate pass), while the
half-length trick pays a full size-M c2c **plus** an O(N) recombination pass.
Closing it needs a native real-input FFT (a large rewrite) — deliberately out of
this round's scope.

## WS-C — c2c small-`inner` residual (ATTRIBUTED → NO-GO this round)

Robust c2c A/B pins the residual precisely via a transpose-symmetry probe:

| shape (f32) | ours/ducc0 fwd |
|-------------|----------------|
| 16×256      | **0.703** (win big) |
| 256×16      | **1.225** (lose)    |

Same element count, axes swapped. **When the large axis is the *inner* (contiguous)
axis we win; when it is the *outer* (column) axis we lose.** So the residual is the
strided **column-pass SoA gather/scatter repack** (`col_dif_execute_ws`): the DIF
itself runs on contiguous scratch at row-pass speed, but the gather-in / scatter-out
transpose is overhead the contiguous row pass never pays. Same pattern: 1024×64 f32
(1.21), 60² f32 (1.27, also the odd-radix small-ido wall), 1024² f64 fwd (1.13).

**NO-GO (bounded):** the obvious lever — a transpose-based 2D FFT (row pass →
transpose → row pass → transpose back) — adds a *second* full transpose to convert
strided-DIF into contiguous-DIF, but the column pass already pays ~one transpose as
its gather/scatter, so the trade is marginal and high-risk. Memory is emphatic that
this is a dead end: `[[packed-kernel-spill-investigation]]` ("feeding transpose is
the O(N) wall; 1D Stockham pipeline PROVEN impossible"), and transpose-based passes
regress via instantiation bloat (`[[transpose-lane-over-b-shipped]]`). Not built —
recorded as an intrinsic column-pass repack residual. Lowest-confidence WS, bounded
NO-GO as the plan anticipated.

## WS-D — generation / ISA-portability / pruning

- **No new TUs, no new codelets.** WS-B reuses the existing `vp::multipass_run`
  (radix-set DIF multipass) for the inner size-M transform; **`FFT_CODELET_MAX_N`
  stays 64** and nothing was added to the catalog. The only new state is one
  plan-owned scratch block. Nothing to prune.
- **ISA compile check.** A standalone instantiation of `real_fft_plan<float/double>`
  compiles **clean at `-march=x86-64-v3` (AVX2: f64 W=4, f32 W=8)** — the target
  ISA. The WS-B tile/recombination code is width-generic (`V::size` throughout, no
  hardcoded W). W=2 ISAs (`-march=x86-64`/`-v2` SSE2/SSE4, and NEON-f64) fail to
  compile — but the error is in the **pre-existing** four-step `vp::vpass_forward`
  `static_assert(W==4||W==8)` (`vecpass.hpp:331`, pulled in transitively because
  `real_fft_plan` embeds a full `plan_impl`), *not* in WS-B code. Generalizing the
  four-step path to W=2 is a separate, pre-existing item; W=2 is not a target ISA
  (AVX2/AVX-512 are, both W≥4).

## Phase-2 verdict

- **GO / SHIPPED:** WS-A trustworthy `--robust` A/B; WS-B batched real-FFT (r2c
  fwd 2–3.7× → 0.97–1.24, c2r beats ducc0 broadly, beats FFTW on pow2 squares).
- **NO-GO (bounded, attributed):** WS-C small-inner column repack (transpose lever
  marginal + memory-backed dead end); r2c-forward structural gap (needs a native
  real-input FFT).
- **do-not-retry:** per-tile scratch alloc in any batched path (WS-B killer bug);
  transpose-based column pass to kill the SoA repack (`[[packed-kernel-spill-investigation]]`).

## Phase-3 — Sprint 1: c2c small-inner column (residual close attempt)

Target: 256×16 / 1024×64 / 60×60 **f32 fwd+rt ≤ 1.00× ducc0** (identity ≈1.000
robust A/B), no regression on winners. Two levers built and measured:

- **Radix-4-only factorization for small-inner pow2 f32 axes (SHIPPED lever).**
  `make_nd_axis_state` forces `build_radix4_plan(len)` (all-radix-4, one trailing
  radix-2 for odd powers) when an outer f32 axis is pow2 **and** `inner` is narrow
  (`nd_col_block/W < 4` or `inner % W != 0`). Radix-4 holds ~10 YMM (0 spills) vs
  B-vectorized radix-8's ~30-live/17-spill. **Strict improvement, no regression:**
  256×16 1.235→**1.211**, 1024×64 1.341→**1.280**, 8⁴ 0.91→**0.69**; winners intact
  (256²/512²/1024² 0.83/0.88/0.96, 64³ 0.88, 128³ 0.83, 16×256 0.71). f64 untouched
  (W=4 already fits radix-8). ctest 110/110.
- **ido-vectorized column driver (`col_dif_execute_ido`, NO-GO — reverted+deleted).**
  Attempted the memory-flagged "real fix": B-major scratch (`c*len+p`), vectorize
  over `a` (ido) per column like the 0-spill 1D `dif_pass`. Measured a clear
  regression on the target shapes: **256×16 f32 fwd 1.54, 1024×64 f32 1.85** (vs
  radix-4's 1.21/1.28), identity ≈1.000. Root cause: for pow2 with the default
  radix-8 plan, `ido` shrinks by the radix each pass (256: 32→4→1), so **only the
  first pass has `ido≥W=8` and vectorizes** — every later pass falls to scalar
  per-column, far slower than B-vectorization even with its spills. ducc0 avoids
  this with a *fused small-radix multipass* over a B-major buffer that keeps the
  SIMD lane full across passes; replicating that is the out-of-scope rewrite from
  `[[small-inner-col-dif-register-wall]]`, not a driver swap. Files deleted
  (`dif_col_pass_ido.hpp`, `dif_col_driver_ido.hpp`).

**Verdict:** ≤1.00 parity **not reached** on AVX2. Radix-4 lever is the best
shippable improvement (partial win, no regression, correct); 60×60 stays a residual
(not pow2 → radix-5 spill, lever N/A). Full parity is gated on the ducc0-style
B-major fused multipass rewrite — bounded, attributed, deferred.

## Phase-3 — Sprint 2: r2c/c2r forward (transpose-batched pack/unpack) — SHIPPED

Target: r2c fwd → ≤1.05× ducc0. Baseline (robust, identity≈1.000): f32 pow2-square
r2c **fwd 1.24–1.27**, rt 1.17–1.25 (f64 near-parity). Perf-attributed (opus agent,
perf + asm): the gap is **f32-only** and owned ~59% by the **pack/unpack scalar-lane
transpose** in `r2c_even_tile`/`c2r_even_tile` (B1 pack 2.27× / B3 recombine 1.65× per
tile vs f64), NOT the multipass (scales correctly) nor the outer column axis (scales
correctly — the accepted Sprint-1 residual). The recombination is *already* a fused
O(M) L1 post-pass, so the plan's **new recombination-aware per-M codelet TU is a NO-GO**
(a TU regenerates the same SoA→strided-AoS scatter permute).

**SHIPPED lever (real_fft.hpp only, NO new TU):** replaced the scalar `vinsertps`
gather / `vpermd` scatter with in-register **WxW `xsimd::transpose`** tiles (mirror of
`four_step.hpp`), processing H=W/2 consecutive j/k per transpose; scalar < H tail +
k=M Nyquist bin. Folded the per-k `k % M` scalar div (16K/call) into a branchless
block path. Applied to all four boundaries: r2c pack, r2c recombine-store, c2r
input-read (the `in[M-k]` mirror is a *reversed* contiguous block, consumed p=H-1-m),
c2r output-unpack. Inner compile-time loops use `poet::static_for` (better codegen:
1024² f32 fwd 1.08→0.99).

**Result (robust A/B, identity≈1.000, both prec):** r2c fwd f32 256² 1.24→**0.91**,
512² 1.27→**1.02**, 1024² 1.25→**0.99**, 64³ 1.02→**0.79**; rt f32 256² **0.93**, 512²
**1.01**, 1024² **0.96**, 64³ **0.81**; f64 fwd all ≤1.03, rt all ≤0.91 (wins). No
regression; vs FFTW 0.59–0.90 (crush). asm: float `r2c_even_tile` **0 spills, 0 div**,
74 transpose ops on the hot path. ctest 110/110.

**Verdict:** GO / SHIPPED. ≤1.05 target met across all pow2 squares + 64³, both prec,
fwd and rt. The plan's per-M codelet TU: bounded NO-GO (recombine already fused).

## Phase-3 — Sprint 3: FFTW pow2-cube gap (64^3, 8^4) — attributed, NO-GO

Perf-attributed (opus agent, perf counters + asm, robust A/B identity≈1.000). We
BEAT ducc0 on both cubes; the loss is vs single-thread FFTW (ESTIMATE).

**64^3 f32 (1.39x vs FFTW): PRIOR CONFIRMED — bounded AVX2 NO-GO.** Per-axis split
(perf cpu_core/cycles, L1/L2-miss = 0 → compute-bound, L2-resident in tiles): row
(innermost) 52.9%, col1 24.4%, col0 24.8%. Algebraic vs FFTW: the row pass is 2.37x
(= the *identical* measured 1D N=64 codelet gap), the two col passes are **0.95x —
we beat FFTW**. The entire 64^3 gap is the 1D N=64 row codelet; the N-D column engine
adds nothing. Closing it needs genfft codelets = AVX-512-only
([[genfft-fused-codelet-avx2-donotretry]], [[pow2-fftw-gap-phase0-go]]). No AVX2 lever.

**8^4 f32 (1.93x vs FFTW): PRIOR REFUTED — three N-D-specific mechanisms.** Not the
1D codelet frontier (1D N=8 *beats* FFTW 0.91x). (A) innermost N=8 has ido=1 → the
radix-8 butterfly runs fully SCALAR (0 AVX) and nd_apply_axis serializes 512 per-line
plan_impl calls while FFTW batches (howmany=512) — 39% of cycles. (B) col2 inner=8=W:
nd_col_block Bt=8 fragments the pass into 64 single-SIMD-batch calls, per-call
overhead 1.75x col0 — 30%. (C) col-driver AoS/SoA deinterleave + broadcast-twiddle
structural cost 1.35x even at best B-amortization — needs a col-driver redesign.
128^3 is a tie (0.97x): large inner extents dilute the row gap + stream memory equally.

**Recorded lever (DEFERRED, not built): transpose-batched innermost + small-inner col
batching.** For a single-radix innermost axis (n_passes==1, ido==1, e.g. N≤8) batch
the lines via an in-register transpose (8 consecutive lines = one contiguous 8x8 tile,
like the r2c pack) instead of 512 scalar calls (est −22%); plus gather-batch the
inner≤W col pass (est −11%). Combined ceiling ~1.30x — **still loses FFTW**, targets
only niche tiny-cube shapes (8^4/8^3/4^5), is correctness-critical, and leaves the [C]
col-driver residual. **NO-GO on ROI** (plan deprioritized Sprint 3; even the win loses).

**Verdict:** 64^3 = confirmed bounded AVX2 NO-GO (genfft frontier). 8^4 = attributed to
addressable N-D batching inefficiencies, but the best AVX2 ceiling still loses FFTW →
deferred, lever recorded for a future tiny-cube sprint if it becomes a priority.

## Phase-4 — small-inner ducc0 gap: chiplet / B-major / AoS all NO-GO (AVX-512-only)

Goal: beat ducc0 on the residual small-inner rectangles (256×16 f32 **1.25×**, 1024×64
**1.29×**, 60² **1.21×**) via the chiplet approach, ISA-parametric. Four opus spikes, each
asm-grounded (objdump/llvm-mca + robust A/B, identity≈1.000). All NO-GO. Extends
[[small-inner-col-dif-register-wall]].

**The single binding constraint — radix-8 spill wall is LAYOUT-INDEPENDENT.** A radix-8
butterfly needs 8 arms × 2 (re/im) = **16 YMM = the entire AVX2 register file**, before any
twiddle or intermediate. `vpass_one<8,float>` = **19 spills** (compiled with project flags).
This ceiling is fixed by butterfly geometry (arm-count × 2 × W-lanes), not by scratch layout:
B-major, ido-major, and AoS all hit it. ducc0's only edge is a 3-pass radix-8 ({8,8,4})
vs our 4-pass radix-4 cap; radix-8 spill-free requires **32 registers (AVX-512 ZMM)**.

**Spikes (all NO-GO):**
1. *Fused-radix-8 chiplet.* Our radix-4 col pass is ALREADY 0-spill at the 16-YMM ceiling;
   the loss is pass-count/dispatch overhead (23% FP util), not spills. A bigger codelet
   cannot help.
2. *Depth-first W-column-tile gather driver.* Reuses vecpass but adds two extra full-N
   gather/scatter SWEEPS → 256×16 f32 1.22→**1.38 WORSE** (IPC 3.01 vs 3.64). Same
   memory-traffic wall as `four_step_batched`, even at L1-resident N=256; the standard path
   already folds deinterleave/interleave into first/last pass.
3. *B-major via vecpass.* `vpass_forward` MIXES lanes (Cooley-Tukey four-step combine of W
   streams into ONE N-point DFT ≠ W independent columns — 11 test failures) AND still
   19-spills. STEP-A refuted before implementation.
4. *AoS-native DIF* (arms interleaved, W/2 complex/YMM, `vpermilps` twiddle). Refuted by the
   same geometry: at B=8 an arm is 8 complex = 2 YMM (spills identically); at B=4 it "fits"
   only by halving column parallelism AND adding a shuffle-port uop per twiddle per pass on
   the already dispatch-bound kernel = strictly worse.

**FMA/arithmetic: already optimal everywhere** (full audit). Every complex twiddle is
plain-expr `owr*sr − owi*si` → single `vfnmadd231`/`vfmadd231` (`xsimd::fnma` correctly
absent — it misses FMA3 on avxvnni). Odd-radix kernels at the conjugate-symmetry multiply
minimum. Only non-fuseable spot = the diagonal twiddle `c*(fr−fi)` (mul-after-sub, structural).
No arithmetic lever exists in 1D/2D/N-D.

**Verdict:** The current `col_dif_execute_ws` is ALREADY B-major (column = SIMD lane),
already 0-spill radix-4, already folds AoS↔SoA at the boundaries — **the AVX2 Pareto
frontier** for small-inner column transforms. We beat ducc0 on all squares/cubes/16×256/8⁴
and FFTW on all squares; the residual narrow-rectangle (ducc0) and 64³/8⁴ (FFTW) losses are
AVX-512-only (32 ZMM → radix-8 0-spill at W=16). Accepted as the AVX2 frontier. AVX-512-gated
radix-8 column chiplet recorded as the future lever (see branch `shelved/avx512-radix8-regbudget`).

---

## Multithreading (opt-in, plan-owned pool)

`fft::plan<T>(shape, nthreads)` / `fft::plan_r2c<T>(shape, nthreads)` cache
`nthreads-1` worker threads in the plan (`detail::thread_pool`, `std::jthread`
parked on a condition_variable) and thread the **batch loops** — the innermost
row pass, the batched-DIF column pass (over slab×column-tile), the scalar
fallback column pass, and the r2c/c2r real tile loop. Static contiguous chunking,
per-worker scratch allocated once per chunk. A single large 1-D transform stays
serial (four-step-parallel is Phase-4 deferred; ducc0 also serializes 1-D).

- **`nthreads == 1` (default) is byte-identical to the pre-threading serial path**
  and spawns zero threads — the tuned single-thread path is untouched.
- Threaded vs serial output is **bit-identical** (rows/columns/tiles are
  independent; chunking doesn't change per-line math). Gated by
  `test/test_fft_threads.cpp`.
- Dispatch gate: only threads when `outer >= 2*nthreads && total >= 1<<15`
  (avoids wakeup cost dominating small transforms; tune `kThreadMinElems`).

### Benchmark pinning (fair MT wall-clock)

CPU-cycle ranking counts only the calling thread, so `--nthreads>1` forces the
wall-clock metric. Pinning rules:

- **Single-thread runs:** `taskset -c 0 ./fft_benchmark --compare-nd ...` (as today).
- **MT runs must NOT pin to one core** — give the process all cores, e.g.
  `taskset -c 0-7 ./fft_benchmark --compare-nd --shapes=1024x1024,512x512x512 --nthreads=8`.
- Build the references with threads: `-DFFT_BENCH_THREADS=ON` (links
  `Threads::Threads` into ducc0; adds `fftw3_threads` when `-DFFT_BENCH_FFTW=ON`).
  Default builds stay exactly as before (references single-threaded).
