# The optimal plan to beat FFTW — derived analysis

Date: 2026-06-29. Hardware: Core Ultra 7 155H (Meteor Lake, AVX2, **no AVX-512**),
g++, pinned `taskset -c 0`, nanobench `cpucycles` (frequency-invariant), accuracy-gated.
This is the "separate plan" the pow2 frontier doc deferred to.

Scope decision (user): **portable across ISAs** — must hold on AVX2 *and* scale onto
AVX-512, codelet selection gated on `poet::vector_register_count()`.

---

## 1. State of play (fresh A/B, fwd, fft/FFTW; >1 = we lose)

We **beat ducc0 almost everywhere** (0.66–0.94). FFTW is the real, harder ceiling.

| class | f64 | f32 | notes |
|-------|-----|-----|-------|
| pow2 512/1024/2048 | 1.54/1.54/1.59 | 1.96/1.72/1.44 | flat ~1.5× wall |
| pow2 4096 | 1.14 | 1.65 | f64 memory-onset |
| composite 1260/1500 | 1.14/1.22 | 1.40/1.46 | |
| composite 2520 | 1.15 | **1.02** (rt **0.98 WIN**) | shipped f32 override |
| composite 3360/4200 | 1.24/1.39 | 1.09/1.03 | f32 near-parity |
| composite 5040/6300 | 1.36/1.34 | **1.59**/1.43 | f32 5040 = worst |
| composite 7560 | **1.03** (rt **0.99 WIN**) | 1.23 | f64 near-parity |
| composite 10080 | 1.35 | 1.12 | |

Existing FFTW wins: **f32 2520 rt (0.98), f64 7560 rt (0.99)** — both via the
ido-ordering override table. They prove composites are winnable per-size.

## 2. Root-cause decomposition (measured + asm)

The FFTW gap is **two distinct mechanisms**, not one:

### 2a. pow2 = FP op-count (CONCLUSIVELY AVX2-closed)

FFTW's fused large-radix genfft codelets do fewer FP ops/point. Our kernel is
FP-throughput-bound: IPC 3.0, **0 cache stalls, 0 spills**, FP ports saturated.
We *already* fuse the output twiddle (a per-codelet port ties on AVX2); late-load
scheduling spills *more* (9/8 vs 0/0) and runs 17% slower; radix-16 register-starves
16 YMM (512 → 3.48× FFTW). **Closing it needs fused large-radix codelets, which need
AVX-512's 32 registers.** See `pow2-fftw-codelet-frontier.md` (do-not-retry on AVX2).

### 2b. composite = scheduling + data-movement (NOT op-count)

**(i) Scheduling — bounded, recoverable, per-size.** The DP cost model
(`twiddles.hpp:61` `dif_stage_cost`) is purely additive and has **no `ido` term**, so
it mis-places large radix. Forced-ordering sweep vs the FFTW gate (f32):

| size | DP default /FFTW | best reorder /FFTW | recoverable |
|------|------------------|--------------------|-------------|
| 5040 | 1.601 (`3-3-4-4-5-7`) | **1.212** (`2-3-3-5-7-8`, radix-8 innermost) | **−24%** |
| 1500 | 1.389 (`3-4-5-5-5`) | 1.344 (`5-3-4-5-5`) | −3% |
| 1260 | 1.419 (`3-3-4-5-7`) | 1.419 | **0%** |

The 5040 win is radix-8 at the **innermost pass (ido=1)** — no twiddles, unit stride,
W lanes full. The DP never finds it (flat radix-8 penalty regardless of ido).

**(ii) Data-movement — the irreducible residual is spill-bound, NOT FP-bound.**
asm census of 1260's odd-radix f32 passes (the size with 0 scheduling headroom):

| pass | insns | FP arith | spill stores | FP share |
|------|-------|----------|--------------|----------|
| `dif_pass<float,5>` | 1653 | 208 | **125** | 13% |
| `dif_pass<float,7>` | 1035 | 192 | 32 | 19% |

Only 13–19% FP arithmetic; the rest is loads/stores/spills/lane-ops. This is the
**opposite** of pow2 (op-bound, 0 spills). The spills are structural on AVX2 (radix-7
unavoidable below `2R+10` live without losing the sym-DFT mult saving; radix-4 GPR
base-pointer spill is source-unfixable — GCC re-folds). They come from **ido-SoA
vectorization of odd radices at small ido**, which FFTW/ducc0 sidestep via codelets +
l1-dimension batching. **E4 static analysis: AVX-512's 32 ZMM make radices 2–8
spill-free** → this residual is AVX-512-recoverable, but by spill elimination, not by
op-count reduction.

> Caveat: 2b(ii) is a static whole-symbol census (`simdref` is broken on this box).
> Execution step 1 = dynamic `perf annotate -C 0` confirmation that the spills are in
> the hot loop, not a cold remainder. Corroborated by prior dynamic data in memory.

## 3. Competitive landscape (sources cloned in `~/scratch`)

| lib | design | coverage | vs us |
|-----|--------|----------|-------|
| **FFTW** | genfft codelet generator + MEASURE planner | pow2 + composite | the ceiling |
| **SLEEF** (`sleefdft`) | own generated codelets (radix ≤64/128, **per-ISA**) + MEASURE + disk-cached plans | **pow2-only** | strong on AVX-512; absent on composites |
| **ducc0/pocketfft** | l1-batched SIMD + √N twiddle | all | **we beat it** |
| **blackwer/fft_bench** | neutral Flatiron gate, Google Benchmark | **pow2-only**, ST+MT, Rome/Skylake-X/**Icelake (AVX-512)** | we're not in it yet |

Two strategic facts: **SLEEF is pow2-only** → composite coverage is our uncontested
ground. **fft_bench tests pow2 on AVX-512 hardware** → it is exactly the pow2/AVX-512
arena, and adding our header-only lib both gives a neutral head-to-head and solves our
lack of AVX-512 hardware.

**Unifying insight:** both gaps trace to one design difference — fused codelets
(+ l1-batching) that avoid both the extra FP ops (pow2) and the ido-SoA shuffle/spill
dance (composite). Our generic ido-vectorized radix-≤8 engine beats ducc0 but is
structurally below FFTW. The single highest-leverage change is a fused/generated
large-radix codelet path — and it pays off on AVX-512 (register file), which is also
where Flatiron deploys and where fft_bench measures.

## 4. The plan

**Honest framing: we will not beat FFTW broadly on AVX2.** Achievable: (1) a curated
set of composite sizes to parity-or-win via overrides; (2) the real prize — an
AVX-512 codelet path attacking both gaps where the hardware actually is.

### Track 1 — composite scheduling (AVX2 + AVX-512, cheap, ship now)
1. **Root-cause DP fix (try FIRST — generalizes).** `dif_stage_cost` (`twiddles.hpp:61`)
   gives radix-8 a blanket `1.16×` for `n>512` regardless of `ido`; but radix-8 as the
   **innermost/last pass (ido≤lanes)** is its *cheapest* placement (no twiddles, unit
   stride, W lanes full). The full 5040 f32 sweep (2582 orderings) confirms **every**
   top ordering has radix-8 last (`5-2-3-3-7-8` 0.728 vs ducc0, proj. ~1.15–1.18 vs
   FFTW; DP default `3-3-4-4-5-7` uses no radix-8 at all). Candidate one-liner:
   `if (radix == 8 && ido <= lanes) cost *= 0.78;`. Should let the DP discover
   radix-8-last orderings automatically across many sizes.
   **Gate: full-catalog interleaved cycle-true A/B** — global DP changes perturb ~82% of
   smooth sizes (`ido-aware-dp-cannot-reproduce-table`); if it regresses anything, revert.
2. **Fallback: per-size f32 overrides.** If the DP fix regresses the catalog, hardcode
   the measured winners into `measured_dif_factor_plan` (`twiddles.hpp:140`) per precision
   — e.g. 5040 f32 → `{2,3,3,5,7,8}` (1.60→1.21). The proven 2520 mechanism; surgical,
   zero catalog risk. (Watch instantiation bloat — `transpose-lane-over-b-shipped`.)
3. Sweep the residual losers (f32 3360/4200/6300/7560/10080; f64 candidates) with the
   **existing** `--factor-sweep` + `--compare --factors` against the FFTW gate — a driver
   *script*, **no runtime searcher** (that verdict stands).
   Expected: a handful to parity/win, several narrowed. Not a broad AVX2 win.

### Track 2 — AVX-512 codelet path (the real prize; needs AVX-512 hardware)
1. **Codegen audit first (local, cheap, no hardware):** cross-compile representative
   passes/codelets `-march=sapphirerapids`; objdump-confirm W→8 f64 / 16 f32,
   `poet::vector_register_count()`→32 relaxes spill gates, radix-16 fits spill-free
   (E4 predicts yes). GO/NO-GO on this.
2. Enable larger-radix codelets gated on `poet::vector_register_count()` (radix-16,
   maybe 32) — SLEEF/genfft approach, instantiated **only** with 32 registers. *This*
   is "more chiplets for different sizes": real on AVX-512, register-starved on AVX2.
3. Re-derive the override table on AVX-512 (W doubles → ido economics change; the 2048
   f64 `{8,8,4,8}` override is AVX2-calibrated). Revisit the two AVX-512 intervention
   points: `vecpass_supported` allowlist (`vecpass.hpp:248`) and the f64 `select_route`
   block that strips vecpass/four_step_batched (`plan.hpp:199`).
4. **Validate on real AVX-512 via blackwer/fft_bench** (add our lib as a backend) —
   neutral vs FFTW/MKL/SLEEF on Icelake/SPR. Needs a Flatiron node or remote env.

## 5. Do-NOT-retry ledger (from memory + this analysis)
- genfft scheduling / radix-16 on AVX2 (register wall, conclusive).
- f64 four-step / vecpass (structure isn't the f64 bottleneck).
- global DP surcharge recalibration (perturbs catalog).
- interleaved-swizzle whole-transform (reverted).
- a runtime factor searcher (offline static table is the right form).

## 6. Answer to "do we need more chiplets optimized for different sizes?"
- **AVX2: no.** We beat ducc0; the gap is scheduling (overrides) + structural spills,
  not missing kernels. Bigger codelets register-starve 16 YMM.
- **AVX-512: yes** — radix-16/32 codelets gated on register count are the lever there,
  and the table must be re-derived for W=8/16. That is the genuine "more/bigger
  chiplets per register budget" need.

---

## 7. Phase 0 results (2026-06-29 session; simdref reinstalled 0.0.4 via `uv tool install`)

### 7a. AVX-512 codegen audit — our radix ≤8 kernels: **spill-free GO**

Cross-compiled the production butterflies (`radix_sym_dft<5>`, `<7>`,
`pow2_dif_butterfly<8>`, f32+f64, via `~/scratch/genfft_compare.cpp`) at
`-march=native` (AVX2, 16 YMM) vs `-march=sapphirerapids` (AVX-512, 32 ZMM;
xsimd widths →16×f32 / 8×f64). Census separates **true register-pressure
evictions** (a computed ZMM stored to stack then ZMM-reloaded) from `T*`-ABI
input-widen / output-narrow scaffolding (YMM-pair staging at the `float*`/`double*`
boundary, which vanishes in-engine where SIMD arrays pass in registers).

| kernel | AVX2 spills (st+ld) | AVX-512 true evict | zmm_live | FP ops (both arch) |
|--------|--------------------:|-------------------:|---------:|------:|
| r5 f32 | 8  (5+3)   | **0** | 22/32 | 36 |
| r5 f64 | 7  (4+3)   | **0** | 21/32 | 36 |
| r7 f32 | 24 (16+8)  | **0** | 32/32 | 72 |
| r7 f64 | 25 (15+10) | **0** | 32/32 | 72 |
| r8 f32 | 32 (20+12) | **0** | 32/32 | 59→62 |
| r8 f64 | 32 (20+12) | **0** | 32/32 | 59→62 |

FP-arith op count is identical across arch — same computation graph, wider lanes
just fit in register. **Confirms E4:** the composite AVX2 residual (§2b-ii) is
spill-bound and AVX-512-recoverable by spill *elimination* (not op-count). GO for
the existing radix ≤8 path on AVX-512. Method: objdump/grep per asm-analysis
workflow §6a (simdref now available for the latency/CPI layer if needed).

**Caveat / open crux (being tested next):** r7 and r8 already hit **32/32 ZMM
live** — zero headroom. A radix-16 chiplet needs ~2·IP SoA vectors just for in/out
→ likely exceeds 32 live and re-spills *even on AVX-512* unless the recursive
even/odd split keeps peak-live ≤32. So "bigger chiplets" is not a free lunch at 32
regs; the chiplet pow2-op-count win is only real if the larger radix stays
spill-free. Testing radix-16/32 codegen at `-march=sapphirerapids` directly.

### 7b. radix-16/32 "bigger chiplet" codegen — the chiplet ceiling, even on AVX-512

Added `pow2_dif_butterfly<16>`/`<32>` (recursive split-radix DIF) to the
cross-compile. True compute-spill census (excluding `memcpy`/ABI array staging):

| kernel | AVX2 (16 YMM) spills | AVX-512 (32 ZMM) spills | regs |
|--------|---------------------:|------------------------:|-----:|
| r16 f32 | 68+49   | **36+29 — still spills** | 32/32 |
| r16 f64 | 77+43   | **0** (compute; leans on memcpy staging — borderline) | 32/32 |
| r32 f32 | 206+168 | 147+99 | 32/32 |
| r32 f64 | 203+168 | **0** (compute, but 850-insn body — uop-cache stress) | 32/32 |

**Verdict — the chiplet approach falls short even on AVX-512 as a naive template
instantiation.** Our recursive split-radix at IP≥16 keeps all 16/32 SoA inputs
live across both even+odd halves → peak-live > 32 ZMM. r16 f32 and both r32 still
spill at 32 registers; only r16 f64 fits (and that borderline-relies on GCC's
memcpy array-staging). FFTW's spill-free large-radix codelets come from **genfft's
live-range-minimizing SCHEDULE**, not from a bigger radix per se — so closing the
pow2 op-count gap needs a scheduled-codelet *generator* (large effort), not just
`pow2_dif_butterfly<16>`. Extends `genfft-fused-codelet-avx2-donotretry`: the
register wall bites at IP≥16 even with 32 ZMM.

**Chiplet GO/NO-GO (vs the user goal "beat FFTW via chiplets"):**
- radix ≤8 spill elimination on AVX-512 — **GO** (§7a); needs AVX-512 hardware to *run*.
- r16 f64 codelet on AVX-512 — narrow opening, untested for speed, f64-only, hardware-gated.
- big fused chiplets (r16 f32, r32) to close the pow2 gap — **NO-GO** even on AVX-512
  without a genfft-style live-range scheduler.

**Reassessment (per goal "if we fall short, investigate why then reassess"):** on
THIS AVX2 box the chiplet lever cannot run at all (16 YMM); even granted AVX-512 it
only buys radix ≤8 + marginal r16 f64, not the big chiplets that close the pow2 gap.
The realistic beat-FFTW path *here* is composite **scheduling** — Track 1 / Phase 1
(DP fix) + Phase 2 (overrides), where 2520 f32 already wins and 5040 f32 is ~24%
recoverable. The full chiplet path (radix ≤8 spill-free + scheduled r16+ codelets)
is real but AVX-512-hardware-gated → Track 2, deferred to a remote AVX-512 env.
The FFTW codelet head-to-head census (Phase 0b) is now *optional* — the register-wall
"why" is grounded in OUR-side asm; the census would only add FFTW's exact op-count
+ the Track-2 schedule template.

### 7c. Phase 1 SHIPPED — f32 ido-aware DP fix (composite scheduling): broad FFTW win

`dif_stage_cost` (twiddles.hpp): the last-pass / small-ido surcharge keyed on `radix>4`
was mis-applied to pow2 radix-8 — §2b-ii showed only ODD radices (5/7/11) spill in the
ido-SoA dance, while radix-8 at `ido<=lanes` is its *cheapest* spot. Scoped the
surcharge to odd radices **for f32 only** (W=8; f64's W=4 economics differ and a blanket
fix regresses f64 — kept unchanged). The DP now places radix-8 innermost. One f32
counter-override: `448→{4,4,4,7}` (the lone f32 size where radix-8-last loses, ~2-3%).

Validation: **ctest 96/96**; cyc-true pinned single-size 7× paired A/B (old vs new
ordering via `--factors`, vs the in-bench FFTW gate); rebuilt-DP `--compare` confirms it
in the shipped build. **f64 factorizations byte-identical to baseline → zero f64 risk.**

f32 `fft/FFTW` fwd, old→new (clean single-size cyc, lower=better):

| size | old → new | | size | old → new |
|------|-----------|-|------|-----------|
| **3136** | 2.24 → **1.00** (rt 2.02 → **0.945 — BEATS FFTW**) | | 2352 | 2.10 → **1.07** |
| 3920 | 2.15 → **1.06** | | 5488 | 1.97 → **~1.00** |
| 784  | 2.28 → 1.11 | | 1344 | 2.09 → 1.18 |
| 1792 | 1.91 → 1.20 | | 2240 | 1.75 → 1.17 |
| 1008 | 2.01 → 1.21 | | 6720 | 1.72 → 1.14 |
| 5040 | 1.68 → 1.27 | | 336  | 1.78 → 1.19 |

~25 f32 composite sizes improved (many to FFTW parity); **3136 f32 roundtrip beats
FFTW**. vs ducc0 several flip to wins (5040 1.065→0.798). No f32 regression >1%.
The chiplet goal's *reassessed* path — composite scheduling — delivers measured
FFTW parity/wins on AVX2.

**Open follow-up (Phase 2):** f64 shows ~23 analogous radix-8-last improvements
(3136 f64 1.47→1.10, 6720 1.42→1.14, 6000 1.58→1.30) but 6 regressions (112 +11%,
1792 +5%, 48/336/720/1296 ~3%) → capture via a both-precision DP fix + 6 f64
counter-overrides. The AVX-512 chiplet path (radix ≤8 spill-free + r16-f64) remains
Track-2 / remote-hardware.

---

## WS8 standing (2026-07-10, HEAD 290009b)

WS8 (P0 census → P1a copy-free OOP → P4 tiny-N codelets → P1b/P1c pass
restructures → c5 fusion admission → c7 dif_pass_last tail elimination →
BASECOST refresh) is closed. Receipt-grade standing vs in-process
MEASURE-planned FFTW (`--fftw-ab`, FFT_BENCH_FFTW_MEASURE=1, fwd ratios):

- **v4 outright wins**: 360 f64 **0.79**, 720 f32 **0.62**, 2520 f64 **0.92**;
  4096 f64 1.07 / 512 f64 1.12 / 2048 f64 1.20 (was the 1.2–2.0x store-bound
  band at WS8 P0). Worst v4 cell: 120 f32 1.74 (was 2.48).
- **v3**: 360/720 f32 ~1.08 fwd with **rt < 1** (0.94/0.96); mid-pow2 1.4–1.7
  (instruction volume on 16 regs — WS8 P0 named reason stands).
- **v2**: 1.5–2.5 uniformly; FFTW's AVX2 codelets make this a yardstick
  mismatch — ducc0 (same -march) is the fair v2 opponent (rt3 tables).
- **60/120**: largest remaining structural losses everywhere (real FLOP
  deficit; genfft-style larger straight-line leaves are the open lever).

Full tables: bench-results/ws8_p6_fftw_ab_{v2,v3,v4}.txt,
bench-results/rt_{v2,v3,v4}_t{1,2}.txt, ledger addendum in
bench-results/ws7_p0_loss_ledger.md.
