# Wide radices (16/32) + clean-slate planner design

## Evidence (2026-07-06, Sapphire Rapids w5-3435X, clang-18, x86-64-v4)

### The v4 f64 8192 loss is structural, not chain-order

- Exhaustive 1706-chain factor sweep (old radix set {2..11}), pinned core 0:
  best chain [4,4,4,4,4,8] = 1.046 vs ducc0-v4; production DP chain [4,4,8,8,8]
  = 1.28. NO old-space chain beats ducc0. (bench-results/ws1_v4_f64_8192_factorsweep.txt)
- Per-pass PMU: radix-4 f64 passes DO NOT scale W4→W8 (2.6→3.1 cyc/elem,
  a REGRESSION) while radix-8 scales 1.7–1.9×. The r4 pass is latency-bound:
  IPC 0.60, 17% of cycles stalled on L1d miss, 43M ld_blocks.address_alias per
  20k spans (4KB j-plane strides). Full 512-bit width utilization — width is
  not the problem; sweep count and stream latency are.
- 4K-alias offset sweep (dst shifted 0..2048B): current page-congruent layout
  sits near a LOCAL optimum (0B: 2.62, 64–512B: 3.5–3.8, 2048B: 2.4 cyc/elem).
- Software prefetch T0, ~16 lines ahead, loads only: 2.65→2.40 cyc/elem (+9%)
  on the streaming r4 pass; prefetch-for-write regresses. Generic (any x86).

### Radix-16/32 passes at 32 vector registers (the AVX-512 unlock)

`pow2_dif_butterfly` is radix-generic (recursive split-scope); radix-16/32 were
never admitted because 16-YMM AVX2 register-starves them (frontier doc Phase A2
do-not-retry). At 32 ZMM they run clean:

| pass (f64, span 8192, v4) | cyc/elem |
|---|---|
| r4 (ref) | 2.5 |
| r8 (ref) | 3.3 |
| r16 | 3.8 |
| r32 | 4.6 |

End-to-end 8192 f64 chains (standalone probe, bit-exact vs production kernels):

| chain | tsc/elem |
|---|---|
| [4,4,8,8,8] (production shape, unfused) | 13.06 |
| [16,16,32] | 10.73 |
| [32,16,16] | **9.56** |

Production-path A/B (--factors-ab, rounds=9): [32,16,16] f64 A/ducc = 1.098
(from 1.28); f32 8192 [32,16,16] 0.688 / [32,8,32] 0.661 (from 1.06 borderline).

### Shipped wiring (branch ws1-valley)

- `dif_wide_radices = poet::vector_register_count() >= 32` (consteval, no CPUID)
  admits 16/32 into `dif_radix_set` (kernel dispatch) and the pow2 enumerator
  candidates. DP factorizer candidates for composites unchanged pending WS2.
- Cost seeds: r16 ×0.79, r32 ×0.63; the 2r+10 register-spill penalty scoped to
  radix<16 (recursive kernel peak-live bound is much lower — measured no-spill).
- v2/v3 compile to byte-identical planning behavior (gate is compile-time).

## Clean-slate planner mandate (user, 2026-07-06)

Replace the accumulated DP patchwork (magic multipliers, n<=512 branches,
post-DP odd-radix reorder, per-size override table, separate sweep-aware pow2
enumerator) with:

1. **Compile-time config**: radix set, width, register count — all
   constexpr/consteval from `xsimd::batch<T>::size` + `poet::vector_register_count()`.
2. **One measured cost table**: cyc/elem per (radix, regime ∈ {first, vec,
   valley, last}) per width-class, from the calibration matrix
   (`scratch_calibrate.sh` → `bench-results/calib_<lvl>.txt`). No magic
   multipliers; the valley term as f(ido, W) subsumes the odd-radix reorder rule.
3. **Linear enumeration**: pow2 chains precomputed as a consteval table
   (chain per lg2 N); runtime DP over the same cost function only for
   composites. No std::numeric_limits::infinity (UB under -ffast-math).
4. **No per-size overrides**: delete `measured_dif_factor_plan` entries the
   model reproduces; keep only provably cross-pass effects, documented.
5. Acceptance: DP-chosen chain == measured-best (--factor-sweep) or within 3%
   on the probe set, every ISA level, both precisions.

## Production truth table (2026-07-06, --compare-nd --robust, identity-gated)

Current branch (wide radices in), v4, 1D, vs ducc0-v4 fwd:

| N | f64 | f32 | note |
|---|---|---|---|
| 8192 | **0.730** | 0.552 | the "sole loss" is GONE (old 1.17-1.35 was partly a CMP-tool artifact) |
| 16384 | 1.035 (tie) | 0.609 | |
| 32768 | **1.132 LOSE** | 0.737 | |
| 65536 | **1.180 LOSE** | 0.754 | also loses to FFTW (1.163) |
| 131072 | 0.931 | 0.504 | |

vs FFTW (system 3.3.10, runtime-dispatched AVX-512 — apples-to-apples at v4
only): win/tie everywhere except 65536 f64.

### Chain A/B (compare_factors_ab, role-swapped): old fused vs wide

- f64: old fused wins 8192 (0.883) and 65536 (0.882), ties 16384/32768,
  wide wins 131072 (1.150). BUT both families lose to ducc0 at 16384-65536 —
  chain choice alone cannot close that band.
- f32: wide radices crush old chains at every size (A/B 1.13-1.63,
  ducc ratios 0.58-0.76). Admission is unconditionally right for f32.

### Footprint calibration (calib_v4_footprint.txt): the missing cost dimension

f64 first-pass (big-ido) cyc/elem vs span footprint:

| radix | 128KB | 1MB | 4MB |
|---|---|---|---|
| 4 | 3.03 | 4.32 | 4.85 |
| 8 | 3.79 | 5.29 | 5.63 |
| 16 | 4.16 | 6.55 | 10.57 |
| 32 | 5.08 | 10.40 | 11.82 |

Wide radices degrade badly with footprint at big ido (r32 = 32 concurrent
4KB-strided input streams vs r4's 4). Mid passes (ido=64, big l1) stay flat.
Valley placement of r16/32 is 5-7x vec cost (20-23 cyc/elem). The WS2 cost
table must be keyed (radix, regime, footprint-class).

### Do-not-retry: software prefetch in dif_pass

Interleaved binary A/B (bench-results/pf_ab_interleaved.txt; plain vs
__builtin_prefetch T0 16-lines-ahead, loads only, footprint-gated): tie to
slight loss on every (radix, span) probed (r4 ido=16384: 4.19-4.57 plain vs
4.57-4.71 pf). The standalone prototype's +9% does not reproduce in the
production kernel. Reverted. Measurement rules reconfirmed: serial runs of
streaming benches hours apart are not comparable (freq state), and streaming
(>L2) benches must never run concurrently (DRAM contention) — interleave on
one core or discard.

## Override re-audit (2026-07-06, bench-results/ovr_audit_{v2,v3,v4}.txt)

All 12 measured_dif_factor_plan entries re-A/B'd against the NEW measured-cost
DP chains (role-swapped cyc, rounds=15x2, pinned, all three ISAs):

- **Deleted (tied or lost everywhere):** f64 48, 1500, 2048, 2700, 5040, 7056.
  5040 f64 is the headline: the new DP chain [5,3,7,3,2,8] beats the old hand
  order [5,4,4,3,3,7] by 1.41x at v4 (ducc 1.208 -> 0.857).
- **Kept:** f64 1260 (wins 0.896 at W=4 only, ties elsewhere); f32 2520/5292/
  2048/4096 (win 0.82-0.88 at W=16, win/tie at W=8/W=4); f32 7560 now gated to
  W>=16 (wins 0.882 there, loses 1.023 at W=8).
- Pattern in the survivors: the DP appends a cheap radix-2/valley tail or
  buries an expensive odd mid-chain — cross-pass placement the additive model
  can't rank.

## v3/v2 acceptance + regs16 re-fit (2026-07-06, bench-results/accept_ab_v3v2.txt, accept_ab2_newchains.txt)

Every size whose DP chain changed at v3/v2 was A/B'd old-vs-new (38 pairs,
role-swapped cyc, rounds=15x2). 25 tied, 3 new-chain wins (v2 8192 f64 1.119,
16384 f32 1.069, 32768 f64 1.090 — kept), 10 robust regressions. Fixes, all
verified by dp_probe chain equality (v4 chains byte-identical throughout):

- **r8 rows re-fitted in-chain** (f32 W8 vec 3.08→3.00; f32 W4 {5.37,1.59}→
  {2.60,1.15}; f64 W4 last 2.95→2.70, r7 vec 3.17→3.50; f64 W2 untouched).
  The isolated `--pass` probes run hot-in-L1 where r8's spills dominate; in a
  chain they hide behind memory traffic, so the isolated numbers mis-rank
  r8-tail chains that measurably win (v2 1024 f32 [4,4,8,8] by 1.19x).
- **Ordering epsilon** (`1e-9·radix·n`, regs<32 only): equal-cost permutations
  of one multiset are additive-model ties; measurement consistently wants
  ascending radix at 16 regs ([2,3,3,5,8] beats [5,2,3,3,8] by 6-15% at
  720/5040). 32 regs measures the opposite and keeps the large-radix tie-break.
- **Overrides added (model-unfittable, measured wins):** f64 16384 W<=4
  [4,4,4,4,8,8] (0.936/0.980) — 16384 wants the 8-8 tail while 32768 at the
  same table wants mostly-4s, outside any constant r8 fit; f32 W4 16384 all-4
  (1.069) vs 1024/65536 wanting 8-8 tails (0.838/0.882) — the sandwiched
  preference at the 128KB side; f32 32768 W<=8 [4,4,4,8,8,8] (0.831 vs all-4,
  1.086 vs the enumerator's [4,4,8,4,8,8]) — the enumerator ducks the ido>=512
  footprint gate by moving an 8 early; lowering the gate to 256 re-ranks
  validated v4 chains, so it stays measured.
- **New chains from the re-fit were themselves A/B'd** (accept_ab2): 6 ties
  accepted, one free win (v2 7056 f32 [2,3,3,7,7,8] 0.918), two losers fixed
  (7056 f64 via the r7 raise; 32768 f32 via the override above).

## WS3 — width-blind route/table fixes (2026-07-06, bench-results/ws3_*.txt)

New instrument: `--route-ab-dif=<N>` — role-swapped engine A/B (engine_ab_core) of
the DEFAULT plan route vs forced iterative_dif (production DP chain). This is the
only trustworthy route-vs-route gate at N>=8192; the serial kernel-only probes
mislead (see f64 below).

- **vecpass combine W-generalized** (W ∈ {4,8,16}); routing allowlist width-keyed:
  - **f32 W=16 ENABLED** {2592, 8064, 10080, 15120, 20160, 30240, 40320}: RABDIF
    0.60–0.83 robust (40320 0.688 directional). Fixes six sizes where v4 f32 LOST
    to ducc0 (def/ducc 1.13–1.36 in the kernel probe). 4032 excluded (loses
    1.45–1.50 at W=16 despite being a W=8-era allowlist member).
  - **f32 W∈{4,8} allowlist DELETED** ({4032,15120,20160}): v3 RABDIF 1.05–1.09 —
    the WS2 measured-cost dif chains overtook the pre-campaign wins.
  - **f64 NOT enabled — probe-vs-production trap**: W=8 16384 won 0.75 in the
    kernel-only vpass probe (3 consistent runs) but LOSES 1.396 robust in
    production: the per-execute 6M-V allocation + planar deinterleave eat the
    win. Do-not-retry (plan-resident scratch rejected — user directive; it is
    also the fwd+inv L2-blowout trap the per-call alloc exists to avoid). Lesson: kernel
    probes rank kernels, never routes; route enablement gates on --route-ab-dif.
- **fsb_split_for width-keyed, then pruned**: small band (128–768) OFF at W=4
  (forced-dif beats fsb 18–35% at 256–640, ties 128/768; ws3_fsb_small.txt);
  re-validated big at W=16 (fsb 2.3–2.9x better at 256/512/768). Band-B
  32768/65536 entries DELETED at all widths (RABDIF: v2 1.33/1.37 robust, v4
  1.487 robust/1.32 directional, v3 tie — stale vs WS2 chains). W=16 algebra:
  both leaves %16 ⇒ N%256==0 ⇒ 128/384/448/640 impossible at W=16 by
  construction, not mistuned.
- **M=8 flat-leaf codelet gate trait-generalized** (2M < regs, AVX2 provably
  unchanged): v4 f64 codelet cyc −6..7% at 24/40/48/64, f32 flat
  (ws3_codsweep_v4_{pre,post}.txt).
- **dif_beats_codelet: no change** — 0 LOSE rows vs ducc0 across the full list at
  v2+v4 both precisions (ws3_dbc_{v2,v4}.txt).
- Hardcode audit (Explore agent): all remaining sizeof(T)/W literals classified
  measured-gate or structural; the only two stale ones were the fsb table and the
  M=8 exclusion, both fixed above.

## WS4 — v2 acceptance + xsimd sse2 3-of-4 mask (2026-07-06, bench-results/ws4_*.txt)

- **Fork fix shipped, but as hygiene, not perf**: xsimd pin bumped a291174 → 50d20bc
  (fork branch `sse2-masked-load-3of4`): native SSE load/store for the {T,T,T,F}
  f32 mask (movlps+shufps / movlps+movhlps+movss). objdump verdict that motivated
  the downgrade: clang-18 -O3 already SRoA'd the old scalar stack-buffer fallback
  into optimal insertps sequences at both hot sites (dif_pass_last<f32,3> 0x1869f0,
  r=3 cofactor codelets) — the fix guarantees the codegen for GCC/-O2/other
  compilers rather than changing ours. Post-pin v2 binary re-audited: 0 stack refs
  in dif_pass_last<f32,3>.
- **Masked-tail inventory (v2 f32 W=4)**: 3-of-4 fires at dif_passes.hpp last-pass
  IP=3 and codelet r=3 cofactor (both hot loops); 2-of-4 (small_ido IDO=2) was
  already native movlps; f64 W=2 instantiates no masked ops. All yafft masks are
  compile-time (`make_batch_bool_constant`); zero runtime masks.
- **Stale "low-half-mask trap" comments refreshed** (codelet.hpp cofactor gate,
  dif_passes.hpp last-pass): the scalar-fallback half of the rationale is gone
  since 50d20bc; the gates stand on the half-idle-batch cost + measured numbers.
  Re-A/B candidate if the small-N f32 set ever matters.
- **v2 acceptance PASSED**: tier1 26/26 + tier2 14/14 win vs ducc0-v2 (worst 0.855
  f64 4032); pre-campaign worst f64 8192 0.84→0.82, f64 512 0.81→0.69. The f64
  16k–64k structural band is a v2 WIN (0.53–0.62) — the residue is v4-only.
- **ratio_table.txt regenerated** (post-WS4, all 3 ISAs): 77/78 cells win; the one
  flag, v3 f64 8192 fwd 1.11, rechecked 3× = {1.08, 1.00, 1.01} → tie within the
  8192-band swing (pre-campaign 1.00). v4 f64 8192 holds 0.73.
- WS2 calibration and WS3 route audits already covered W=4/2 (validated in those
  workstreams); no new v2 tuning candidates surfaced.

## Open items

- DP currently picks [32,8,32] over measured-best [32,16,16] (the r8 n<=512
  0.78 discount) — resolved by the calibration table, not another patch.
- Old-space measured best [4,4,4,4,4,8] (fused3) at 1.046 says fusion of
  narrow chains is competitive; wide chains cut sweeps 5→3 without tiles.
  Candidate follow-up: fuse (16,16) pairs / prefetch in dif_pass for the
  remaining ~10% on f64 8192.
- f32 W=16 valley set: small_ido_set has no f32-relevant idos beyond {2,4};
  wide chains change which valleys are reachable — recheck after planner rewrite.
