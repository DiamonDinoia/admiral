# Bridging the band-B gap to FFTW — evidence-based plan (2026-07-04)

Scope: pow2 8192–65536 ("band B"), where we lose to FFTW MEASURE by
f64 1.32–1.59x / f32 1.53–1.70x (interleaved `--fftw-ab`, rounds=15).
Elsewhere we win or tie (large-N 262144 ≈ parity; N-D mostly wins; composites won).

## Evidence stack (all same-machine, pinned core 2, per-transform)

1. **perf counters** (ours vs FFTW, f64/f32 8192+32768):
   - instructions 1.08–1.24x ours; cycles 1.22–1.74x; loads EQUAL; stores 1.3–1.5x ours.
   - **Topdown L1: ours 64–69% backend-bound / 30% retiring; FFTW 52–55% / 43–44%.**
     Frontend + bad-spec < 2% both. L2/L3 misses ≈ 0 both. dTLB ≈ 0. NOT memory-,
     frontend-, or ISA-bound. FFTW takes 1.8–2.5x MORE L1 misses and still stalls less:
     they hide latency; we expose it.
   - **FP vector instruction counts**: f64 32768 ours 495.6k vs FFTW 451.1k — and FFTW's
     are all mul/add (no FMA): their real flop count ≈ 1.80M = 27% BELOW naive 5N·lgN
     (genfft DAG). Ours ~10% more instructions carrying more flops each.
2. **perf record**: `dif_pass_fused2<f64,fwd,4,4>` = **58.7% of all cycles** at 32768;
   boundary passes 26.6%; plain radix-4 pass 14.5%. FFTW's heat: t2fv_4/t2fv_8/t1fv_8
   codelet bodies (radix ≤ 8!) + 9.7% memmove (buffering they can afford).
3. **simdref annotate + llvm-mca (alderlake)** on the hot loops:
   - ours fused2: 2 loops × 60 instr/66 uops, 29–30 FP ops each, ZERO YMM spills,
     **4 GPR spill-reloads/iter** (hot: 1.85–1.92% each; 8 input + 8 twiddle stream
     base pointers exceed 15 GPRs). mca: RThr 11.0, **P0/P1/P5 co-saturated at 10/11
     (91%)**; rotation chain vmulpd(4)→vfnmadd(4) = 8-cycle dep.
   - Real-world gap vs static: measured IPC ≈ 1.76 vs mca-ideal 5.4 uops/cyc → the
     kernel runs ~3x below its static port bound: cross-iteration latency exposure
     (L2-resident loads + dep chains) dominates, THEN the port ceiling binds.
   - FFTW t2fv_4: 40 instr, 0 spills, uops/output 6.1 vs ours 8.25; their per-16-output
     RThr 16.4 vs ours 22.0 (1.34x static gap ≈ observed wall-clock gap).
   - FFTW t1fv_8 is WORSE than ours statically (24.2) — their planner still picked it:
     plan-shape + latency hiding beat static beauty.
4. **Experiments already resolved**:
   - Staged radix-16 (single twiddle layer, stack tile): built, correct, **1.27x SLOWER**
     (both prec) — tile store→load latency on an already latency-bound kernel. NO-GO.
     Matches FFTW's own choice of radix ≤ 8 on 16-register AVX.
   - Fusion-aware factor sweep: f64 32768 {4,4,4,8,8,8} wins 7% (override rejected by
     user; must come from the model — see W3). f32 radix-8-heavy regresses. Other ties.

## Verdict

The gap is (a) **exposed latency / insufficient ILP** in the pass kernels (dominant),
(b) **~10% (f64) to ~30% (f32-fsb) more instructions**, mostly twiddle-rotation
arithmetic that genfft's DAG (Linzer-Feig-style FMA factorization + constant folding)
avoids, (c) plan shape chosen by a cost model that predates fusion. It is NOT the ISA,
NOT caches, NOT the wide-vector deficit (FFTW wins with AVX, no FMA).

## Workstreams (in order)

### W1. Kernel ILP / latency (dominant lever) — dif_pass + fused2

**W1a/W1b RESOLVED (2026-07-04).** W1a twiddle packing SHIPPED as a perf-neutral
refactor/enabler; W1b prefetch NO-GO (do not retry).
- W1a merge form (plan-time `dif_fusion_schedule` + packed stream replacing the
  pair's plain tables): asm census f64 (4,4) hot inner loops 74→59 instructions,
  4→0 GPR spill-reloads, 0 YMM spills, mca RThr 11.0→10.3. Whole-plan PMU cycles
  UNCHANGED (32768 −0.04%, 65536 +0.04%, f32 band +0.002%): the kernel is
  latency-bound and the spill-reloads sat in the dependency-stall shadow —
  **GPR spills were NOT a band-B lever**. Shipped anyway (honestly labeled) for:
  single-source-of-truth fusion schedule (W3 prerequisite), one twiddle
  representation (no duplicate L2 footprint), and ~15 uops/iter + GPR headroom
  freed for W1c's U=2 double-pump. Knob-phase A/B history: the earlier duplicate
  storage (packed built alongside plain) cost a real ~1.4% at f64 65536
  (repeatable 4/4 pairwise, +2.8% L2 misses) — eliminated by the merge form.
- W1b `__builtin_prefetch` (input/twiddle/output, L1 hints): no effect at any
  band-B size — data is L2-resident and HW prefetchers already cover the
  streams. NO-GO.
- **PROTOCOL FINDING (gates all future band-B A/B):** cross-BINARY `--fftw-ab`
  m=tsc ratios at band B carry >±30% per-size run-composition luck (allocation
  layout / 4K aliasing at pow2 + wall-clock): the same base/cand pair measured
  ratio-swings of +32%/−37% per size while total PMU cycles were EQUAL to
  ±0.05%, and single-size runs did not reproduce the 4-size runs' ratios.
  Same-binary interleaved A/B (--factors-ab) remains trustworthy; cross-binary
  verdicts MUST gate on pinned `perf stat` total-cycle deltas (FFTW side is
  constant in-process, so Δcycles attributes to our side).
- **W1a. Twiddle-stream packing**: build fused2's two twiddle layers into ONE
  contiguous per-iteration stream at plan time → 1 base pointer instead of ~6;
  removes the 4 GPR spill-reloads/iter and address-dep pressure. Cheap; no numerics.
- **W1b. Software prefetch of the next group** into the FMA shadow (loads run 58%
  port-occupancy; there is headroom). Cheap A/B.
- **W1c. Double-pump U=2** (two independent (l1,ido) groups interleaved with disjoint
  accumulators) on the plain radix-4 pass first, then fused2 if registers allow.
  Long-term home: a `poet::pipelined_for<U>` primitive (U from
  `vector_register_count()` and FMA latency×ports) — upstream ask.
  **RESOLVED NO-GO on AVX2 (2026-07-04, by prior measurement — no prototype
  needed):** dif_pass already carries the U-way path (`dif_pass_unroll<IP>` →
  `poet::static_for`, U = regs/peak_live per the E4-validated model) and it
  correctly picks U=1 on 16 registers (radix-4 peak_live=18 > 16; forcing U=2
  doubles live values → certain YMM spills → the ship gate fails by
  construction; explicit-unroll regressions 14–28% are already on record).
  W1a's PMU-neutral result confirms the mechanism: the t-iterations are already
  independent and the OoO window, not instruction-stream dependency structure,
  is the binding constraint. `poet::pipelined_for<U>` is SHIPPED upstream
  (~/repos/POET branch pipelined-for, 8a90dd1, awaiting user push + CPM pin);
  its consteval pipeline_depth returns the same U=1 here and U=2 on AVX-512
  (32 regs) — that is where this lever activates.
- Gate each via interleaved A/B + the accuracy gate; verify spill/port picture with
  the same objdump→simdref→mca pipeline before/after (asm-analysis workflow §6a).

### W2. Arithmetic: Linzer-Feig dual-select FMA butterfly (op-count lever)

**RESOLVED NO-GO (2026-07-04, paper derivation — do not retry).** The L-F 8→6-op
FMA saving applies ONLY to the DIT pair form `A = a + Wb, B = a − Wb` where both
outputs share one twiddle: the factored terms s1/s2 are reused by A and B. Our DIF
radix-4 arms k∈{1,2,3} each carry a distinct twiddle applied as a standalone rotate
on an already-butterflied value — nothing to share (and radix-4 DIT has the same
standalone pre-multiplies, so restructuring doesn't rescue it). Derived schedule:
L-F dual-select rotate = fma,fma,mul,mul = 4 P0/P1 ops, 8-cycle chain — IDENTICAL
to our mul,fnma,mul,fma. Full per-butterfly column: 29 FP ops = 29, P5 bottleneck
17 add-class ops unchanged, same table footprint (s,t vs c,d planes). Accuracy:
arXiv:2604.00567 (Bergach 2026, read via markitdown) explicitly reports f32/f64
neutral — the dual-select |ratio|≤1 bound only pays at FP16. The distributivity
escape (folding the outer scale s into the butterfly add tree) fails because the 3
arms have 3 distinct scales and arm 0 has none. Verified sources: arXiv PDF
(downloaded to ~/material) + L-F 1993 formulas quoted therein; clean-room upheld.
Classic L-F FMA factorization computes the rotation+butterfly in ~6 FMA-class ops vs
our 8 (2mul+2fma rotation + add/sub), and shortens the mul→fnma chain. Its cot/tan
singularity problem is solved with zero runtime overhead by per-twiddle dual-select
(arXiv:2604.00567, 2026): choose tan- or cot-form per table entry so |ratio| ≤ 1; in
SIMD the per-lane form select becomes a constant operand-swap mask baked into the
plan's twiddle tables (2 cheap P5 blends off the critical path).
- Step 1: paper-derive the exact radix-4 DIF pass schedule (ops + latency count) —
  must beat 29 FP ops / 8-cycle chain before any code is written.
- Step 2: prototype in one pass kernel, A/B + accuracy gate (f32 especially).
- Clean-room from the papers (Linzer-Feig 1993; the 2026 dual-select paper) — do NOT
  transcribe FFTW's generated codelets: public mirrors exist but they are GPL-2+.
  (User decision if GPL contamination is acceptable; clean-room preferred.)

### W3. Planner: sweep-aware exhaustive pow2 chain selection (NO overrides)
Replace the fitted DP constants + r8 hacks for pow2 with exact enumeration of all
{4,8}-compositions of lg N (≤ ~20 chains), scored by: per-pass cost (existing model)
+ sweep count obtained by SIMULATING the driver's real fusion gates. Must reproduce,
with no override table: f64 32768 → {4,4,4,8,8,8} (measured +7%), the existing pow2
override entries (f64/f32 2048, f32 4096) — then DELETE those entries. Re-audit after
W1/W2 land (kernel changes move the constants).

**RESOLVED 2026-07-04 — SHIPPED for the fusion band; override deletion REFUTED
below it.** `enumerate_pow2_dif_plan<T>` (twiddles.hpp) exhaustively scores every
ordered {4,8}-composition for pow2 N ≥ kDifFuseMinN (8192):
`cost = Σ dif_stage_cost + α·N·sweeps`, sweeps counted from `dif_fusion_schedule`
(the driver's REAL gates, single source of truth). α calibration against the
7-point interleaved cyc factor sweep (docs/bridge-evidence/factor-sweep.txt):
every α in **[0.4, 1.45]** reproduces all four binding verdicts (f64 32768 flip,
f64 65536 + f32 8192/16384 non-flips); shipped α = 0.8 (mid-region). Net chain
changes vs the DP: **only 32768, both precisions** → {4,4,4,8,8,8}. Same-binary
interleaved A/B (docs/bridge-evidence/w3-chain-ab-2026-07-04.txt): f64 fwd 0.933 /
rt 0.955 (A/ducc 0.930 — beats ducc0; DP chain was 0.996), f32 fwd 0.939 / rt 0.944
(NEW: the f32 flip was never in the calibration set — independent confirmation).

**Why the 2048/4096 overrides stay (measured, not a concession):** re-A/B'd under
the current fused+merge-form driver 2026-07-04 — f64 2048 {8,8,4,8} fwd 0.944 /
rt 0.938, f32 2048 fwd 0.941 / rt 0.952, f32 4096 {8,8,8,8} fwd 0.904 / rt 0.906 —
robust 5–10% wins, not ties. The α-model provably cannot absorb them: below
kDifFuseMinN no fusion fires, so the sweep term reduces to pass COUNT — invariant
under ordering — while {8,8,4,8}-vs-{4,8,8,8} is a pure ORDER effect; and raising
α far enough (≥2.0) to force f32 4096's {8,8,8,8} flips the calibrated f64 65536
non-flip. These entries encode the cross-pass working-set effect three DO-NOT-RETRY
audits already established as outside additive/per-pass model classes. Maintenance
rule satisfied: entries re-validated on the cost-model change, none tied.

### W4. f32 four-step route re-audit
f32 32768/65536 (fsb-routed) carry +30% instructions. After W1/W2, re-run route-ab;
the fixed dif kernels may reclaim these sizes (then simplify by narrowing the fsb table).

**RESOLVED 2026-07-04 — route re-validated, fsb entries STAY.** `--route-ab`
rounds=15 on the post-W3 binary (new {4,4,4,8,8,8} chain): dif/fsb fwd 1.072 /
rt 1.236 at 32768, fwd 1.030 / rt 1.213 at 65536. The new chain narrowed dif's
fwd deficit (was 1.147 / 1.057 when the entries were added) but four-step still
wins decisively on roundtrip. No table change.

## Dependency asks (user offered to fix upstream)
1. **xsimd**: `fnma`/`fms` dispatch on avxvnni falls back to vxor+vfmadd (12 local
   workarounds `c - a*b`). Fix restores the semantic API.
2. **poet**: `pipelined_for<U>` / register-group unroller emitting U architecturally
   independent body instances (disjoint accumulators), U consteval-derived from
   register count + latency·ports. This is W1c's reusable form.
3. xsimd `batch<std::complex>` stays banned in kernels (slow); we hand-roll planar
   arithmetic — no API needed, W2 adds our own rotate helpers.

## Hot-loop unroll audit (2026-07-04, user-directed): -funroll-loops REMOVED
Full per-symbol tables: docs/bridge-evidence/unroll-audit/ (46 dif-pass symbols,
8 codelet/vecpass, 6 glue). Method: per-TU recompiles at exact production flags
± -funroll-loops, instr/unroll-factor/spill census per hot loop; then alternated
`taskset -c 2 perf stat` cycle A/Bs (flag-on vs flag-off binaries; FFTW is a
shared .so so its work is constant cross-binary).
- EVERY perf-critical kernel is flag-INVARIANT (byte-stable): dif_pass r4/r5/r7/r8,
  all fused2 pairs, fused3, all codelet_apply (fully template-unrolled), col_dif
  r4/r8, f32 transpose pack, recomb butterfly. Their unrolling is already explicit
  (poet::static_for / templates) — codegen does not depend on compiler heuristics.
- The flag ONLY ADDED code where it acted: dif r2/r3 (+15–47%, UF≤2), dif_pass_last
  r2/r3 (+19–111%), vpass_forward f32 (+30% instrs for +13 FMAs), r2c f64 WxW pack
  (+380%), four_step_batched f32 (+11%). Zero spill-count changes anywhere.
- PMU verdict: 5-size basket (1260,2048,2520,4032,5040) and r3-focused basket
  (1260,2520,5040), 3–4× alternated, both precisions — all deltas <1.2% with
  2–7% spreads, signs inconsistent → NEUTRAL. The historical "+5–28% regression
  without the flag" (2026-06 pre-fusion kernels) does NOT reproduce today.
- Action: flag deleted from cmake/CompilerOptions.cmake; static bloat sites fix
  themselves. Remaining spills are the characterized flag-invariant register wall
  (r7 8/11, first-r8 13/8, fused2-8x· 5/1, col_dif r8 33/33) — latency-shadowed
  per W1a; AVX-512-only remedies, no C++ lever.

## Campaign close-out (2026-07-04): causes, mitigations, headline

**MEASURE headline** (docs/bridge-evidence/measure-headline-2026-07-04.txt,
`FFT_BENCH_FFTW_MEASURE=1 --fftw-ab` rounds=15, m=tsc single-binary so per-size
figures carry 5–21% spread; 262144 skipped — MEASURE planning + roundtrip gate
exceeded a 45-min timeout; 1M likewise skipped per plan):
- f64 fwd/rt: 8192 1.63/1.58 · 16384 1.46/1.37 · 32768 1.36/1.30 · 65536 1.31/1.25
- f32 fwd/rt: 8192 1.56/1.55 · 16384 1.49/1.41 · 32768 1.77/1.83 (fsb route) · 65536 1.48/1.46

**Where the gap lives (measured, cross-confirmed):** the band-B kernel is
latency-bound, not port-, ISA-, cache-, or spill-bound. dif_pass_fused2 runs at
IPC ~1.76 vs mca-ideal ~5.4; FFTW stalls less while taking 1.8–2.5× MORE L1
misses (latency hiding via genfft's scheduled codelets); FFTW wins with
SSE2+AVX only and 27% fewer flops than naive 5NlgN. Every kernel-level lever
measured this campaign was neutral or NO-GO: W1a spill-kill (cycle-neutral —
reloads sat in the latency shadow), W1b prefetch (L2-resident + HW prefetchers),
W1c U=2 (16 YMM wall), W2 Linzer-Feig (DIT-only saving), staged r16 (1.27×
slower), -funroll-loops (neutral, deleted).
**What shipped:** W1a packed twiddle stream (enabler: single fusion schedule,
GPR/uop headroom), W3 sweep-aware pow2 enumerator (32768 both prec → 0.93×,
beats ducc0 there), -funroll-loops removal (deterministic codegen), W4 route
re-validation. Overrides at 2048/4096 re-validated as real (5–10%), kept.
**Residual mitigations (all AVX-512-gated or structural):** 32 ZMM unlocks U=2
double-pump (poet::pipelined_for ready upstream) and spill-free r16; below that
the remaining FFTW delta is genfft-style cross-butterfly instruction scheduling
— a code-generator, not a kernel tweak. N-D axes of length 32768 inherit the
new chain unmeasured (col driver has no fusion): spot-check on next N-D pass.

## Non-goals (measured dead ends, do not retry)
Staged/keep-live radix-16+ passes; deeper L1-tile fusion (fused3 >16384); four-step
for f64; explicit unrolling without independent accumulators; genfft t1_8-style small
fused codelets (tied); bigger-radix-as-op-count-fix (latency dominates).

## External references
- OTFFT (Stockham+AVX, claims FFTW-beating pow2): technique pages unreachable today;
  revisit as prior art for W1.
- arXiv:2604.00567 dual-select FMA butterfly (W2 numerics).
- Linzer & Feig 1993, FFTs on FMA architectures (W2 basis).
- FFTW generated-codelet mirrors (GPL — reference only).

## W5. Dependencies & toolchain (added 2026-07-04, user-directed)

### Current pins (cmake/Dependencies.cmake)
- xsimd: CPM GIT_TAG e79f9d36c04b88fd902307c338527b73e77882c9 = **master HEAD as of
  2026-06-27 (verified via gh 2026-07-04; nothing newer)**. Carries the avxvnni FMA
  routing fix (#1368) per the pin comment — the 12 in-tree `c - a*b` fnma workaround
  sites are now equivalent to `xsimd::fnma`, not required. Action: periodic re-check
  `gh api repos/xtensor-stack/xsimd/commits/master --jq .sha`; bump + ctest + band-B
  interleaved A/B guard on every bump.
- poet: CPM GIT_TAG b55580f (DiamonDinoia/poet, default branch `main`).

### D1. poet pipelined_for (owner: us, in user's poet repo)
1. `git clone https://github.com/DiamonDinoia/poet.git ~/repos/poet` (does not exist
   locally yet); branch `pipelined-for` off main.
2. Implement `poet::pipelined_for<U>(first, last, body)`:
   - Semantics: iterate [first,last) round-robin across U *architecturally
     independent* body instances — each instance gets its own accumulator set;
     the expansion interleaves the U instances' operations so the OoO window
     always sees >= U independent dependency chains. body receives
     (index, std::integral_constant<std::size_t, group>) — group tag lets the
     caller keep per-group register state in a static_for-indexed array.
   - Companion consteval: `poet::pipeline_depth(live_regs_per_group,
     latency = 4, ports = 2)` -> U = clamp(available_registers()/live,
     1, latency*ports/chains-ish); pure function of poet::vector_register_count().
   - Tail: remaining (last-first) % U iterations run single-instance.
   - Style: header-only C++20, integral_constant-passing like static_for; unit
     tests in poet's suite (sum-reduction with U accumulators must equal serial).
3. Commit to branch, push (user auth), point fft's CPM at the branch sha. For local
   iteration before push: configure with `-DCPM_poet_SOURCE=$HOME/repos/poet`.
4. Consume in fft: W1c double-pump in dif_pass<4> (then fused2 if regs allow),
   gated A/B per the protocol; ship only on interleaved win.

### D2. xsimd limitations -> fork branch (on demand)
Fork exists: DiamonDinoia/xsimd (default master). Process when W1/W2 prototypes hit
a missing/suboptimal xsimd API: branch `fft-<feature>` on the fork, implement + test
(xsimd's own suite), pin fft's CPM to the fork sha, upstream PR when stable.
Known candidates (none blocking today): none — fnma fix already in pinned master;
planar-complex helpers stay in fft per user direction (hand-rolled batch<T> math).

## Execution state (2026-07-04, session 2)
- E6 done first: memory notes local-resource-inventory (NB: ~/material papers are
  Bailey / FFTW3-2005 / Johnson-Frigo-split-radix / toms1062 — the filenames named
  in the approved plan don't exist) + CAMPAIGN-VERDICT cross-link into the band-B
  diagnosis note.
- E2 RESOLVED NO-GO (see W2 section above) — skip to E3.
- D1 poet pipelined_for DONE: ~/repos/POET branch `pipelined-for` commit 8a90dd1
  (pipelined_for<U> + pipeline_depth consteval; 18/18 tests C++17/20/23;
  clang-tidy clean; user's dirty files preserved exactly, user pushes). fft
  consumes via -DCPM_poet_SOURCE=$HOME/repos/POET until pushed+pinned.
- E1 (W1a/b) RESOLVED — packed twiddle stream shipped as perf-neutral refactor,
  prefetch NO-GO, W1c U=2 NO-GO on AVX2 (see W1 section).
- E4 (W3) RESOLVED SHIPPED — see W3 section. Verification: ctest 114/114; chain
  probe confirms only 32768 flips both prec; interleaved A/B 0.933/0.955 (f64) and
  0.939/0.944 (f32); production instruction count at 32768 drops a stable −0.21%
  (3× alternated perf stat). Protocol note: the process-total PMU gate is BLIND at
  sizes < 64K because the O(N²) accuracy-reference DFT dominates process cycles —
  use the same-binary interleaved A/B (possible here) or instruction counters.
- Next: E5 (W4 f32 four-step route re-audit — NB decomp-report shows f32
  32768/65536 route to four_step_batched with MISMATCH flags vs the dif model
  best; the new 32768 chain took dif to A/ducc 0.715 there, so dif may reclaim) +
  user-directed hot-loop poet/asm audit.

## Replan state (2026-07-04, end of design phase)
- Committed: 997a4a1 (fused policy + f32 fsb route + scale folds), 25998d5 (bench
  gate cap). Working tree: clean except docs/bridge-plan.md (this file).
- In-flight agents: W1a/W1b worker (twiddle-stream packing + prefetch, worktree,
  env knobs FFT_TWPACK/FFT_PREFETCH, must report GPR spill delta); W2 derivation
  agent (Linzer-Feig dual-select radix-4 schedule, GO/NO-GO on op counts).
- Worktree kept as negative result: agent radix-16 branch (staged r16, correct,
  1.27x slower, do not merge).
- Evidence artifacts: preserved in docs/bridge-evidence/ (copied from the session
  4c09dd1e scratchpad; see its README). perf-{ours,fftw}.data stayed in /tmp only
  (binary, re-derivable; the hot-symbol tables are preserved).
- Memory notes: band-b-gap-not-isa-bound, band-b-diagnosis-latency-bound (+ index).
- Decision queue after agents report: (1) merge W1a/b if A/B wins; (2) GO/NO-GO W2
  prototype; (3) D1 poet branch (user auth needed for push); (4) W3 enumerator;
  (5) W4 f32 fsb re-audit; (6) final causes/mitigations report.
