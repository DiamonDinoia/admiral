# Single-threaded FFT architecture — design verdict & redesign frontier

Date: 2026-06-24 (Intel Core Ultra 7 155H, AVX2/16 ymm, AVX-512 absent; pinned
`taskset -c 2`, cycle-true via nanobench). Consolidates a full investigation
campaign (asm-guided, paired A/B, parallel-agent probes) into one design.

Goal framing: the best *single-threaded* architecture — maximal hardware
exploitation, general/reusable/maintainable via zero-cost compile-time
abstractions, dead code pruned.

---

## 0. TL;DR

- The engine is **at the practical FFT roofline (~40% of AVX2 FMA peak)
  almost everywhere, co-located with ducc0** — the missing 60% is shuffles +
  complex-twiddle muls + loads that `5·N·log₂N` doesn't count, not recoverable
  compute headroom.
- It **beats ducc0 broadly** (f32 nearly everywhere; f64 win/parity nearly
  everywhere).
- There is **exactly one structural loss: large power-of-two f64** (N=8192
  **1.43×**, 16384 **1.43×**). Every cheap/medium fix has been measured and
  killed (§3).
- The **one lever the winner actually uses** is *transform-per-lane batched
  multipass* execution — **which we already ship for f32 as the `vecpass`
  route** (and which is precisely why f32 beats ducc0 at 8192). Generalizing it
  to f64 large-pow2 is the only remaining win, and it is a **scoped campaign,
  not a slam-dunk** (§4).
- Independent of perf: a set of **zero-cost generality refactors** (§5) and a
  **verified prune list** (§6).

---

## 1. Absolute baseline (this machine, cycle-true)

Peak = pure AVX2 FMA ceiling (16 flop/cyc f64, 32 f32). Throughput vs the
FFTW `5·N·log₂N` convention. Ratio = ours/ducc0, <1 = we win.

| N | f64 flop/cyc (%pk) | f64 fwd/rt vs ducc0 | f32 flop/cyc (%pk) | f32 fwd/rt vs ducc0 |
|---|---|---|---|---|
| 256 | 8.1 (50%) | 0.54 / 0.53 ✅ | 11 (34%) | 0.40 / 0.32 ✅ |
| 1024 | 7.1 (44%) | 1.01 / 0.82 | 12 (38%) | 0.70 / 0.64 ✅ |
| 4096 | 6.3 (40%) | 0.91 / 0.88 ✅ | 12 (38%) | 0.66 / 0.70 ✅ |
| **8192** | **4.3 (27%)** | **1.43 / 1.40 ❌** | 9.9 (31%) | 0.89 / 0.85 ✅ |
| 2520 | 6.4 (40%) | 0.78 / 0.83 ✅ | 11 (34%) | 0.74 / 0.72 ✅ |
| 1260 | 5.4 (34%) | 1.07 / 0.99 | 8.2 (26%) | 0.99 / 1.03 |
| 1500 | 4.9 (31%) | 1.15 / 1.12 ❌ | 7.7 (24%) | 1.07 / 0.98 |
| 5040 | 5.0 (31%) | 1.13 / 1.01 | 7.9 (25%) | 1.04 / 0.97 |
| 127 (Rader) | 1.0 (6%) | 0.79 / 0.78 ✅ | 1.7 (5%) | 0.51 / 0.46 ✅ |

The smooth-composite trio (1260/1500/5040) f64 is at the instruction-level
frontier (radix-5 at small `ido`); ~10–15% gaps, instruction-bound, small prize
(f32 already pushed to the transpose limit, see `decomposition-frontier.md`).
The headline loss is **8192+ f64**, where throughput craters to 27% peak while
ducc0 holds 40% — a cache cliff.

---

## 2. Current architecture & the two execution strategies

`plan_impl::select_route(N)` dispatches per-N (the correct meta-architecture —
each route is measured-optimal in its regime):

```
codelet (≤64, straight-line SIMD)  · four_step_batched/vecpass (f32 smooth)
iterative_dif (pow2 / 7-/11-smooth) · four_step · rader (p>64) · direct · bluestein
```

The decisive fact is that two **execution strategies** coexist:

| Strategy | Where | SIMD axis | Working set at large N | Cache behavior |
|---|---|---|---|---|
| **breadth-first ido-vectorized** (`iterative_dif`, `dif_driver.hpp`) | f64 + most | vectorize `ido` *within* one transform | streams full N (~256 KB) once **per pass** | **L2-bandwidth-bound at large N** |
| **transform-per-lane batched** (`vecpass`, f32 only) | f32 smooth subset | W *independent* transforms, one per lane | tiny batched tile, **L1-resident** | cache-flat at all N |

This dichotomy *is* the result table: transform-per-lane (vecpass) is why f32
wins at 8192 (0.89×); breadth-first ido-vec is why f64 loses there (1.43×).
**ducc0 uses transform-per-lane everywhere** (`cfft_multipass`, packs `vlen`
sub-transforms into a `Cmplx<doubleN>` SIMD struct, L1-resident `tbuf`) — that,
not any special large-N algorithm, is its entire large-pow2 edge.

There is **no cache blocking / loop-over-cache-levels** in the engine; locality
comes only from codelet sizes + the (breadth-first) pass structure.

---

## 3. The probe graveyard (measured, do-not-retry)

All paired/asm-audited this campaign. None of these is the fix.

| Idea | Verdict | Mechanism of failure |
|---|---|---|
| f64 lane-over-b transpose (trio) | DEAD-BY-ASM, 6–7.5× | ido>W at f64 → baseline already 2 overlapping batches, no scalar waste to recover |
| Naive six-step (AoS `plan_impl` leaves) | DEAD, 2.7× | per-leaf AoS↔SoA repack + 2 full-N transposes bury the L1 gain |
| Fused-SoA six-step (planar, twiddle fused into transpose, L1-resident leaves) | DEAD, 2.7–3.0× | 4 sequential **pure-memory** full-N passes; iterative_dif overlaps its L2 traffic with compute |
| Scratch cache-conflict padding | already shipped | `scratch.hpp:62`, pad-16 when `size%256==0` (ducc0's trick) — banked |
| Naive l1-batched (gather/scatter around `dif_pass`) | DEAD-BY-ASM, 1.6× worse | per-element strided gather/scatter (1 `imul`/elem) costs more than the L2 saving; `dif_pass` at `l1=W` doesn't vectorize |
| Batched/vecpass four-step on f64 (prior) | DEAD, ~1.27× (1260) | O(N) intermediate materialization dominates **at compute-bound sizes** |
| radix-16/32/64 DIF pass; U=2 unroll; HW gather | DEAD (2.9× / 23–51% / loses to transpose) | register-starved / latency / port pressure |

Key nuance: the f64 batched four-step lost at **1260 (compute-bound)**, where
extra movement is pure loss. **8192 is cache-bound** — a different regime, and
the only one where transform-per-lane's L1-residency could pay. No probe this
campaign tested a *true* transform-per-lane multipass at 8192 (they all wrapped
the breadth-first `dif_pass`, which the asm showed is the wrong substrate).

---

## 4. The one redesign lever: generalize transform-per-lane to f64 large-pow2

**What it is.** Make the `vecpass` execution strategy (one independent
transform per SIMD lane, the whole multi-pass DIF chain run in batched SoA,
gather/scatter only at the N-boundary as struct loads — *not* per-element)
the engine for large factorizable N at **both** precisions, not just f32. This
is structurally what ducc0's `cfft_multipass` does and what already makes our
f32 path win.

**Why it could win where everything else lost.** It is the only approach whose
working set is L1-resident *independent of N* (the breadth-first engine's
working set grows with N → the 8192 cliff). At 8192 f64 it replaces ~5
L2-streaming passes (~1.3 MB traffic) with L1-resident batched passes + a
~2N-traffic boundary gather/scatter.

**Why it is NOT a slam-dunk (honest risks):**
1. Prior f64 transform-per-lane attempts **regressed compute-bound sizes**
   (1260, 1.27×) — so it must be **size-gated to large N** (where ido-vec is
   cache-bound), keeping `iterative_dif` for small/medium N. The result is two
   engines selected by size, not one — less "unified," but perf-correct.
2. The boundary gather/scatter (~2N) might still not beat 5 streaming passes;
   needs a real prototype (the cheap probes don't settle it).
3. It is a **substantial rewrite**: new twiddle layout, new batched pass loop,
   boundary swizzle — equivalent to productionizing `vecpass` for arbitrary
   radix + f64. Scope it as its own campaign.

**Prize.** The large-pow2 f64 class (8192/16384/32768…) — common in
convolution/imaging — from ~1.43× to ~parity. Plus a **generality win**: f32
`vecpass` and the new f64 path become one templated transform-per-lane engine
(serves the "general & reused" goal).

**Recommendation.** This is the only remaining perf lever and the one the
winner uses. Pursue it as a **scoped, size-gated campaign** with a true
transform-per-lane prototype at 8192 f64 as the go/no-go gate — *before* any
engine integration. Do **not** rewrite the whole DIF chain speculatively: the
engine already wins or ties ducc0 on the vast majority of sizes, and a botched
generalization risks the f64 small/medium wins.

### 4a. Go/no-go gate — RESOLVED: **NO-GO** (2026-06-24)

Probe: `report/probe_recursive_tpl.cpp` (build: `report/build_probe_recursive_tpl.sh`).
It measures the *two-factor* transform-per-lane (the shipped `vecpass` machinery,
which is `V`-templated, instantiated for f64 / W=4) at 8192 and 16384 f64 against
`iterative_dif` and ducc0. Indicative cycles (`taskset -c 2`, nanobench):

| N | TPL (vecpass\<f64\>) vs iterative_dif | TPL vs ducc0 | iterative_dif vs ducc0 |
|---|---|---|---|
| 8192  | **1.135×** (slower) | 1.473× | 1.298× |
| 16384 | **1.188×** (slower) | 1.643× | 1.383× |

Both correct (L2 vs ducc0 < 4e-16). The two-factor TPL is *worse* than the
shipped engine — its Phase-1/3 (pack + per-lane twist + cross-lane DFT) overhead
sits on top of an already-L2-bound 256 KB Phase-2.

**Why this settles it without building the recursive Phase-2.** The *identical*
structure **wins at f32** (8192 = 0.89×). The only differences at f64 are W=4
(half the lanes) and a 2× working set. The depth-first recursive Phase-2 attacks
only the *working-set* half (L2→L1); it cannot recover the *lane-throughput* half
— and `iterative_dif` is already the L2-bound 256 KB engine that ducc0 beats by
L1-residency. Every prior depth-first f64 restructure lost for the same reason
the recursive multipass would share: at W=4 each memory-movement op processes
only 4 doubles, so the gather/scatter/restructure cost exceeds the L2 saving
(six-step L1-leaves 2.7×, l1-batched 1.6×, three TPL four-step attempts ~1.27–
1.29×, see §3 and [[f64-batched-fourstep-loses]]). Best case for the multi-hour
recursive build is **parity** with ducc0 on one size class — not a win — against
a strong prior of ~1.3×. Negative expected value.

**Verdict.** Concede large-pow2 f64 (8192/16384/32768…) as the **single
documented structural loss** (~1.3–1.4× ducc0). The engine is at the AVX2
roofline and wins or ties ducc0 everywhere else. Do not pursue the recursive
transform-per-lane campaign (§4 / 3c) unless an AVX-512 target (W=8 f64) changes
the lane-throughput arithmetic. The probe stays in `report/` as the record.

---

## 5. Generality / "negative abstraction" refactors (zero runtime cost)

The codebase is already close to the ideal (the V-templated `dif_butterfly`,
`cofactor_batch_width`, `dif_pass_unroll`, `aos_deinterleave` are exemplary
hardware-derived consteval). Queued zero-cost cleanups (verify-asm where noted):

1. **Single source of truth for the radix set** — `dif_radix_set`
   (`dif_driver.hpp`) vs `dif_candidate_radices` (`twiddles.hpp`) vs the
   `vpass_dispatch` switch (`vecpass.hpp`): derive all from one
   `integer_sequence`; route the switch through `poet::dispatch`. Zero codegen.
2. **Derive the `256` conflict threshold** in `scratch.hpp:span_stride` as
   `4096/sizeof(T)` (L1 set period) — makes the cache linkage explicit; template
   `span_stride` on `T`. Zero codegen.
3. **`if constexpr (sizeof(T)==4)`** throughout `select_route` so f64 dead
   branches strip at compile time (the predicates are already `constexpr<T>`).
4. **Collapse `rader_apply`/`rader_apply_batched` and `bfly_chunk`/`_batched`**
   into one `V`-generic body (`V=T` scalar, `V=batch<T>` SIMD). VERIFY-ASM that
   `V=T` collapses to scalar.
5. **`dif_pass`/`dif_pass_first` shared body** behind Src/Dst policy tags
   (`if constexpr`, inner lambda stays `always_inline`). VERIFY-ASM; keep
   `dif_pass_last` separate (its ido==1 transpose-load path is distinct).
6. **Document, don't derive:** `SBO_MAX=4096` (measured policy constant) and
   `kNoinlineMinSize=16` (empirical) — pin with `static_assert` + comment, keep
   the value. Cache *size* is the only genuinely runtime quantity; width &
   register count are already consteval-portable.

Sequencing: defer #1/#5 until the §4 go/no-go, so the engine is refactored once.
Per owner constraint, all machine constants must derive from
`xsimd::batch<T>::size` / `poet::vector_register_count()` / a cache query, never
hardcoded. A portable global-once cache-detection util (`sysconf` /
`sysctlbyname` / `GetLogicalProcessorInformationEx`) is only worth building if
§4 ships (it's the sole runtime consumer); otherwise YAGNI.

---

## 6. Prune list (verified; pending owner approval)

Safe (not in any build, findings committed):
- **~170 MB regenerable perf captures** — `report/perf.*.data`, `report/n1000/`.
- **Stale standalone probes** — `spill_probe.cpp` (won't compile; includes the
  deleted `fft_kernels.h`), `bfly_probe.cpp`, `bfly_codegen_probe.cpp`,
  `vpass_proto.cpp`, `vpass_kernel.hpp`, `probe_move.cpp`; superseded
  `PHASE0_BASELINE.md`, `PHASE1_FINDING.md`, `PHASE_FOURSTEP.md`,
  `PHASE_CODELET_BASELINE.md` (~850 LOC).
- This campaign's own probes once mined: `probe_sixstep.cpp`, `probe_pad.cpp`,
  `probe_laneb_f64.cpp`, `probe_l1batched.cpp`.

**KEEP (agent was wrong / load-bearing):** the **`direct` route** — live
fallback for **reduced-catalog builds** (`FFT_CODELET_MAX`); `probe_codelet.cpp`
(reusable asm tool); `PHASE1_OPTIMALITY.md` (U=2 do-not-retry evidence base);
the f32 N=60 routing override.
