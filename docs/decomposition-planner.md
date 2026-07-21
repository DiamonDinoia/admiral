# Decomposition planner: eliminating the hand-maintained override table

Goal: stop hand-maintaining `measured_dif_factor_plan` overrides; let a principled
DP choose the radix decomposition. This doc grounds the design in how FFTW3 actually
does it (Frigo–Johnson 2005 + FFTW source), and lays out the plan.

## How FFTW chooses a decomposition (source-grounded)

FFTW's planner is a **memoized recursive dynamic program** — structurally identical
to our `build_dif_factor_plan` (rod-cutting over divisors):

- `kernel/planner.c:mkplan` hashes the (problem, precision) to a 128-bit key, looks
  it up in a hash table; on miss, every registered solver proposes a plan, each
  recursively solving and memoizing sub-problems. `cost(n) = min over r|n of
  { cost(stage r) + cost(n/r) }`. The table stores the **winning solver identity**
  (which radix / codelet), not the cost.
- Radix enumeration = "every registered codelet whose radix divides n" (set {2..16,
  32,64} + a generic smallest-prime-factor solver gated off except EXHAUSTIVE).

Two cost oracles plug into that same DP:

| mode | cost per plan | quality |
|---|---|---|
| **ESTIMATE** | weighted op-count: `add + mul + fma(+fma if no HW FMA) + other` (`iestimate_cost`, planner.c:426). `other` = twiddle loads/stores/copies. Pure static. | **20% median / 72% max slower than MEASURE** on the same machine (paper §V-C, Fig 9) |
| **MEASURE/PATIENT** | actually times the plan (`measure_execution_time`, cycle counter), memoizes the winning decomposition into **wisdom** (`htab_blessed` persistent, `htab_unblessed` session) | the reference |

**The paper's explicit verdict (§V-B):** *"there is no longer any clear connection
between operation counts and FFT speed, thanks to the complexity of modern
computers."* FFTW has **no static ido-ordering model** — depth-first ordering and
which radix lands at which level are discovered *empirically*.

⇒ Our `measured_*` overrides are a hand-curated approximation of what FFTW_MEASURE
would discover. A *pure static* DP fundamentally cannot be made override-free
(FFTW proved it). But we can get most of the way cheaply, and automate the rest.

## Our situation

`build_dif_factor_plan<T>` already IS the rod-cutting DP. The 5 overrides split into
two kinds:

1. **ido-ordering** (2520, 1260, 1500, 5040): exist only because `dif_stage_cost`
   makes the odd-radix penalty **ido-independent** (`cost *= 1+0.055*(radix-2)`).
   Since the DP peels the first radix off the *largest* ido, an ido-aware penalty
   would make it pick "expensive-odd-radix-first" on its own. **Subsumable into the
   cost model.**
2. **cache-driven** (2048 `{8,8,4,8}`): the cross-pass cache effect that
   `simdref-grounded-dp-verdict` proved is **structurally non-additive**. A static
   per-stage DP cannot model it (same wall FFTW hit). **Needs measurement.**

## Plan

### Phase 0 — revalidate on xsimd master FIRST
The #1368 avxvnni→fma3 fix (just pinned) changes the *measured* codegen (fewer
port-5 vxor uops). Any cost-model re-tuning must target the new reality, not the old.
Rebuild + ctest + re-baseline the loser set before touching the planner.

### Phase 1 — ido-aware static cost (cheap, principled; removes 4 of 5 overrides)
1. Add an ido-dependent odd-radix term to `dif_stage_cost` (penalty for radix 5/7
   that *decreases* with ido — a radix-5 butterfly batched over ido=504 is far
   cheaper than over ido=56).
2. Verify the DP reproduces {5,3,3,7,8}/{5,3,3,7,4}/{5,5,3,5,4}/{5,4,4,3,3,7} for
   2520/1260/1500/5040 **without** the override entries.
3. Sweep ALL composite sizes: confirm no currently-good decomposition regresses
   (the cost model also drives routing). Gate: ctest + per-size interleaved A/B.
4. Delete the 4 ido-ordering overrides. ~80% of the maintenance burden gone,
   principledly.

### Phase 2 — opt-in MEASURE + wisdom (the "never hand-maintain again" layer)
For the residual cache-driven cases (2048, future N>64K), mirror FFTW exactly:
1. DP enumerates the top-K candidate decompositions (rod-cutting over the codelet
   radix set), not just the argmin.
2. Time each once with the existing interleaved cycle-true harness; cache the
   **winning decomposition** keyed by (N, precision, layout) in a wisdom file
   (persistent = FFTW `htab_blessed`; session = `htab_unblessed`).
3. **Default off** (use the Phase-1 static DP); opt-in flag enables the search.
   This is the principled replacement for the 2048-style override and revisits
   `factor-search-verdict` (which rejected a runtime engine when overrides were
   cheap — the maintenance-elimination goal changes that trade-off).

## Verdict on "pure static DP, no overrides"
Not fully achievable — FFTW's own data (20–72% ESTIMATE penalty) proves op-count DP
can't match measurement. BUT: an **ido-aware static DP removes 4/5 overrides for
near-zero maintenance**, and an **opt-in measure+wisdom layer** handles the residual
cache cases the way FFTW does. End state: no hand-curated per-N table; the static DP
is principled, the wisdom cache is auto-generated.

## 2026-06-30 — Closed-form rebuild REVERTED; per-pass-memory regime hypothesis MEASURED & refuted

A magic-NUMBER-free rebuild of `dif_stage_cost` (derived op-counts + valley + spill +
cache, deleting the override table) was attempted and reported "verified, byte-identical,
ctest 96/96". **The verification was wrong**: the plan-diff used a piped `diff | grep`
that silently returned 0. A robust `paste`+`awk` field compare showed the rebuild
**shredded 195/310 f64 and 149/310 f32 plans** into radix-2/3 chains
(`2048: 4-4-4-4-8 → 2-2-2-2-2-4-4-4`). ctest still passed because any valid factorization
is numerically correct. **Root cause:** real op-counts make radix-2 (2 ops/elem) look ~6×
cheaper than radix-8 (~12.6 w/ spill), so an op-count-minimizing DP maximizes passes —
but FFT cost is per-pass memory traffic, which *minimizes* passes. The rebuild was
**reverted** to the known-good fitted model.

**Then the "re-center on per-pass memory traffic + regime dispatch" hypothesis was
MEASURED** (role-swapped cycle-true A/B, rounds=15×2, pinned core 0) and **refuted**:

- A clean per-pass model (`c_mem·N + w_arith·ops + spill`, all ×N) **provably cannot**
  reproduce the known-good pow2 ladder: `512=4-4-4-8` needs `c_mem < 0.5w+14·sc` while
  `1024=4-4-8-8` needs `c_mem > 0.5w+14·sc` — strictly contradictory.
- "Minimize passes / maximize radix-8" is **wrong** as a rule: measured, `1024` OLD
  `4-4-8-8` beats max-r8 `8-8-8-2` by **1.6× (f64) / 2.2× (f32)**; `128`/`256` similar.
  The OLD DP already produces the good multisets (it avoids trailing-r2).
- Where pow2 *does* have a win it is a **pass-ORDER** choice the per-stage cost cannot
  rank: `2048` `8-8-4-8` is **2× faster than the same multiset `8-8-8-4`** in f32. That is
  the same cross-pass wall the composite overrides hit — it belongs in the residual table.
- A separate pow2 decomposer would therefore either duplicate the DP (f64 is already
  optimal/tie everywhere) or get ordering wrong (2× f32 regression). **Not built.**

**Shipped instead:** the only un-captured measured money — 2 f32 pow2 override entries
(`2048 → 8-8-4-8` +4.6%, `4096 → 8-8-8-8` +8%; f64 ties, omitted; f32 `512`'s `8-8-8` win
is real but production routes 512 f32 to `four_step_batched`, so an override there never
fires). Diagnostic tooling kept: `--cost-audit`, `--factors-ab`, `--decomp-report`.
The robust plan-diff (`paste OLD NEW | awk '$3!=$6'`) is mandatory — the piped `diff` lied.

## 2026-06-30 — Residual ducc0-loser sweep → the f64 odd-radix reorder "silver bullet"

A `--compare` scan (build-fftw, pinned core 0) flagged ~24 size+precision combos still
losing to ducc0. Rather than bank one override per size (which would grow the table past
what the DP decides), a **322-size role-swapped cycle-true A/B sweep** (every iterative_dif
7-smooth size ≥512, reorder-vs-DP, `scratchpad/ab_reorder_out.txt`) was run to find the
*pattern*. It is sharp:

> The DP ranks plans on additive per-pass cost, which sorts **small radices first** and so
> **clusters ≥2 expensive odd radices (5/7/11) at the small-ido tail** — exactly the
> `1<ido<W` scalar valley where odd radices spill. Moving them to the **extremes** (one at
> the largest ido = first pass, one at ido=1 = last pass) fixes it.

Measured outcome of the blanket reorder, by regime:

| regime | win | tie | lose | verdict |
|---|---|---|---|---|
| f64, no radix-8, ≥2 expensive odds | **33** (avg 0.95, to 0.90) | 42 | 3 (≤1.04, all-identical-odd) | **rule** |
| f32, no radix-8, ≥2 expensive odds | 31 | 9 | 33 | DP wins — **scoped out** |
| any, radix-8 present | 3 | 65 | 7 | radix-8 ido=1 pass dominates — **scoped out** |
| single expensive odd | — | — | big (567: 1.43×) | DP already optimal (odd belongs LAST) — **scoped out** |

So the lever is **f64-only, no-radix-8, ≥2-expensive-odd**, shipped as a post-DP reorder
`reorder_odd_radices_to_extremes<T>` in `twiddles.hpp` (a *reorder of the final chain* —
distinct from the per-pass cost term that was provably refuted above and in the memory
do-not-retry notes). It converts ~33 DP-losing f64 composites to wins (many now beat ducc0:
`2058 0.66`, `1575 0.80`, `1250 0.81`, `2352 0.94`, `945 0.92`, `4410 0.75`, `3780 0.89`)
**with no per-size entries**. The 3 marginal all-identical-odd ties (`675/1875/4802`, ≤1.04)
are the accepted collateral.

**Structural-floor vs ducc-loser distinction (re-confirmed, do not chase):** sizes that
carry a **radix-11** (`7920`, `9240`, `7700`, `4620`, `2310`, `15120`) or are
single-large-odd pow2-heavy (`5040`, `1260`, `1500`) lose to FFTW (~1.1–1.4×) for a
**kernel-quality** reason (no genfft-scheduled large-radix / radix-11 codelet, no AVX-512
spill headroom), **not** ordering — a forced best-ordering A/B on them still loses, and
radix-11-first was a tie. Those get no entry.

**Residual hand-tuned table entries that beat the generic rule** (second-order placement
the rule does not capture — both odd copies front, or pow2 ahead of the 3s; role-swapped
A/B, rounds=15×2, ×2-confirmed):

- f64 `2700 [5,5,3,3,3,4]` (vs rule `[5,3,3,3,4,5]`, which only ties DP) — 0.956/0.949, ducc 1.035→0.990
- f64 `7056 [7,4,4,3,3,7]` (vs rule `[7,3,3,4,4,7]`, ties DP) — 0.942/0.950, ducc 1.242→1.169
- f32 `5292 [7,3,3,3,4,7]` (rule is f64-only; f32 wants this split) — 0.750/0.748, ducc 1.435→1.076
- f32 `7560 [5,7,3,3,3,8]` (has radix-8, out of rule scope) — 0.971/0.975, ducc 0.797→0.769

`3780` is produced verbatim by the reorder rule, so its entry was **deleted** (table
shrinks). Method note: `--compare` is noisy on these composites (saw `3780` read 0.898 then
1.045 for the *same* factorization) — the **role-swapped `--factors-ab`** is the only gate
that decides; every LOSE flag here was re-checked and was within-noise (`6300` rule-vs-DP
1.015 = tie, `2352` = robust 0.936 win).
