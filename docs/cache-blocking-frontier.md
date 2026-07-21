# cache-blocking / working-set frontier — ledger

Question (2026-06-26): the FFTW gap on large pow2 f64 (4096/8192) was believed
memory-bound — can we close it with cache-blocking, and adapt the FLAME/BLIS
analytical block-size methodology, parametric on HW? Verdict: **measured No for
the target range** — those sizes are L2-resident, not cache-bound.

## Method

`perf` working-set sweep, P-core pinned (`taskset -c 0`), tight-loop driver
(plan once, `execute_forward` in a loop), per-transform-normalized counters
(`scripts in $CLAUDE_JOB_DIR/tmp`: `measure_cache.sh`, `probe_factors.cpp`).
155H P-core geometry (`lscpu -C`): L1d 48 KB / 12-way / 64 sets, L2 2 MB / 16-way,
L3 24 MB, 64 B lines.

## Measured (forward, f64)

| N | 2^k | cyc/(N·log₂N) | IPC | LLCmiss/xf | stallL1d% | stallL2% |
|------|------|------|------|------|------|------|
| 4096 | 12 even | 0.98 | 2.33 | 0.0 | 3.8 | 0 |
| **8192** | **13 odd** | **1.34** | **1.79** | 0.2 | 6.6 | 0 |
| 16384 | 14 even | 1.00 | 2.24 | 0.7 | 5.1 | 0 |
| **32768** | **15 odd** | **1.39** | **1.70** | 3.0 | 6.3 | 0 |
| 65536 | 16 even | 1.20 | 1.84 | 4.3 | 11.3 | 0 |
| 131072 | 17 odd | 1.70 | 1.37 | **56.5** | 11.6 | 0 |

## Findings

1. **L2-resident, not DRAM-bound.** LLC-miss/xf ≈ 0 through 65536 (2 MB);
   it only jumps at 131072 (4 MB > L2). The real DRAM cliff is ~128K, far above
   the 4-8K campaign focus.
2. **Low memory stalls.** `stalls_l1d_miss` 6.6 % @8192, `stalls_l2_miss` 0 %.
   The old "8192 = L2 cliff / 40 % memory-bound" framing is imprecise — the wall
   is not memory traffic.
3. **The real signal is the 2^odd-power penalty.** cyc/(N·log₂N) spikes at odd
   powers (8192/32768/131072, +30-40 %, IPC drops) and is cheap at even powers.
   Factorizations (`probe_factors.cpp`): even = `4^a·8²` (two wide radix-8),
   odd = `4^a·8` (one radix-8 + more narrow radix-4 passes):

   | N | factorization | passes |
   |---|---|---|
   | 4096 | `4,4,4,8,8` | 5 |
   | 8192 | `4,4,4,4,4,8` | 6 |
   | 16384 | `4,4,4,4,8,8` | 6 |

4. **Both cheap levers already spent.** DP factorization is at the optimum
   (radix-8 chains LOSE 8192: 1.158 vs 1.021, `simdref-grounded-dp-verdict`);
   re/im inter-span cache conflict is already padded (`scratch.hpp` span_stride:
   `n % 256 == 0 → n+16` for f64, ducc0's trick — historically fixed 4096).

## Verdict

Cache-blocking / fused-multipass / four-step is **not the lever** for ≤64K pow2
f64 — they are L2-resident. The 8192 gap is the intrinsic **2^odd radix-schedule /
W=4 lane wall** (radix-4-vs-8 spill-vs-width tradeoff), confirming the existing
NO-GO-unless-AVX-512 verdict but correcting its mechanism label. The FFTW edge
here is genfft straight-line codelets + planner, designed out of this engine.

## Where cache-blocking *would* pay, and the BLIS-parametric model

- **N > 64K** (past the DRAM cliff): depth-first / fused-multipass reduces L1↔L2↔
  DRAM traffic. Out of the campaign's target range; revisit only if large-N
  matters for the workload.
- **BLIS analytical model (the "parametric on HW" ask):** there is *no published
  BLIS→FFT analytical block-size mapping* — it would be novel. Its value here is
  **code-quality + portability**, not perf: replace the hand-tuned `×1.16` radix-8
  cache-proxy penalty and the per-size measured overrides with a closed-form
  derived from `{S_L1, W_L1, line, SIMD width, N_reg}`, so the engine retunes
  itself for AVX-512 (where the 2^odd wall *also* lifts: 32 ZMM kills radix-8
  spills, W=8 halves the lane wall). FFT analogues of the BLIS formulas:

  ```
  fusion factor   F_max = floor( log_r( S_L1 / (r · W · S_data · 4) ) )
  resident tile   T     ≤ S_L1 / (2 · S_data · (1 + F))
  reg-tile bound  4·r ≤ N_reg      (r=4 fits 16 YMM; r=8 spills on AVX2)
  ```
  For 155H (L1 48 KB, W=4 f64): r=4 → F_max=3 (64-pt tiles, = FFTW base case);
  r=8 → F_max=1.

## Recommendation

No fused-multipass rewrite for the target range (refuted). If pursuing the
BLIS-parametric model, scope it as a portability/maintainability refactor of the
DP cost model, gated on AVX-512 hardware for the perf claim — not an AVX2 win.
