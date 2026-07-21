# Beat-FFTW/ducc0 audit — 2026-07-02

Full re-benchmark of fft vs ducc0 and FFTW plus perf/asm attribution of every
gap band, on the Ultra 7 155H (Redwood Cove P-cores, **AVX2-only**, W=4 f64 /
W=8 f32). Everything below is grounded in the artifacts under the session job
dir (`sweep/`, `prof/`, `asm/`) and reproducible with the commands given.

## 0. Measurement method (and the trap this audit found)

nanobench's `cpucycles` metric is **TSC-based — constant rate, i.e. wall clock
in disguise**. It does not cancel frequency. Consequences, all observed today:

- Fresh single-size `--compare` runs ride cold turbo for the first
  measurements: pro-fft bias on fwd (measured first), anti-fft on rt
  (measured later). Produced a phantom "our inverse is 2.2× our forward at
  16384+ f64" (steady-state inv/fwd is ~1.03 everywhere).
- Long sweep rows inherit the previous size's cache/thermal state. Produced a
  phantom "f32 16384 loses ducc0 1.28" (interleaved anchor: 0.84 win).

**Rule: `--compare` output is survey-grade. Every win/lose claim must come
from an order-alternating interleaved harness**:

- `--factors-ab=A:B` — ours-vs-ours + per-round ducc0 anchor (existed).
- `--fftw-ab --sizes=...` — fft↔FFTW, added in this audit. Alternates
  measurement order each round, reports median ratio + spread.
  `FFT_BENCH_FFTW_MEASURE=1` flips FFTW planning from ESTIMATE (pessimistic
  bound, default) to MEASURE (FFTW's fair tuned ceiling; used for all 1-D
  numbers below). Also added in this audit, both uncommitted in
  `benchmark/bench_fft.cpp`.

```
FFT_BENCH_FFTW_MEASURE=1 taskset -c 2 ./build-fftw/benchmark/fft_benchmark \
  --fftw-ab --prec=both --rounds=9 --sizes=...
```

FFTW_MEASURE itself has ~5-10% plan variance; rows with spread >10% are
marked noisy but medians are directionally consistent.

## 1. Where we stand vs ducc0 (1-D, interleaved/fresh-verified)

We **win essentially everywhere**, both precisions. Every "loser" the batch
sweep flagged evaporated under trustworthy measurement: f64 16384 → 0.69,
65536 → 0.72, f32 65536 → 0.52, composites 0.77–0.93. Residuals:

- f64 7056 fwd ≈ 1.05 (already has a hand override; second-order)
- f32 1260/1500 rt ≈ 1.0 (ties)

N-D (c2c + r2c, default shape list): broadly win vs ducc0; the known
small-inner losers reproduce (f32 256×16 ≈ 1.30, 60² ≈ 1.39, 1024×64 ≈ 1.26
— see `small-inner-col-dif-register-wall`, B-major rewrite out of scope).
r2c f64 256² loses ≈ 1.20/1.11. FFTW N-D numbers used ESTIMATE (planning cost
prohibitive at large shapes) so they flatter us; fair N-D ceiling is future
work.

## 2. Where we stand vs FFTW MEASURE (1-D, interleaved `--fftw-ab`, rounds=9)

We lose nearly across the board. Median fft/FFTW (fwd/rt), <1 = we win:

| band | f64 | f32 |
|---|---|---|
| N=16–256 (codelet route) | 1.38–2.59 | 1.31–2.48 |
| pow2 512–4096 | 1.25–1.62 | 1.48–1.83 |
| pow2 8192–65536 | **1.36–1.89** | **1.69–1.90** |
| composites 1000–7560 | 1.07–1.38 | 0.98–1.49 |
| 15120/20160 | 1.10–1.64 | 1.14–1.20 |

Only f32 3136 wins (0.976/0.955); f32 5488/1000/2520 ≈ parity. The prior
"f64 4096 / 5040 fwd wins" from single-shot runs were cold-turbo artifacts.

## 3. Gap attribution (perf + asm, per band)

Profiles: `perf record` pinned to a P-core, 5 targets; hot symbols extracted
with objdump and annotated with simdref (arch=alderlake, ADL-P measured
tables). Counters: `perf stat` cycles/instructions/FP-port-dispatch/L1-repl/
L3-miss.

| target | IPC | FP p0/p1/p5 | L1repl/kc | L3miss | regime |
|---|---|---|---|---|---|
| 2048 f64 | 2.44 | 23/42/21% | 245 | 0 | schedule/op-count bound |
| 16384 f32 | 1.99 | 19/32/20% | 255 | 0 | L2 latency + spills |
| 7056 f64 | 2.22 | 32/35/17% | 234 | 0 | GPR-spill slot waste |
| 64 f64 | **3.75** | 19/41/33% | 0 | 0 | pure instruction bloat |
| 65536 f64 | **1.35** | 14/25/16% | 199 | 0 | L2/L3 latency, 7 sweeps |

### 3a. Small N ≤ 256 — instruction bloat (biggest relative gap, 1.4–2.6×)

IPC 3.75 with zero cache traffic: the core is busy executing *unnecessary*
instructions. At N=64 (codelet route):

- `kernel<16>::apply` is **fully scalar** — 72 `vaddsd`/`vmovsd`, zero FMA,
  zero vector ops (182 insns).
- `radix_butterfly_v<4>` lambda clones are **not inlined**: `push %rbp` /
  `sub $8,%rsp` show up at 7% *inside the hot loop* (a call per butterfly
  group), plus YMM stack spills (`vmovapd %ymm9,-0x78(%rsp)` at 11%).
- `codelet_apply<64>` is shuffle-dominated: **160 of 503 insns are
  vperm/vunpck/vinsert** (AoS↔SoA boundary swizzle), p5 33% busy; 25% of its
  cycles sit on the entry `vxorps` (skid onto the first insn = upstream
  call/frontend stall).

FFTW's n1_64 is one flat, fully-vectorized, call-free codelet. This band is
fixable on AVX2 — no genfft port needed (see plan P1).

### 3b. pow2 512–4096 f64 — op-count/schedule (1.25–1.6×)

`dif_pass_first<8>` hot lines are pure FP (vsub/vmul/vfnmadd + some
vinsertf128), FP ports only ~40% loaded, spills minor in first/last. This is
the known genfft fused-codelet gap: FFTW does fewer total ops with better
schedules. Prior verdicts stand (`genfft-fused-codelet-avx2-donotretry`):
porting genfft ties on AVX2 (16 YMM starve radix-16), the lever is
**AVX-512-only** (32 ZMM). `dif_pass<4>`'s stack traffic (below) contributes
~10-15% here too.

### 3c. pow2 8192–65536 — memory scheduling (now the worst absolute band)

IPC 1.35, FP ports <25%, L1repl 199/kc, **zero L3 misses** — LLC-resident but
L2-latency-bound. The DP picks radix-4-heavy chains (65536: 4-4-4-4-4-8-8 =
7 full sweeps of the array; 32768: 7 sweeps where 8-8-8-8-8 is 5). Max-r8
chains measured **slower** anyway (`--factors-ab` 16384: DP wins 0.90/0.62 —
radix-8 passes touch 8 strided streams and thrash worse), so the fix is not
factor order: it's **pass fusion / depth-first blocking** so consecutive
passes reuse L1/L2-resident tiles (what FFTW's recursive DIT and ducc0's
fused multipass do). Also visible here: `dif_pass<8>` spills its 8 stream
pointers and does `add %rdx,0xNN(%rsp)` RMWs on the *stack copies* (9+8+4% of
symbol cycles at 65536).

### 3d. Composites — GPR spill tax in `dif_pass<4>` (1.07–1.5×)

Static census of `dif_pass<double,4>`: 938 insns, **175 spill loads + 103
spill stores + 13 RMWs ≈ 30% of all instructions are stack traffic** (f32
identical shape). At 7056, seven `mov 0xNN(%rsp),%rax` lines alone are ~36%
of the symbol's cycles. Known and previously judged source-unfixable at the
loop level (`dif-pass4-gpr-spill-source-unfixable` — GCC re-canonicalizes);
the structural escape is fewer live base pointers per loop (half-pass split /
B-major restructure), not IV rewriting.

### 3e. Harness note

`__memmove_avx_unaligned_erms` at 7-10% in the profile loops is the harness
input copy; both sides of every A/B pay an equivalent copy, so ratios stand.

## 4. Reconciliation with prior verdicts

- "FFTW is the real ceiling, 1.05–1.8×" — **confirmed and sharpened**: the
  interleaved numbers say 1.1–1.9× (worse than the flattering single-shot
  runs suggested; several previously-claimed wins don't survive).
- Chiplet/genfft on AVX2 — **do-not-retry stands**; nothing in today's asm
  contradicts it (pow2 mid-band is schedule-bound with idle FP ports, and
  register pressure is already the binding constraint). AVX-512 remains the
  designed home for radix-16/32 chiplets.
- "pow2 f64 ≤64K L2-resident not cache-bound" — true ≤8192; at 16384–65536
  the IPC-1.35/latency-bound profile says the multi-sweep structure is now
  the dominant cost. New territory, not covered by that verdict.
- ducc0 loser map — stale: after fresh verification we beat ducc0 everywhere
  in 1-D except f64-7056-fwd (~1.05) and two f32 rt ties.

## 5. Plan of attack (priority order, EV × confidence)

Gate every step on `--fftw-ab` / `--factors-ab` medians (≥9 rounds, pinned,
quiet machine) + ctest + asm audit of the changed symbol. Ship rule: ≥+2%
median on the target band, no >1% regression elsewhere.

- **P1 — small-N codelet quality (N ≤ 256, both prec).** Gap 1.4–2.6×,
  instruction-count-bound, fully AVX2-addressable:
  1. SIMD-ify `kernel<16>` (scalar today — the single clearest defect found).
  2. Force-inline the `radix_butterfly_v<4>` lambda clones (calls in the hot
     loop; `FFT_LAMBDA_ALWAYS_INLINE` exists for exactly this).
  3. Attack `codelet_apply<64>`'s 160-shuffle boundary: reuse the
     transpose-load trick from `dif_pass_last` (overlap tiles) instead of
     the generic swizzle.
  Expected: N≤256 band → ≤1.3×; also lifts every composite that routes
  through codelet cofactors.
- **P2 — large-N pass fusion (N ≥ 8192, both prec).** Gap 1.4–1.9×,
  L2-latency-bound. Fuse two adjacent passes over an L2-sized tile (process
  a block of columns through pass k and k+1 before moving on) — depth-first
  blocking inside `iterative_dif`, no plan-structure change. The 2-pass tile
  keeps ≤ W×radix² values live; start with the 4×4 fusion (radix-16
  two-level) since radix-4 passes dominate these chains. This is the ducc0
  fused-multipass idea, applied only where the data says it pays (≥16384;
  the earlier "big rewrite, small prize" verdict was for f64 1260 — the
  prize here is 1.6–1.9×).
- **P3 — `dif_pass<4>` GPR-spill structure (composites + everywhere).**
  ~30% of the workhorse pass's instructions are stack traffic. Do-not-retry
  on IV rewrites stands; the viable variant is structural: split the 4
  output legs into two half-passes over the same input tile (halves live
  base pointers), or emit the 4-leg loop over a single base + precomputed
  offset table in a register. Verify with the static census (spill count
  must drop; bench second).
- **P4 — N-D small-inner + r2c f64 256² residuals.** Existing verdicts:
  B-major scratch rewrite (out of scope earlier) is the real fix; revisit
  after P2 since tile-fused passes change the economics.
- **P5 — AVX-512 chiplet track.** Unchanged: radix-16/32 fused codelets go
  spill-free only with 32 ZMM. Keep gated on hardware; P1-P3 are all
  ISA-portable and would compound with it.
- **Explicitly not doing** (do-not-retry ledger): genfft t1_8 port on AVX2,
  op-count DP rebuild, runtime factor searcher, max-radix-8 chains at large
  N, scalar-gather lane-over-b, alignas-on-vector.

## 6. Artifacts

- Sweeps: `$CLAUDE_JOB_DIR/tmp/sweep/{1d-measure,nd-c2c,nd-r2c,recheck,fftw-ab}.txt` (+ parsed `table.txt`)
- Profiles: `$CLAUDE_JOB_DIR/tmp/prof/<target>/{perf.data,annotate.txt,hot.txt,pstat.txt}`
- Extracted hot-symbol asm + simdref annotations: `$CLAUDE_JOB_DIR/tmp/asm/`
- Bench additions (uncommitted): `--fftw-ab`, `FFT_BENCH_FFTW_MEASURE` env switch.
