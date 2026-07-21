# 1D optimal N-decomposition over the ≤64 codelet catalog — frontier & verdicts

Date: 2026-06-22 (Intel Core Ultra 7 155H, AVX2/16 ymm, pinned `taskset -c 0` P-core,
cycle-true via nanobench cpucycles). Builds on the codelet campaign
(`docs/codelet-optimization-frontier.md`, catalog 0/63 vs ducc0).

Goal: a throughput-aware decomposition that, for any N, uses the min-measured-cost
composition of catalog codelets — enabled by a fast inter-factor movement — **or** a
documented verdict that small-radix `iterative_dif` is already optimal.

---

## Phase 1 — Gather/scatter verdict (the enabling-tech microbenchmark)

**Probe:** `report/probe_move.cpp` (standalone, production flags minus LTO, nanobench
cpucycles, `taskset -c 0`). Isolates the two decision-relevant data movements of a
batched big-codelet four-step step, holding all DFT work out of the measurement:

- **(A) inter-factor transpose-scatter** — the inner pass emits M=N2 planar batches of
  width W (lane = cofactor copy / n1); they must land in `G` as W rows of length N2
  (`G[l*N2 + k2]`), an N2×W → W×N2 planar transpose. This is the exact
  `four_step_batched_ct` bottleneck (`four_step.hpp:229-235`, scalar store loop) and the
  same pattern as the shipped cofactor scatter (`codelet.hpp:511`).
- **(B) strided column gather-load** — the classic four-step leaf reads a stride-N1
  column `x[n2*N1 + n1]` (`four_step_execute:150`).

Variants: **scalar** / **hardware gather** (`_mm256_i32gather_{pd,ps}`) / **in-register
transpose** (`xsimd::transpose` = vunpck/vperm) / **zip** (note: for the planar transpose,
`zip_lo/zip_hi` *are* the W=2 transpose and the building block of `xsimd::transpose` for
W=4/8 — so the "zip" and "transpose" variants converge; `aos_deinterleave` is the AoS↔SoA
boundary op, not this inter-factor transpose, so it is not a separate competitor here).

### (A) Transpose-scatter (cyc, lower = better)

| | N2=16 | 32 | 64 | 128 |
|---|--:|--:|--:|--:|
| **f64 W=4** scalar | 87.4 | 172.8 | 343.5 | 685.0 |
| transpose | **42.7** | **85.4** | **170.9** | **341.5** |
| hw gather | 76.2 | 140.5 | 276.6 | 555.6 |
| scalar / transpose | 2.05× | 2.02× | 2.01× | 2.01× |
| hwgather / transpose | 1.79× | 1.65× | 1.62× | 1.63× |
| **f32 W=8** scalar | 172.8 | 343.5 | 685.0 | 1368.0 |
| transpose | **44.5** | **87.4** | **175.6** | **353.4** |
| hw gather | 131.5 | 254.3 | 497.4 | 985.0 |
| scalar / transpose | 3.88× | 3.93× | 3.90× | 3.87× |
| hwgather / transpose | 2.95× | 2.91× | 2.83× | 2.79× |

The in-register transpose **wins decisively**: 2.0× (f64) / 3.9× (f32) faster than the
committed scalar store loop, and 1.6× (f64) / 2.8× (f32) faster than hardware gather.
**Hardware gather loses to transpose at every size/precision.**

### (B) Strided column gather-load (cyc; scal/gath < 1 ⇒ scalar faster)

f64 W=4 and f32 W=8, swept N∈{16,32,64} × stride∈{16,32,64}: `scal/gath` ratio ranges
**0.71–1.01** — scalar strided loads are **as fast or faster** than hardware gather
(gather up to 1.4× slower); the two tie only at the N=16 latency floor (~24 cyc).
Hardware gather for the strided load is never worth it.

### Codegen audit (objdump, `-fno-lto`)

- `scatter_gather_*` / `gload_gather_*`: emit `vgatherdpd` (f64) / `vgatherdps` (f32) —
  hardware gather, as claimed.
- `scatter_transpose<double>`: 12× `vunpcklpd`/`vunpckhpd` + 12× `vperm2f128` +
  12× `vinsertf128`, **zero** `vgather` — the in-register transpose, as claimed.

### Verdict (the definitive answer)

**Hardware `vgather`/scatter is never worth it on this AVX2 uarch** — it loses to the
in-register transpose for scatter (1.6–2.9×) and to scalar strided loads for the gather
(up to 1.4×). This confirms the strong prior (`xsimd-complex-slow`,
`aos-soa-hand-swizzle`). **The winning inter-factor movement primitive is the tiled
in-register `xsimd::transpose`.** It is 2–4× cheaper than the scalar store loop the
committed `four_step_batched_ct` uses → swapping it in is the Phase-2/3 enabler. (In the
real four-step the inner-pass output is already in registers, so the transpose is even
cheaper in context than this isolated probe; the scalar loop forces a store/scalar-reload.)

Caveat carried into Phase 2: the full-tile transpose needs `N2 % W == 0`; a partial tile
falls back to a scalar remainder. Combined with `four_step_batched_ct`'s `N1%W==0 &&
N2%W==0` requirement, the batched form only applies when both factors are W-multiples —
which (foreshadowing) are the *smooth* sizes where `iterative_dif` is already pass-optimal.

---

## Phase 0 — Harness (enabler)

`benchmark/bench_fft.cpp --fsb`: for curated W-divisible 2-factor splits (N1%W==0 &&
N2%W==0), runs `four_step_batched_ct<N1,N2>` (the transpose-scatter form) as a drop-in
(deinterleave + batched four-step + reinterleave) and reports, within one process
(drift-free), `fsb/ducc`, `def/ducc` (default plan_impl), and **`fsb/def`** — the
production-relevant ratio (def = current `iterative_dif`). `def/ducc` reproduces the known
large-N numbers (1024 f64 ≈0.95, 2048 ≈1.0, 4096 ≈0.89), confirming calibration. The
helper SKIPs splits whose factors are not a multiple of that precision's W.

## Phase 2 — Empirical optimum (the decomposition verdict)

**f64: batched four-step loses everywhere.** `fsb/def` = 1.12–1.65 across 128…4096 and
all smooth composites (720/1296/2304/3600/1600). The transpose scatter (Phase 1) narrowed
the old scalar-four-step gap (2–4×, `report/PHASE_FOURSTEP.md`) to 1.12–1.65×, but the
default `iterative_dif` still wins. **Do-not-retry batched four-step for f64.** (W=4 packs
only 4 leaf-columns/register — half the f32 lane density — and f64's 2× working set hits
cache pressure sooner.)

**f32: a real win class, N ≤ 768.** Best measured split per N (`fsb/def` < 1 = beats
current production), drift-free `--fsb --reps=20`:

| N | best split | fsb/def | fsb/ducc | def/ducc |
|---|---|--:|--:|--:|
| 128 | 8×16 | 0.925 | 0.377 | 0.408 |
| 256 | 16×16 | 0.914 | 0.366 | 0.400 |
| 384 | 16×24 | 0.914 | 0.581 | 0.635 |
| **448** | **8×56** | **0.661** | 0.647 | 0.978 |
| 512 | 16×32 | 0.921 | 0.654 | 0.710 |
| 640 | 16×40 | 0.989 | 0.753 | 0.762 |
| 768 | 16×48 | 0.971 | 0.735 | 0.757 |

f32 N≥1024 loses (`fsb/def` 1.23–1.34) — cache pressure + iterative_dif already strong.
320 (1.007) and 576 (1.004) are marginal losses, excluded. **N=448 is the standout
(+34%)**: 448 = 2⁶·7 makes `iterative_dif` ~parity with ducc0 (def/ducc 0.978), while the
8×56 batched four-step beats ducc0 1.5×.

**Why (ducc0-informed):** ducc0 vectorizes only across its `l1>1` multi-transform
dimension; for a single 1D transform the early/late passes (`l1` or `ido` small)
underutilize SIMD lanes — the same limitation as our `iterative_dif`. The batched
four-step instead keeps W full leaf-columns batched through the *entire* leaf transform
(`kernel_batched`, ~100% lane use), so for f32 (W=8) on small/medium N two big-codelet
passes beat the radix chain. The radix-16/32/64 *pass* approach was tried and failed
(loses the leaf SIMD batch, memory `chiplet-dif-pass`); the four-step is the correct
vehicle for big-codelet composition because it preserves the leaf batch.

Split choice: N1=16 is best for 256/384/512/640/768; 8×16 for 128; 8×56 for 448. Shipped
as a measured split table (`fsb_split_for<T>`), not a model — the win class is small,
specific, and a model would risk mispredicting the split (which matters: e.g. 384 16×24
0.914 vs 8×48 0.927).

---

## Large-N scratch allocation: investigation + allocator-template verdict

The iterative-DIF / direct-DFT routes allocate 4 planar scratch spans per execute.
`N <= SBO_MAX` (4096) uses an `alignas(64)` stack buffer (alloc-free); `N > 4096`
uses a heap `std::vector`. Per-execute heap `resize` pays a `malloc` + a value-init
memset each call (the ~13% tail on 5040/7560/8192).

**What was tried (measured, per-size-interleaved A/B, cyc, taskset -c 0):**

- **Plan-owned aligned buffer** (allocate once in the ctor, 64-byte aligned, reuse):
  forward win of **-12.5% (5040), -15.7% (7560), -13.3% (8192)** in f64 (similar f32).
  BUT a forward+inverse **roundtrip regressed** (5040 rt +9%, 7560 rt +16%): two
  *separate* plan objects each keep a resident ~240 KB buffer, so the rt working set
  ~doubles (~600 KB) and blows L2. The old per-call `malloc` recycled one address
  across fwd+inv (~360 KB resident), so baseline rt was actually cache-friendlier.
- **Shared `thread_local` buffer** (one per thread, grown once): fixes the footprint
  doubling (fwd+inv share it) AND keeps the forward win, and `execute_iterative_dif`
  is a verified non-reentrant leaf (top-level / sequential N-D sub-plans only;
  four-step/rader/bluestein use their own scratch) so it is safe. **Rejected by
  design preference**: hidden mutable global state + thread-lifetime retention.
- **Plain `std::vector` (16-byte aligned) for <=4096 too** (unifying the path):
  **regressed 2520/4096 by 16-23%** -- losing the stack buffer's `alignas(64)` and
  stack locality. Confirms SIMD scratch alignment is load-bearing here.

**Shipped:** keep the clean master scratch (stack SBO <=4096, per-execute heap >4096)
and add an **`Alloc` template parameter** to `plan_impl` / `soa_scratch` /
`forward_plan` / `inverse_plan` (default `std::allocator`). The default is
behaviour-identical to master (A/B parity, +-2-3% noise, both precisions); a user who
cares about the large-N per-call `malloc` supplies a pooling/arena allocator
explicitly. Verified: `--verify` PASS; `report/probe_alloc.cpp` shows a custom
allocator is actually routed through (2 allocs for fwd+inv) with correct round-trip
(L2 = 5e-16). Do-not-retry: thread_local scratch (design), plain-vector unification
(alignment regression).

**Still-open ducc0 losers** (this detour did not close them -- they are asm-bound, not
alloc-bound): 8192 (~1.5x, pure 2^13 -- needs radix-8 passes extended past 512 + the
critbuf cache-conflict copy, both ducc0 tricks), 5040 (~1.4x), 1260/1500 (smooth
composites already alloc-free on the stack path; f64 four-step is do-not-retry).

---

## Vecpass-style small-ido pass: transpose lane-over-b (SHIPPED, f32)

Date: 2026-06-22. Targets the measured odd-radix smooth-composite losers
(N=1260/1500/5040) whose hot pass is `dif_pass<5>` running at **ido<W** (the
factorizer emits `r5 ido=7 l1=36` for 1260, `r5 ido=5 l1=60` for 1500, `r5 ido=7
l1=144` for 5040). At ido < W=8 the standard a-vectorized loop fills **zero** SIMD
lanes → the radix-5 butterfly runs fully scalar (perf: 30-40% of runtime). f64 is a
*different* problem (W=4, ido=5/7 > W → the a-loop already vectorizes with a small
scalar tail), so this is **f32-only**.

**Fix (`dif_pass_small_ido`, `dif_passes.hpp`):** vectorize over the group dim `b`
instead of `a`. Pack W consecutive b into lanes; the output twiddle `W_N^{k l1 a}`
depends only on `a` so it broadcasts. Movement is the whole game here —

- **scalar gather/scatter** (the obvious lane-over-b): **net-negative, abandoned.**
  Same-session A/B: f32 1260 −5%/1500 −9% but 5040 **+7%** (l1=144 → memory-bound);
  f64 **+19%/+41%** (replaces good a-loop with strided scalar). It only converts
  scalar→SIMD butterflies without cutting memory traffic. This is the Phase-1 verdict
  (scalar loses ~3.9× f32 to transpose) hitting in production.
- **in-register transpose** (shipped): masked-load W b-rows of the contiguous ido-run
  → `xsimd::transpose` → per-`a` batches over b → full-width radix-5 butterfly →
  transpose back → masked-store. Compile-time `IDO` mask (exact, no over-read into the
  neighbour span); reads input span, writes output span, **no staging buffer**. The
  `l1%W` tail is folded into the transpose path (zero-padded partial b-block), not left
  scalar (asm showed the scalar tail cost ~as much as the whole vectorized body).

**Measured (155H, taskset -c 0, cyc, 13-round paired A/B vs master; raw vs ducc0):**

| N (f32) | cand/base fwd | vs ducc0 (master → shipped) |
|---|--:|--|
| 1260 | **0.778** (−22%) | 1.32 → **1.04** (near parity) |
| 1500 | **0.786** (−21%) | 1.42 → **1.10** |
| 5040 | **0.814** (−19%) | 1.31 → **1.17** |

No regression elsewhere (7560/512/720/2520/4096/1024/8192 within ±1% noise). **Gated
to `IP==5, ido∈{5,7}`** — the only (radix,ido) the factorizer emits for these sizes;
instantiating the unused odd radices/idos bloated the TU and perturbed other sizes'
code layout (measured ~3-5% on 7560/512 before trimming). Everything else falls
through to the unchanged a-loop.

**asm audit (objdump `dif_pass_small_ido<float,true,5,5>`):** confirmed `vunpcklps`/
`vunpckhps`/`vperm2f128`/`vinsertf128` (the transpose), `vmaskmovps` (masked
load/store), full-width `vfmadd231ps`/`vmulps` (SIMD butterfly). Residual gap to ducc0
= staging spills (IP·ido batches, ido=7 → 35×4 batch slots) + ducc0's hand-tuned
vecpass. **Verdict: substantial narrowing (1.3-1.4× → 1.04-1.17×), correct, no
regression — but does NOT fully beat ducc0.** Closing the last 4-17% needs ducc0's
actual vecpass (sub-FFT-packing with a transposed buffer layout), a larger rewrite.
f64 1260/1500/5040 remain losers (no ido<W lever).

<!-- Phase 3 ship + final A/B appended below. -->
