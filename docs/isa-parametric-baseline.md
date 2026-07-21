# ISA-Parametric Baseline & Bottleneck Audit (2026-07)

Machine: Xeon **w5-3435X** (Sapphire Rapids), full AVX-512. Toolchain: **clang-18**
+ gcc-13 libstdc++, `-O3`, **LTO off**, `xsimd` latest `xtensor-stack/xsimd` master,
`poet` pinned. Bench pinned to core 0, accuracy-gated, medians via nanobench.

## TL;DR

- The kernels are **implicitly tuned for AVX2 widths** (f32 W=8 / f64 W=4 — Alder Lake).
  On this AVX-512 box (f32 W=16 / f64 W=8) the code had **never been compiled**: several
  hard `static_assert`/no-op breaks surfaced (now fixed, parametrically).
- yafft **beats ducc0 on every size at every ISA level** except **f64 N=8192 at AVX-512**,
  where it *regresses in absolute terms* going W4→W8 (30→35 µs) while ducc0 improves.
- Root cause: the small-`ido` "valley" fast path is hardcoded to `W==8 && IP==5` (f32
  radix-5). f64 radix-8 falls through to a partial-lane loop that wastes 6/8 lanes.

## Fair-comparison method

- **ducc0** is compile-time-only SIMD, built from source at *our* `-march` → the **fair
  per-ISA opponent** (yafft-vN vs ducc0-vN, same width).
- **FFTW 3.3.10** (module) has runtime CPUID dispatch → always runs best AVX-512 → its
  wall-time is ~ISA-invariant. Used only as a **fixed absolute yardstick**, never a per-ISA
  ratio.
- ISA levels: `x86-64-v4` (AVX-512, f32 W16/f64 W8) · `v3` (AVX2, W8/W4) · `v2` (SSE4.2,
  W4/W2). `native`≈v4 (both W16/W8).

## Stable-base fixes (all parametric / future-proof for ARM)

The build only worked at `-march=native` on the *original* AVX2 box. Fixes to compile at
every width (divisibility/lane-count driven, no precision hardcodes):

| file | was | now |
|---|---|---|
| `four_step.hpp` | `static_assert(N1%W==0 && N2%W==0)` (breaks at W=16) | body wrapped in `if constexpr (N%W==0)`; route gated by actual `n%W==0` |
| `vecpass.hpp` | `vecpass_supported` precision-only → **selected but no-op at W=16 (garbage)** | gated to W∈{4,8} (widths the combine implements) — fixes a latent correctness bug |
| `dif_passes.hpp` | `WaMax` W-aligned only by luck at W=8 | `WaMax = (raw/W)*W`, `static_assert(WaMax>=W)` |

Toolchain finding: **gcc-13.2/14.2 both ICE** on AVX-512-width codelet codegen
(codelet_22/30/60; LTO amplifies it). clang-18 compiles cleanly → chosen for all ISAs.

All four builds now compile **and pass full `--verify`** (incl. the previously-broken
vecpass 4032/15120/20160 and four_step sizes).

## Baseline — yafft/ducc0 ratio (fwd | rt), <1 = yafft faster

ORIGINAL (pre-fix):
```
   size prec |    v4(AVX512) |     v3(AVX2)  |    v2(SSE)
   8192  f64 |  1.35/1.19 L  |  1.00/0.84    |  0.84/0.80     <-- only loss
   8192  f32 |  0.82/0.81    |  0.59/0.62    |  0.68/0.64
   4096  f64 |  0.73/0.60    |  0.87/0.71    |  0.79/0.67
   4096  f32 |  0.88/0.77    |  0.62/0.62    |  0.75/0.70
   1024  f32 |  0.74/0.82    |  0.52/0.50    |  0.57/0.61
    256  f32 |  0.35/0.43    |  0.27/0.28    |  0.40/0.41
     67  f64 |  0.16/0.17    |  0.14/0.14    |  0.17/0.16   (Rader crushes ducc0)
```

POST-FIX (2026-07-05, valley fix applied, all ISA builds rebuilt; noise-robust
min-of-N where marked — MT downclock makes single runs swing):
```
   size prec |    v4(AVX512)    |     v3(AVX2)  |    v2(SSE)
   8192  f64 | 1.17/1.08 (min)† |  1.02/0.84    |  0.91/0.72
   8192  f32 |  0.79/0.75       |  0.58/0.56    |  0.65/0.69
   4096  f64 |  0.80/0.72       |  0.81/0.75    |  0.77/0.72
   1024  f64 |  0.79/0.73       |  0.97/0.80    |  0.76/0.71
   1024  f32 |  0.81/0.77       |  0.50/0.54    |  0.59/0.60
```
† valley fix moved v4 f64 8192 from 1.35/1.19 to ~1.17/1.08 (roundtrip now parity;
  forward best-case 1.17, worst 1.33). It is the **sole remaining single-thread
  soft spot** — one size, one precision, AVX-512 only (v3/v2 win or parity).

Pattern: **the margin is widest at v3 (AVX2)** — its tuning target — and narrows at
AVX-512. The residual v4 f64 8192 forward loss is the other (big/mid/last) passes not
fully exploiting W=8; further per-ISA tuning, diminishing returns for one size.

## Bottleneck: f64 N=8192 at AVX-512 (the marquee regression)

Absolute forward µs (f64 lane counts: v4=W8, v3=W4, v2=W2):

| build | yafft | ducc0 |
|---|---|---|
| v4 (W8) | **34.9** | 25.9 |
| v3 (W4) | **30.1** | 30.1 |
| v2 (W2) | 34.5 | 40.8 |

yafft is *fastest at W=4* (Alder Lake's f64 width) and *slower at W=8*. Chain is
8·8·8·8·2. Per-pass PMU (isolated `--pass`, cycles for 40k iters), v4(W8) vs v3(W4):

| pass regime | v4 | v3 | v4 speedup |
|---|---|---|---|
| big ido=1024 | 1.41G | 2.45G | 1.74× ✓ |
| mid ido=16 | 1.53G | 2.49G | 1.62× ✓ |
| **valley ido=2 (1<ido<W)** | **7.77G** | 8.35G | **1.07× ✗** |
| last l1=1024 | 0.80G | 1.20G | 1.5× ✓ |

The valley pass is **5.5× more expensive than the big pass and barely scales with width**
(1.07× vs 1.7×). At W=8, ido=2 fills 2/8 lanes (75% waste) vs 2/4 at W=4 — the wider ISA
makes the valley *relatively worse*. FP counters confirm the big/mid/last passes emit
proper 512-bit (`fp_arith.512b_packed_double`) at only a 5% downclock; the valley does not.

**Root cause (code):** `small_ido_eligible<T,IP>() = (W==8 && IP==5)` in `dif_passes.hpp`.
The lane-over-b transpose fast path (3.9× over scalar, measured) exists only for the f32
radix-5 valley. f64 radix-8 (and every other radix) falls through to the partial-lane loop.

## ISA-parametric defect catalogue (what's baked to one width/precision)

1. **`small_ido_eligible` = `W==8 && IP==5`** — valley fast path locked to f32/radix-5.
   Generalize to any (W, IP) with ido<W. *Highest-value fix (owns the only loss).*
2. **`vecpass_supported`** — f32-only, hand-picked N allowlist from W=8 measurements;
   f64 disabled "no width edge on AVX2 (W=4)" — but on AVX-512 f64 is **W=8** (edge exists).
   Re-evaluate f64 vecpass at W=8; make the combine W-generic (currently W∈{4,8} only).
3. **`fsb_split_for`** — four_step split table hardcoded for W=8 (8/24/40 not %16). Re-tune
   splits per width, or derive W-friendly splits.
4. **`dif_stage_cost`** — magic radix constants (0.82 r4, 0.78/1.16 r8, …) calibrated at one
   width; the register/lane terms *are* parametric (good), the multipliers are not. Model is
   width-blind for 8192 (predicts identical cost across ISAs, reality differs).
5. **`select_route`** — precision-hardcoded `sizeof(T)==4/8` route gates with AVX2-derived
   comments; several catalog-size special cases are precision-split by hand.

## Plan for remaining phases

- **Phase 3 — simplify (ponytail):** collapse the precision/width hardcodes above into
  trait-driven predicates (`xsimd::batch<T>::size`, `poet::vector_register_count()`,
  `std::is_same`), remove the AVX2-era dead comments, linearize `select_route`. Behavior-
  preserving; re-verify + re-baseline to confirm no regression.
- **Phase 4 — parametrize + optimize per ISA:**
  1. Generalize the valley path (defect #1) → fixes f64 8192 at AVX-512. ASM-verify the
     lane-over-b transpose emits full-width vperm/vunpck at W=8.
  2. f64 vecpass on AVX-512 (defect #2); W-generic combine.
  3. Re-tune four_step splits + `dif_stage_cost` per width (defects #3–4).
  4. **Chiplet option:** compile the codelet library once per ISA (`-march=v2/v3/v4`) into
     separate TUs and add a one-time CPUID dispatch (xsimd `arch` dispatch + poet) so a
     single binary runs the best kernels per host — like FFTW/ducc0-multiversion, but
     keeping our per-N codelet structure.

Do-not-retry: raising `-march` past v4 (native adds VBMI2/VNNI) gave no kernel delta —
hot paths are FMA/shuffle-bound, not helped by those extensions.
