# Fixed-`-march` ASM audit + fair ducc0 comparison (SSE / SSE4.2 / AVX2)

Date: 2026-07-02. CPU: AVX2-only (no AVX-512). FFTW excluded (runtime CPUID
dispatch → `-march`-invariant → unfair). ducc0 built from CPM source, inherits
our `-march`.

## Build mechanism

Two `-march=native` sources neutralised so one chosen `-march` applies to every
TU (fft lib + header-instantiated engine in the bench TU + ducc0):

- `benchmark/CMakeLists.txt` — `-march=native` gated behind `FFT_USE_NATIVE_ARCH`
  (`-ffast-math` kept unconditional). Only intended source edit.
- Configure with `-DFFT_USE_NATIVE_ARCH=OFF -DCMAKE_C_FLAGS=-march=$LVL
  -DCMAKE_CXX_FLAGS=-march=$LVL` per level, `FFT_BUILD_TESTS=ON` (stale-binary guard).

### Required extra edit (build blocker, not just a finding)

`vecpass.hpp:186 static_assert(W==4||W==8)` is a **hard compile error at W=2**
(f64 on 128-bit SSE), so x86-64 / x86-64-v2 did not build at all. The `vpass_forward`
body is instantiated for f64 even though the route is never selected there
(`vecpass_supported<double>` is always false). Fix: gate the two call sites in
`vecpass_plan::execute` with `if constexpr (W==4||W==8)` so the W=2 body is never
instantiated. Provably a no-op at AVX2 widths (f64 W=4, f32 W=8 → always-true
branch → identical codegen; v3 ctest 114/114 confirms).

## ISA verification (final linked binary, not `.o` — LTO makes objects bitcode)

| Level | `-march` (compile_commands) | ymm | zmm | vfma | verdict |
|---|---|---|---|---|---|
| x86-64    | `x86-64`    | 0      | 0 | 0     | pure SSE, no FMA ✓ |
| x86-64-v2 | `x86-64-v2` | 0      | 0 | 0     | pure SSE, no FMA ✓ |
| x86-64-v3 | `x86-64-v3` | 194496 | 0 | 48175 | AVX2 + FMA3, no AVX-512 ✓ |

Exactly one `-march` per tree; flag took; no leak.

## Step 2 — ASM audit of the register heuristic

`poet::vector_register_count()` = **16 for SSE2/SSE4.2/AVX2 alike** → the
count-driven choices (flat-leaf N≤8, unroll U=1, noinline r>3) are **byte-identical
across all three x86 levels**; only lane width `W` changes. Non-LTO `-S`/`objdump`
of `dif_pass<4>`, `dif_pass<8>`, `kernel_batched<8>` (codelet_8), radix-5 (codelet_5),
both precisions, forward:

Per-symbol (fwd; **SSE2 ≡ SSE4.2 byte-for-byte** so collapsed to one "SSE" column):

| kernel (f64 fwd) | SSE vSTsp / vLDsp / #xmm / fma | AVX2 vSTsp / vLDsp / #ymm / fma |
|---|---|---|---|
| radix-5 `kernel<5>::apply` | 0 / 0 / 16 / 0 | 1 / 1 / 16 / 12 |
| flat-leaf `codelet_apply<8>` | 15 / 14 / 16 / 0 | 1 / 1 / 16 / 6 |
| `dif_pass<4>` | 0 / 2 / 16 / 0 | 0 / 0 / 16 / 36 |
| `dif_pass<8>` | 22 / 17 / 16 / 0 | 22 / 14 / 16 / 64 |

f32 mirrors f64 (fewer f64-only spills). Findings:

1. **`#xmm`/`#ymm` ≤ 16 everywhere** — no width-driven register-count pressure.
   The heuristic counts distinct SIMD *values*, which is width-invariant, so the
   same choice is correct at every width. Narrower width = fewer bytes/reg = strictly
   more headroom, never less.
2. **FMA folds on v3** (`c-a*b` → `vfnmadd`), **falls back to `mulpd+addpd` on SSE**
   with no compile error. Confirmed: codelet_8 f64 = 128 mul/add/sub + 0 FMA (SSE)
   vs 112 + 24 FMA (v3).
3. **The one real asymmetry — flat-leaf N=8 f64 spills 15 on SSE vs 1 on v3 — is
   FMA-driven, not a heuristic bug.** Without FMA each complex multiply needs a live
   scratch (mul+mul then add/sub), extending live ranges of a size-8 DFT (already
   16 SIMD values for I/O) past 16 → spills. This is intrinsic to the SSE ISA; no
   flat-leaf/unroll retune fixes it (shrinking the leaf trades spills for
   call/dispatch overhead — a wash; FMA is simply unavailable).
4. **`dif_pass<8>` 22 spills identical on SSE and v3** — the pre-existing,
   width-independent radix-8 register wall (documented; not introduced by narrowing).
   **`dif_pass<4>` clean everywhere.** The known-benign `dif_pass<4>` GPR
   base-pointer spill is source-unfixable and expected.

**Verdict: the count-driven heuristic is optimal/correct as-is for SSE/v2.** Narrower
width leaves **no** headroom for a larger flat leaf or more unroll — the binding
constraints (distinct-value count + FMA-temp pressure) are both width-invariant.
Confirms the prior prediction.

## Width-hardcode findings (dead paths at SSE, separate from spills)

The register *count* is fully parametric; several sites hardcode the SIMD *width*
to the AVX2-f32 value (8), silently disabling optimization paths at SSE widths:

- `vecpass.hpp:186` `static_assert(W==4||W==8)` — **compile error at W=2** (fixed above).
- `dif_passes.hpp:34` `small_ido_eligible() → size==8` — small-ido lane-over-b
  transpose OFF for all SSE f32 (W=4) and all f64; N=1260/1500/5040 ido=5/7 passes
  fall to the scalar valley.
- `vecpass.hpp:275` `N % 8` size filter — should be `N % W`.
- `four_step.hpp:282` `fsb_split_for` — splits baked for W=8.
- `twiddles.hpp:247` `measured_dif_factor_plan` — overrides A/B-measured on AVX2 only,
  served verbatim at SSE (suboptimal, not incorrect).

**But these do NOT cost us the ducc0 comparison** (see Step 3): ducc0 loses its own
width edge at SSE at least as fast as we do, so relative ratios hold or improve.
Width-parametric fixes are a real follow-up for *absolute* SSE throughput, not a
correctness or a "losing-to-ducc0" issue.

## Step 3 — Fair fft-vs-ducc0 comparison (same `-march`, `m=cyc`, `taskset -c 2`)

1-D `--compare` (fwd_ratio = fft/ducc, **<1 = we win**), N-D `--compare-nd --robust`
(ABND = ours/ducc0), both precisions. All levels: **ctest 114/114, `--verify` 0 fails,
0 `FAIL (inaccurate)`.**

| Level | 1-D f64 geomean fwd / wins | 1-D f32 geomean fwd / wins | N-D ours-faster / ducc-faster | IDENT sanity (mean fwd / max spread) |
|---|---|---|---|---|
| x86-64 (SSE2)    | 0.439 / 41-41 | 0.379 / 41-41 | **16 / 0** | 1.004 / 6.4% |
| x86-64-v2 (SSE4) | 0.441 / 41-41 | 0.383 / 41-41 | **14 / 2** | 1.001 / 6.1% |
| x86-64-v3 (AVX2) | 0.423 / 40-41 | 0.379 / 41-41 | **10 / 6** | 0.997 / 13.0% |

Large pow2 f64 fwd_ratio (512 / 1024 / 2048 / 4096): SSE `0.67 0.70 0.77 0.66`;
AVX2 `0.85 0.91 0.91 0.88`.

**Read:** we beat ducc0 broadly at *every* level. Counter-intuitively our *relative*
advantage is **larger at SSE** — ducc0 (whose vlen is hardcoded to 4) degrades more
than we do when the ISA narrows. The N-D cases ducc0 wins at AVX2 (512²/1024² f64
column passes, its genfft AVX2 codelet frontier) evaporate at SSE. IDENT (ours/ours
control) ≈ 1.0 at every level → harness trustworthy; the single 13% AVX2 spread is
one high-variance large cube (mean unbiased).

## Scope notes

- SSE2 and SSE4.2 produce **byte-identical** vector hot-loop codegen (16 XMM, no FMA);
  v2's only wins are a few scalar/blend/round insns outside the FP loops. Collapsed.
- v4/AVX-512 skipped (CPU can't run it) — the only level where the heuristic actually
  changes (flat-leaf N≤16, U=2) and where r16-f32 might spill. Revisit on an AVX-512 box.
