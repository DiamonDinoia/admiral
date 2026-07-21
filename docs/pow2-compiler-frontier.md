# pow2 kernel frontier: compiler-constraining vs the FP-port wall

Consolidated findings + plan from the 2026-06-29 asm audit of the pow2 DIF pass
kernels (512/1024/2048 f64/f32), where we lose to FFTW ~1.4–1.7×.

## What the gap is (measured)

The pow2 ~1.4× FFTW gap splits across three passes (simdref/perf, 512 f64, pinned
P-core, `cpu_core` PMU):

| pass | share | bottleneck |
|---|---|---|
| `dif_pass<4>` middle | 39% | **GPR base-pointer spill** (not YMM, not poet) |
| `dif_pass_last<8>`   | 31% | compute/load-bound — codelet quality |
| `dif_pass_first<4>`  | 27% | compute-bound — codelet quality |

The middle pass hot inner a-loop: 65 insns, 51 vector, **10 stack reloads/iter**
(`mov 0xNNN(%rsp),%rdx` → `vmovupd (%rdx,%r8,8)`). Radix-4 SoA needs **22 distinct
array×leg base pointers** (4 in + 4 out + 3 tw, ×re/im) sharing one index `%r8`;
12 fit the 16-GPR AVX2 file, 10 spill. Address arith ≈ 73% retirement-cycle share
(misleading — overlaps on the OoO core).

**Not a poet problem.** The hot addressing is raw pointer arithmetic + `xsimd`
loads; poet only supplies the dead `U>1` `static_for` and the consteval
register-count query.

## Constraining the compiler — what was tried (all ≥ baseline)

llvm-mca (alderlake P-core proxy, the actual inner loop):

| variant | reloads | Block RThroughput |
|---|---|---|
| **GCC baseline** (22 folded bases) | 10 | **12.3 cyc/iter** |
| GCC `--param iv-max-considered-uses=1..4` | 4 | 12.7 (worse) |
| GCC source: pointer-IV strength reduction | 10 | identical codegen |
| GCC source: shared-index (`ci[j]=aa+ido*j` reused across re/im/tw) | 10 | identical codegen |
| **clang-22** (`-O3 -march=native -ffast-math`) | 3–10 | **11.5–11.7** |

Key results:
- **GCC's baseline is near-optimal.** Reloading an L1-resident fixed stack slot
  (0.33 thrpt on load ports 02/03, which sit at 8.3 — *below* the FP ports) is
  genuinely **cheaper** than computing indices on the saturated FP ports. The
  `iv-max-considered-uses` test proves it: cutting reloads 10→4 *raised*
  throughput to 12.7 because the removed reloads became index-arith uops on
  ports 0/1/5. The "10.3 without reloads" figure assumed free address generation
  — it is not free.
- The loop is **FP-port-throughput-bound** (port01 ≈ 10.7, port05 ≈ 10.0; loads
  02/03 ≈ 8.3 have spare capacity). IPC ≈ 5.0 of a 6-wide machine.
- **clang is ~5–6% faster on this kernel** (loop-versioning + a low-spill
  variant), but emits ~30% more code and the advantage is variant-dependent.

## The honest ceiling

Compiler/pragma constraining buys **at most ~5–6% on the 39% middle pass ≈
~2–3% on the whole pow2 transform**. pow2 still loses to FFTW ~1.4× afterward.
**Compiler tweaks do not flip pow2 to a win.** The real floor is FP-port
throughput (~10–11 cyc of radix-4 butterfly + 3 twiddle complex-muls). Only
**fewer FP ops** moves it — i.e. genfft-style minimal-op / larger straight-line
codelets (radix-8/16 middle passes that amortize twiddle muls and cut pass
count). That is the structural FFTW gap, designed out of this engine.

## Plan

### Phase A — targeted compiler constraining (cheap, ~1 session, low risk)
Bank the ~2–3% if it survives a real A/B; do NOT touch global flags.
1. **Per-TU clang for the pass kernels.** Compile only the `dif_passes`
   instantiation TU(s) with clang while the rest stays GCC (separate `.o`,
   ABI-compatible). CMake `set_source_files_properties(... COMPILE_OPTIONS)`
   or a dedicated object library. Gate: paired-interleaved A/B on pow2
   (512/1024/2048) **and** a composite guard set (1260/1500/2520/5040/7560) —
   the ido-ordering overrides were tuned on GCC; verify no regression there.
2. **Per-loop pragmas** (portable, per-compiler) on the a-loop:
   `#pragma GCC unroll` / `#pragma clang loop interleave_count / vectorize_width`.
   Note [[unroll-levers-dead-end]]: manual unroll regressed — but pragma
   interleave/vectorize_width is untested. A/B each.
3. If neither clears a threshold (say ≥3% on pow2, no composite regression),
   record do-not-retry and stop. Compiler choice is not the lever.

### Phase B — the real lever: genfft-style codelets (hard, uncertain, the actual gap)
Cut FP-port pressure below the ~10–11 cyc floor.
1. Prototype a **larger straight-line middle pass** (radix-8/16) with minimized
   op count + scheduled twiddle muls; model with llvm-mca (target the FP-port
   pressure down, fewer passes over memory).
2. Compare against the current radix-4 chain on 512/1024/2048 both prec.
3. This is FFTW's actual advantage; success is uncertain against mature genfft
   codelets, but it is the only path to a pow2 win.

### Phase C — orthogonal, proven (incremental wins now)
Extend the **composite ido-ordering override** search (the only lever that has
already beaten FFTW — 2520 f32 fwd 0.97×) to more smooth composites
(3360/4200/6300/10080…). Interleaved factor-search + N-independent-plan
vs-FFTW gate. See [[ido-ordering-2520-f32-beats-fftw]].

## Do-not-retry (this session)
- Source-level addressing restructure of `dif_pass<4>` (pointer-IV, shared-index):
  GCC canonicalizes both back to the 22-base/10-spill form. See
  memory `dif-pass4-gpr-spill-source-unfixable`.
- GCC `--param iv-max-considered-uses`: cuts reloads but raises throughput.
