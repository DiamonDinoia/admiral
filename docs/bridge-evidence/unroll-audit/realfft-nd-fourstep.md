# Unroll-loops ASM Audit — real_fft glue + col_dif + four_step_batched
Date: 2026-07-04  |  Arch: alderlake AVX2  f32=W8 f64=W4  |  g++ 15.3.0

## Symbol resolution note

`wrap_col_pass_f32_r4`, `wrap_col_pass_f32_r8`, and `wrap_four_step_batched_f32`
compile to single-`jmp` stubs under `-fvisibility=hidden` (template comdat body
in separate symbol). Metrics are taken from the **real implementation bodies**
extracted from the same TU.

## Per-symbol table

| Symbol | Instr OFF | Instr ON | Δ% | SpillSt OFF | SpillSt ON | SpillLd OFF | SpillLd ON | Branches OFF | Branches ON | Unroll | Vec | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| audit_pack_f32 | 68 | 68 | 0% | 0 | 0 | 0 | 0 | 2 | 2 | 1x | ymm | FLAG-INVARIANT |
| audit_pack_f64 | 30 | 144 | +380% | 0 | 0 | 0 | 0 | 2 | 6 | ~4x | ymm | FLAG-DEPENDENT |
| audit_recomb_f32 | 270 | 270 | 0% | 0 | 0 | 0 | 0 | 3 | 3 | 1x | ymm | FLAG-INVARIANT |
| wrap_col_pass_f32_r4 | 1032 | 1030 | -0.2% | 9 | 9 | 19 | 19 | 14 | 14 | 1x | ymm | FLAG-INVARIANT |
| wrap_col_pass_f32_r8 | 1034 | 1030 | -0.4% | 33 | 33 | 33 | 33 | 6 | 6 | 1x | ymm | FLAG-INVARIANT |
| wrap_four_step_batched_f32 | 451 | 501 | +11% | 72 | 72 | 88 | 88 | 3 | 3 | ~1x | ymm | FLAG-DEPENDENT |

Spill stores = `%[xy]mm, N(%rsp)` (AT&T syntax). Spill loads = `N(%rsp), %[xy]mm`.
Branches = all j* opcodes. Instr = lines matching tab+lowercase.

## Per-symbol notes

- **audit_pack_f32**: FLAG-INVARIANT. W=8 f32 outer loop (stride H=4) is already
  covered by the compile-time static_for<0,8> inner body (68 instr). GCC ignores
  the flag. Zero spills. Optimal.

- **audit_pack_f64**: FLAG-DEPENDENT (+380% instr, 2->6 branches = ~4x unroll).
  W=4 f64 per-iteration body is small (30 instr OFF = 1x); -funroll-loops triggers
  ~4x outer-loop unroll (144 instr ON). No spills either way. Extra code is pure
  i-cache overhead with no register benefit -- potentially harmful for small M.

- **audit_recomb_f32**: FLAG-INVARIANT (0% delta). 270-instr body from
  static_for<0,H=4> butterfly pairs is large enough GCC refuses further unrolling.
  Zero spills. Optimal.

- **wrap_col_pass_f32_r4**: FLAG-INVARIANT (-0.2% noise). 14-branch ido/l1 nest
  refuses unrolling. 9 spill-stores + 19 spill-loads structurally fixed by radix-4
  butterfly crossing ido-stride gathers. NOT caused by -funroll-loops.

- **wrap_col_pass_f32_r8**: FLAG-INVARIANT (-0.4% noise). Radix-8 saturates AVX2
  16 YMM (8 re + 8 im live + twiddle broadcasts -> 33 spill-stores + 33 spill-loads).
  Structurally unavoidable without AVX-512. Flag makes no difference. DO-NOT-RETRY.

- **wrap_four_step_batched_f32**: Mildly FLAG-DEPENDENT (+11% instr, same branches
  and same spill counts). GCC inserts extra vector copies inside a fixed 3-branch
  skeleton -- partial 2x micro-unroll of a short inner SIMD sequence. Spills
  identical (72 stores / 88 loads). Extra 50 instructions are i-cache overhead of
  uncertain benefit.

## Summary

Four of six hot symbols are FLAG-INVARIANT: GCC's own heuristics choose identical
code regardless of -funroll-loops. The two FLAG-DEPENDENT cases are: (a)
audit_pack_f64: 4x outer-loop unroll (+380% code, zero spill change -- i-cache risk
for small M); (b) wrap_four_step_batched_f32: +11% code with identical spills --
marginal. Neither flag-dependent symbol shows increased spill pressure. The large
col_dif spill counts (r4: 19 loads; r8: 33 stores+loads) are structurally
AVX2-register-bound and completely flag-invariant -- the only remedy is AVX-512.
