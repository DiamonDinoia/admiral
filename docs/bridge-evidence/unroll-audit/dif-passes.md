# DIF Passes ASM Audit: -funroll-loops impact

Compiler: g++ 15.3.0
Flags: -O3 -ffast-math -fno-math-errno -ffinite-math-only -march=native -mtune=native -fomit-frame-pointer -std=c++20 [+funroll-loops for ON variant]

**ON** = with `-funroll-loops` (production).  **OFF** = without.
**li** = instructions in largest loop body.  **fma** = vfmadd/vfnmadd/vmulps count in loop.
**UF** = estimated unroll factor (on_fma / off_fma, rounded).  
**spill_s** = vmovap*/vmovup* %ymm→(%rsp) stores.  **rsp_r** = %ymm loads from (%rsp).
**FLAG-DEPENDENT**: UF>1 or loop-instr delta >5% or spill count changed between ON/OFF.

| Symbol | ON li | ON fma | ON UF | ON spills | ON reloads | OFF li | OFF fma | OFF spills | OFF reloads | Δ% | Verdict | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| dif_pass_f32_r2 | 400 | 20 | 1 | 0 | 0 | 344 | 14 | 0 | 0 | 16% | FLAG-DEPENDENT | ok |
| dif_pass_f32_r3 | 758 | 108 | 1 | 0 | 0 | 613 | 76 | 0 | 0 | 24% | FLAG-DEPENDENT | ok |
| dif_pass_f32_r4 | 659 | 36 | 1 | 0 | 0 | 660 | 36 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_f32_r5 | 610 | 92 | 1 | 0 | 0 | 614 | 92 | 0 | 0 | 1% | FLAG-INVARIANT | ok |
| dif_pass_f32_r7 | 632 | 114 | 1 | 6 | 9 | 641 | 114 | 6 | 9 | 1% | FLAG-INVARIANT | spills:6wr/9rd |
| dif_pass_f32_r8 | 606 | 53 | 1 | 2 | 0 | 606 | 53 | 2 | 0 | 0% | FLAG-INVARIANT | spills:2wr/0rd |
| dif_pass_f64_r2 | 329 | 26 | 2 | 0 | 0 | 240 | 11 | 0 | 0 | 37% | FLAG-DEPENDENT | ok |
| dif_pass_f64_r3 | 654 | 92 | 2 | 0 | 0 | 445 | 52 | 0 | 0 | 47% | FLAG-DEPENDENT | ok |
| dif_pass_f64_r4 | 590 | 33 | 1 | 0 | 0 | 591 | 33 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_f64_r5 | 378 | 56 | 1 | 0 | 0 | 378 | 56 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_f64_r7 | 612 | 108 | 1 | 8 | 11 | 617 | 108 | 8 | 11 | 1% | FLAG-INVARIANT | spills:8wr/11rd |
| dif_pass_f64_r8 | 619 | 53 | 1 | 8 | 0 | 621 | 53 | 8 | 0 | 0% | FLAG-INVARIANT | spills:8wr/0rd |
| dif_pass_first_f32_r2 | 533 | 13 | 1 | 0 | 0 | 415 | 11 | 0 | 0 | 28% | FLAG-DEPENDENT | ok |
| dif_pass_first_f32_r3 | 852 | 98 | 1 | 0 | 0 | 743 | 82 | 0 | 0 | 15% | FLAG-DEPENDENT | ok |
| dif_pass_first_f32_r4 | 1014 | 33 | 1 | 0 | 0 | 1006 | 33 | 0 | 0 | 1% | FLAG-INVARIANT | ok |
| dif_pass_first_f32_r5 | 416 | 60 | 1 | 3 | 2 | 416 | 60 | 3 | 2 | 0% | FLAG-INVARIANT | spills:3wr/2rd |
| dif_pass_first_f32_r7 | 701 | 114 | 1 | 8 | 15 | 722 | 114 | 8 | 15 | 3% | FLAG-INVARIANT | spills:8wr/15rd |
| dif_pass_first_f32_r8 | 691 | 53 | 1 | 13 | 8 | 691 | 53 | 13 | 8 | 0% | FLAG-INVARIANT | spills:13wr/8rd |
| dif_pass_first_f64_r2 | 462 | 11 | 1 | 0 | 0 | 352 | 9 | 0 | 0 | 31% | FLAG-DEPENDENT | ok |
| dif_pass_first_f64_r3 | 733 | 70 | 1 | 0 | 0 | 630 | 58 | 0 | 0 | 16% | FLAG-DEPENDENT | ok |
| dif_pass_first_f64_r4 | 868 | 27 | 1 | 0 | 0 | 857 | 27 | 0 | 0 | 1% | FLAG-INVARIANT | ok |
| dif_pass_first_f64_r5 | 347 | 46 | 1 | 1 | 1 | 346 | 46 | 1 | 1 | 0% | FLAG-INVARIANT | spills:1wr/1rd |
| dif_pass_first_f64_r7 | 528 | 84 | 1 | 7 | 13 | 531 | 84 | 7 | 13 | 1% | FLAG-INVARIANT | spills:7wr/13rd |
| dif_pass_first_f64_r8 | 546 | 47 | 1 | 13 | 7 | 546 | 47 | 13 | 7 | 0% | FLAG-INVARIANT | spills:13wr/7rd |
| dif_pass_fused2_f32_4x4 | 249 | 24 | 1 | 0 | 0 | 249 | 24 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_fused2_f32_4x8 | 417 | 46 | 1 | 0 | 0 | 417 | 46 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_fused2_f32_8x4 | 427 | 46 | 1 | 5 | 1 | 429 | 46 | 5 | 1 | 0% | FLAG-INVARIANT | spills:5wr/1rd |
| dif_pass_fused2_f32_8x8 | 589 | 68 | 1 | 5 | 1 | 589 | 68 | 5 | 1 | 0% | FLAG-INVARIANT | spills:5wr/1rd |
| dif_pass_fused2_f64_4x4 | 247 | 24 | 1 | 0 | 0 | 247 | 24 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_fused2_f64_4x8 | 416 | 46 | 1 | 0 | 0 | 418 | 46 | 0 | 0 | 0% | FLAG-INVARIANT | ok |
| dif_pass_fused2_f64_8x4 | 427 | 46 | 1 | 5 | 1 | 429 | 46 | 5 | 1 | 0% | FLAG-INVARIANT | spills:5wr/1rd |
| dif_pass_fused2_f64_8x8 | 588 | 68 | 1 | 5 | 1 | 588 | 68 | 5 | 1 | 0% | FLAG-INVARIANT | spills:5wr/1rd |
| dif_pass_fused3_f32_4x4x4 | 525 | 72 | 1 | 2 | 2 | 525 | 72 | 2 | 2 | 0% | FLAG-INVARIANT | spills:2wr/2rd |
| dif_pass_fused3_f64_4x4x4 | 536 | 72 | 1 | 2 | 2 | 536 | 72 | 2 | 2 | 0% | FLAG-INVARIANT | spills:2wr/2rd |
| dif_pass_last_f32_r2 | 341 | 0 | 1 | 0 | 0 | 186 | 0 | 0 | 0 | 83% | FLAG-DEPENDENT | ok |
| dif_pass_last_f32_r3 | 440 | 30 | 1 | 0 | 0 | 370 | 24 | 0 | 0 | 19% | FLAG-DEPENDENT | ok |
| dif_pass_last_f32_r4 | 352 | 0 | 1 | 0 | 0 | 284 | 0 | 0 | 0 | 24% | FLAG-DEPENDENT | scalar-only |
| dif_pass_last_f32_r5 | 313 | 36 | 1 | 4 | 0 | 315 | 36 | 4 | 0 | 1% | FLAG-INVARIANT | spills:4wr/0rd; scalar-only |
| dif_pass_last_f32_r7 | 137 | 24 | 1 | 8 | 0 | 138 | 24 | 8 | 0 | 1% | FLAG-INVARIANT | spills:8wr/0rd; scalar-only |
| dif_pass_last_f32_r8 | 127 | 6 | 1 | 0 | 0 | 127 | 6 | 0 | 0 | 0% | FLAG-INVARIANT | scalar-only |
| dif_pass_last_f64_r2 | 335 | 0 | 1 | 0 | 0 | 159 | 0 | 0 | 0 | 111% | FLAG-DEPENDENT | ok |
| dif_pass_last_f64_r3 | 350 | 24 | 1 | 0 | 0 | 277 | 18 | 0 | 0 | 26% | FLAG-DEPENDENT | ok |
| dif_pass_last_f64_r4 | 385 | 0 | 1 | 0 | 0 | 367 | 0 | 0 | 0 | 5% | FLAG-INVARIANT | ok |
| dif_pass_last_f64_r5 | 83 | 8 | 1 | 0 | 0 | 90 | 8 | 4 | 1 | 8% | FLAG-DEPENDENT | scalar-only |
| dif_pass_last_f64_r7 | 121 | 20 | 1 | 0 | 0 | 121 | 20 | 0 | 0 | 0% | FLAG-INVARIANT | scalar-only |
| dif_pass_last_f64_r8 | 130 | 6 | 1 | 0 | 0 | 131 | 6 | 0 | 0 | 1% | FLAG-INVARIANT | scalar-only |

## FLAG-DEPENDENT symbols
- **dif_pass_f32_r2**: UF=1, delta=16%, on_spills=0/0, notes=ok
- **dif_pass_f32_r3**: UF=1, delta=24%, on_spills=0/0, notes=ok
- **dif_pass_f64_r2**: UF=2, delta=37%, on_spills=0/0, notes=ok
- **dif_pass_f64_r3**: UF=2, delta=47%, on_spills=0/0, notes=ok
- **dif_pass_first_f32_r2**: UF=1, delta=28%, on_spills=0/0, notes=ok
- **dif_pass_first_f32_r3**: UF=1, delta=15%, on_spills=0/0, notes=ok
- **dif_pass_first_f64_r2**: UF=1, delta=31%, on_spills=0/0, notes=ok
- **dif_pass_first_f64_r3**: UF=1, delta=16%, on_spills=0/0, notes=ok
- **dif_pass_last_f32_r2**: UF=1, delta=83%, on_spills=0/0, notes=ok
- **dif_pass_last_f32_r3**: UF=1, delta=19%, on_spills=0/0, notes=ok
- **dif_pass_last_f32_r4**: UF=1, delta=24%, on_spills=0/0, notes=scalar-only
- **dif_pass_last_f64_r2**: UF=1, delta=111%, on_spills=0/0, notes=ok
- **dif_pass_last_f64_r3**: UF=1, delta=26%, on_spills=0/0, notes=ok
- **dif_pass_last_f64_r5**: UF=1, delta=8%, on_spills=0/0, notes=scalar-only

## Spill summary (prod flags / ON)
- **dif_pass_f32_r7**: 6 stores / 9 reloads in hot loop
- **dif_pass_f32_r8**: 2 stores / 0 reloads in hot loop
- **dif_pass_f64_r7**: 8 stores / 11 reloads in hot loop
- **dif_pass_f64_r8**: 8 stores / 0 reloads in hot loop
- **dif_pass_first_f32_r5**: 3 stores / 2 reloads in hot loop
- **dif_pass_first_f32_r7**: 8 stores / 15 reloads in hot loop
- **dif_pass_first_f32_r8**: 13 stores / 8 reloads in hot loop
- **dif_pass_first_f64_r5**: 1 stores / 1 reloads in hot loop
- **dif_pass_first_f64_r7**: 7 stores / 13 reloads in hot loop
- **dif_pass_first_f64_r8**: 13 stores / 7 reloads in hot loop
- **dif_pass_fused2_f32_8x4**: 5 stores / 1 reloads in hot loop
- **dif_pass_fused2_f32_8x8**: 5 stores / 1 reloads in hot loop
- **dif_pass_fused2_f64_8x4**: 5 stores / 1 reloads in hot loop
- **dif_pass_fused2_f64_8x8**: 5 stores / 1 reloads in hot loop
- **dif_pass_fused3_f32_4x4x4**: 2 stores / 2 reloads in hot loop
- **dif_pass_fused3_f64_4x4x4**: 2 stores / 2 reloads in hot loop
- **dif_pass_last_f32_r5**: 4 stores / 0 reloads in hot loop
- **dif_pass_last_f32_r7**: 8 stores / 0 reloads in hot loop

## Symbol → impl mapping (on.s)

| Symbol | Impl (demangled) |
|---|---|
| dif_pass_f32_r2 | `dif_pass<float, true, 2>` |
| dif_pass_f32_r3 | `dif_pass<float, true, 3>` |
| dif_pass_f32_r4 | `dif_pass<float, true, 4>` |
| dif_pass_f32_r5 | `dif_pass<float, true, 5>` |
| dif_pass_f32_r7 | `dif_pass<float, true, 7>` |
| dif_pass_f32_r8 | `dif_pass<float, true, 8>` |
| dif_pass_f64_r2 | `dif_pass<double, true, 2>` |
| dif_pass_f64_r3 | `dif_pass<double, true, 3>` |
| dif_pass_f64_r4 | `dif_pass<double, true, 4>` |
| dif_pass_f64_r5 | `dif_pass<double, true, 5>` |
| dif_pass_f64_r7 | `dif_pass<double, true, 7>` |
| dif_pass_f64_r8 | `dif_pass<double, true, 8>` |
| dif_pass_first_f32_r2 | `dif_pass_first<float, true, 2>` |
| dif_pass_first_f32_r3 | `dif_pass_first<float, true, 3>` |
| dif_pass_first_f32_r4 | `dif_pass_first<float, true, 4>` |
| dif_pass_first_f32_r5 | `dif_pass_first<float, true, 5>` |
| dif_pass_first_f32_r7 | `dif_pass_first<float, true, 7>` |
| dif_pass_first_f32_r8 | `dif_pass_first<float, true, 8>` |
| dif_pass_first_f64_r2 | `dif_pass_first<double, true, 2>` |
| dif_pass_first_f64_r3 | `dif_pass_first<double, true, 3>` |
| dif_pass_first_f64_r4 | `dif_pass_first<double, true, 4>` |
| dif_pass_first_f64_r5 | `dif_pass_first<double, true, 5>` |
| dif_pass_first_f64_r7 | `dif_pass_first<double, true, 7>` |
| dif_pass_first_f64_r8 | `dif_pass_first<double, true, 8>` |
| dif_pass_fused2_f32_4x4 | `dif_pass_fused2<float, true, 4, 4>` |
| dif_pass_fused2_f32_4x8 | `dif_pass_fused2<float, true, 4, 8>` |
| dif_pass_fused2_f32_8x4 | `dif_pass_fused2<float, true, 8, 4>` |
| dif_pass_fused2_f32_8x8 | `dif_pass_fused2<float, true, 8, 8>` |
| dif_pass_fused2_f64_4x4 | `dif_pass_fused2<double, true, 4, 4>` |
| dif_pass_fused2_f64_4x8 | `dif_pass_fused2<double, true, 4, 8>` |
| dif_pass_fused2_f64_8x4 | `dif_pass_fused2<double, true, 8, 4>` |
| dif_pass_fused2_f64_8x8 | `dif_pass_fused2<double, true, 8, 8>` |
| dif_pass_fused3_f32_4x4x4 | `dif_pass_fused3<float, true, 4, 4, 4>` |
| dif_pass_fused3_f64_4x4x4 | `dif_pass_fused3<double, true, 4, 4, 4>` |
| dif_pass_last_f32_r2 | `dif_pass_last<float, true, 2, false>` |
| dif_pass_last_f32_r3 | `dif_pass_last<float, true, 3, false>` |
| dif_pass_last_f32_r4 | `dif_pass_last<float, true, 4, false>` |
| dif_pass_last_f32_r5 | `dif_pass_last<float, true, 5, false>` |
| dif_pass_last_f32_r7 | `dif_pass_last<float, true, 7, false>` |
| dif_pass_last_f32_r8 | `dif_pass_last<float, true, 8, false>` |
| dif_pass_last_f64_r2 | `dif_pass_last<double, true, 2, false>` |
| dif_pass_last_f64_r3 | `dif_pass_last<double, true, 3, false>` |
| dif_pass_last_f64_r4 | `dif_pass_last<double, true, 4, false>` |
| dif_pass_last_f64_r5 | `dif_pass_last<double, true, 5, false>` |
| dif_pass_last_f64_r7 | `dif_pass_last<double, true, 7, false>` |
| dif_pass_last_f64_r8 | `dif_pass_last<double, true, 8, false>` |
