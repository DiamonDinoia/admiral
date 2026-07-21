# pow2 FFTW-gap codelet frontier

Campaign: close the pure-power-of-2 gap to FFTW with genfft-style fused
twiddle+DFT codelets. We already beat ducc0 on every pow2 size; FFTW is the
real ceiling.

## Phase 0 — fresh same-machine A/B + gap attribution (2026-06-29)

In-bench FFTW reference shipped (`-DFFT_BENCH_FFTW=ON`, `compare_min_of_n`,
FFTW_MEASURE plans reused, accuracy-gated). Measured `taskset -c 0`, nanobench
`cpucycles` (frequency-invariant), reps=15.

### Gap: fft / FFTW (cyc, forward)

| N    | f64  | f32  |
|------|------|------|
| 512  | 1.54 | 1.96 |
| 1024 | 1.54 | 1.72 |
| 2048 | 1.59 | 1.44 |
| 4096 | 1.14 | 1.65 |

(For reference, fft / ducc0 fwd is 0.66–0.94 everywhere — we beat ducc0; FFTW
is faster.) f64 gap is a stable ~1.5× at 512–2048, then collapses to 1.14× at
4096 — the memory-traffic onset.

### Attribution — perf counters on our forward kernel (tight loop)

| N    | IPC  | stalls_total | L2-miss stalls | L3-miss stalls | FP uops/call (p0+p1+p5) |
|------|------|--------------|----------------|----------------|--------------------------|
| 512  | 3.04 | 14.1%        | 0              | 0              | ~3748 (vector, W=4)      |
| 2048 | 2.37 | 18.5%        | 0              | 0              | ~3933                    |

**Zero cache-miss stalls at both L2-resident sizes** → the 512–2048 gap is NOT
memory traffic. High IPC + low stalls → the kernel is op-throughput-bound: it
executes *more ops* than FFTW, efficiently. (4096's 1.14× wall gap is memory
onset, consistent with `pow2-f64-l2-resident-not-cache-bound`: ~40% mem-bound
at N>=4096.)

### Op-count: why FFTW does fewer ops

genfft fused twiddle codelets (one per radix, FMA-scheduled):

| codelet | adds | muls | fmas |
|---------|------|------|------|
| t1_8    | 44   | 14   | 22   |
| t1_16   | 104  | 30   | 70   |
| t1_32   | 236  | 62   | 198  |

FFTW fuses the output twiddle multiply *into* the size-R DFT, in register, one
pass. We do a **separate twiddle-multiply pass + an add-only `dif_butterfly`**
— more total ops/point AND an extra memory round-trip per pass. That is the
~1.5× f64 gap.

### Decision: **GO** to Phase 1

The 512–2048 f64 gap is recoverable op-count/fusion, not memory. Port one fused
genfft codelet (t1_8, then t1_16 if it fits) to `batch<T>` and perf-gate it.
Crux risk unchanged: AVX2 16-YMM register file (prior flat radix-32 spilled
537×; radix-8 middle pass register-starved). Phase 1 must measure register fit
(objdump spill census) before any wiring; abort to AVX-512-only if it spills.

Methodology gate (load-bearing): perf `cpu_core/cycles` / nanobench cpucycles,
pinned, accuracy-gated. rdtscp min-of-N ranking is forbidden (frequency-warmup
artifact — see memory `rdtscp-minofn-frequency-trap`).

## Phase 1 — port fused t1_8 to batch<T> + perf-gate (2026-06-29): **ABORT**

Ported the FMA variant of FFTW genfft `t1_8` (44a+14m+22fma) faithfully to a
straight-line `batch<T>` codelet (scratch `probe_t1_8.cpp`). Correct: identity
twiddles reproduce the forward DFT8 at l2=7.8e-16; plain `a*b+c` expressions
contracted to 40 vfmadd/vfnmadd (no stray vxorpd).

Baseline = the CURRENT engine path: 7 input twiddle complex-muls (`poet::static_for`)
+ the real `dif_butterfly<8>`. Both kernels agree to 1.5e-16.

| metric (f64, W=4, 50M calls, perf cyc) | fused t1_8 | current path |
|----------------------------------------|------------|--------------|
| stack-spill stores (objdump)           | 20         | 23           |
| cycles                                 | 3.2528e9   | 3.2575e9     |

**Result: a dead TIE (0.14%), fused spills no worse.** (An early "8.5% win" was a
measurement bug — a manual `for` baseline that didn't fully unroll; fixing it to
`poet::static_for` erased the gap.)

### Why it ties — the Phase-0 premise was wrong

Reading `dif_passes.hpp:219-233`: the engine **already fuses the output twiddle
into the butterfly emit, in-register, at the store** —
`(owr*sr - owi*si).store_unaligned(...)`. There is NO separate twiddle pass / extra
memory round-trip. genfft's twiddle-fusion advantage is something we already have,
so a per-codelet port is structurally a no-op on AVX2.

### Pass-count lever (the only thing left) also bottoms out

`--factors` sweep vs FFTW (f64): radix-4 chain for 512 = 2.69× FFTW (5 passes,
terrible); all-radix-8 (8-8-8, 3 passes) is best but **still 1.59× FFTW**; radix-16
fails the accuracy gate (not wired), and 512=16×16×2 is still 3 passes anyway while
radix-32 register-starves. Fewer/larger passes can't beat the 16-YMM wall.

### Decision: do-not-retry on AVX2; AVX-512-only

The ~1.5× pow2-f64 FFTW gap at L2-resident sizes is NOT recoverable on AVX2 via
genfft fused codelets (we already fuse the twiddle) nor via larger-radix passes
(register wall). Confirms the long-standing AVX-512-only verdict. No engine wiring;
`probe_t1_8.cpp` kept in scratch only. The in-bench FFTW reference (Phase 0b) ships
as the reusable gate.

Minor follow-up (not chased): forcing 8-8-8 for 512 f64 measured ~3% under the
default route (0.947 vs 0.977us, consistent fwd+rt) — possible small override if
the default isn't already 8-8-8; within nanobench noise, needs N-independent-plan
confirmation.

## Phase A — the load-SCHEDULE hypothesis, tested decisively (2026-06-29): **NO-GO**

Phase 1's probe pre-loaded `V xr[8]` arrays, so it tested genfft's *arithmetic*
(equal to ours at radix-8) but not its *load schedule*. New hypothesis: genfft
reads `ri[WS(rs,j)]` interleaved with the compute DAG (late loads), which might
lower peak-live, cut radix-8 spills, and make radix-16 fit 16 YMM. Tested both
arms; both fail.

### A1 — late-load t1_8 codelet vs current DIF path (scratch `probe_t1_8.cpp`)

Rewrote the t1_8 FMA codelet with `batch::load_unaligned` at genfft's *scheduled*
read points (NOT hoisted). Correct: l2 = 9.0e-16 vs forward DFT8. Baseline = the
real engine `do_batch` (load-all-up-front 16 input batches + `dif_butterfly<8>` +
fused output twiddle). Both f64 W=4. objdump spill census + 7-run paired perf
`cpu_core/cycles`, `taskset -c 0`.

| metric (one radix-8 butterfly) | late-load t1_8 | current DIF (load-all) |
|--------------------------------|----------------|------------------------|
| stack-spill stores / reloads   | **9 / 8**      | **0 / 0**              |
| FP ops (fma / mul / add)        | 40 / 10 / 40   | 19 / 16 / 52           |
| perf cyc (median, 10M calls×16) | 9.85e9         | 8.42e9                 |

**The late-load codelet SPILLS (9/8) and is 17.0% SLOWER. The opposite of the
hypothesis.** Why: gcc -O3 already reorders our "load-all" source — it schedules
the 16 loads late on its own (0 spills with a full register file of inputs). What
spills is genfft's *flat named-temp* schedule: temps like `T12,T17,TF,TL` live
across both output blocks, longer live ranges than our recursive even/odd split,
which kills temps per scope (`pow2_dif_butterfly`'s two `{}` halves). Our structure
is strictly better for peak-live on AVX2. Source load position does NOT control
register allocation; the dependency DAG does.

### A2 — radix-16 wired (`dif_radix_set += 16`) + forced `--factors` vs FFTW

radix-16 instantiates correctly (`pow2_dif_butterfly<16>`, l2 ~1-2e-15) but is a
large regression — register-starved exactly as predicted (2·16 = 32 batches on
16 YMM):

| N (f64 fwd) | forced radix-16 chain | default route | (both vs FFTW) |
|-------------|-----------------------|---------------|----------------|
| 512  (16-16-2)  | 3.48× | 1.72× |
| 1024 (16-16-4)  | 1.82× | 1.54× |
| 4096 (16-16-16) | 1.66× | 1.25× |

No headroom; radix-16 reverted. Confirms the schedule (not the radix) was the
hypothesized lever — and the schedule lever is disproven by A1.

## Phase C — concede + redirect (definitive do-not-retry)

The residual pow2 f64 AVX2 gap to FFTW at L2-resident sizes (512–2048 ~1.5×) is
**NOT recoverable in a generic radix engine on AVX2** by either (a) genfft-style
late/interleaved load scheduling — which spills MORE and runs 17% slower than our
recursive butterfly because gcc already schedules our loads optimally — or (b)
larger-radix passes (radix-16 register-starves 16 YMM). It is genfft's mature
op-minimal codelet *library* + FFTW_MEASURE empirical scheduling, reproducible
only with a codelet generator and (effectively) AVX-512's 32 ZMM. Both the Phase-1
"we already fuse the twiddle" finding and this Phase-A "late-load is worse" finding
independently close the door. **Do not retry genfft scheduling / radix-16 on AVX2.**

Redirect to the only lever that has actually beaten FFTW: ido-ordering overrides
for smooth composites (2520 f32 fwd 0.97×; candidates 3360/4200/6300/10080).
The in-bench FFTW reference (`-DFFT_BENCH_FFTW`) is the reusable gate. Separate plan.
