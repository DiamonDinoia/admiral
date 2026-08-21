# Benchmarks

`admiral_benchmark` times Admiral against ducc0 with nanobench,
and against FFTW when built with `-DADM_BENCH_FFTW=ON`. Not installed.

## Build and run

```bash
cmake --preset release          # benchmarks build in this preset
cmake --build --preset release
./build/release/benchmark/admiral_benchmark --compare       # 1-D vs ducc0
./build/release/benchmark/admiral_benchmark --compare-nd    # N-D vs ducc0
```

Main modes:

| Mode | What it reports |
|------|-----------------|
| `--compare` | 1-D ratio vs ducc0 (and FFTW), per size, both precisions |
| `--compare-nd` / `--compare-2d` | The same for N-D shapes; `--r2c` switches to real transforms |
| `--verify` | Correctness sweep only, nonzero exit on failure |
| `--fftw-ab` | Interleaved Admiral vs FFTW A/B (needs `-DADM_BENCH_FFTW=ON`) |
| `--size=N` | Single-size profiling of the library alone |
| `--factors-ab=A:B` | In-process A/B of two DIF factorizations of the same N |

Every mode takes `--prec=f32|f64|both`, `--reps`, `--inner`, and most take
`--sizes=`/`--shapes=`. The remaining modes are development diagnostics
(`--pass`, `--codelet-sweep`, `--base-cost`, `--chain-sweep`, `--factor-sweep`,
`--model-dump`, `--decomp-report`, `--cost-audit`, `--route-ab-dif`,
`--fs-split-sweep`). A comment at the top of each block in `bench_fft.cpp`
documents that mode.

## Build options

- `ADM_BENCH_FFTW=ON` adds the FFTW reference arm. Needs system `fftw3` and
  `fftw3f`, found through pkg-config.
- `ADM_BENCH_THREADS=ON` threads the reference libraries too (ducc0's pool, and
  FFTW's `fftw3_threads` companions under `--nthreads=N`).

Threaded scaling, same host, `--compare-nd --nthreads=t` (`fft_fwd_us`, min
across rounds): threading is flat up to ~25k elements (1.0-1.1×), pays from 32k
(32³: 2.9-4.2×), and ramps to 9-11× at 128³ with 16 threads. Rectangles split on
the innermost extent: 64×4096 reaches 6.9×, 4096×64 only 2.6-5.3×. The
`nthreads = 0` auto heuristic (`kAutoSerialElems` / `kAutoElemsPerThread` in
`thread_pool.hpp`) comes from this sweep.

## Measurement conventions

- Nanobench, min across repetitions; `--inner=0` lets it auto-tune the epoch.
- The metric is TSC cycles where available; `--nthreads > 1` forces wall clock
  (cycle counting is per-thread).
- Every timed size is accuracy-gated against a reference DFT before timing.
- Single-shot compares carry measurement-order frequency bias (cold turbo
  flatters the first arm), so A/B modes alternate arm order round by round and
  report the median plus the round-to-round spread; `--robust` adds an identity
  control. A delta inside the spread is a tie.
- FFTW arms plan with `FFTW_MEASURE`. `ADM_BENCH_FFTW_ESTIMATE=1` switches them
  to the heuristic plan. That leaves FFTW untuned and flatters Admiral, so it
  is not the default.

## The headline numbers in the README

Measured on a Xeon w5-3435X (16 physical cores, AVX-512, `-march=native`):

- ducc0 comparison (`c4dda23`): `--compare` over 61 1-D lengths and
  `--compare-nd` over 27 N-D shapes, both precisions. Ratios are their time over
  Admiral's, so above 1 Admiral is faster.
- FFTW comparison: `FFTW_MEASURE`, single thread, 19 1-D lengths and 9 N-D
  shapes, both precisions pooled.
