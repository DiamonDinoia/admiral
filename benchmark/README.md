# Benchmarks
`admiral_benchmark` times Admiral against ducc0 with nanobench, and against FFTW when built with
`-DADM_BENCH_FFTW=ON`. The binary is not installed.

## Build and run
```bash
cmake --preset release          # benchmarks build in this preset
cmake --build --preset release
./build/release/benchmark/admiral_benchmark --compare       # 1-D vs ducc0
./build/release/benchmark/admiral_benchmark --compare-nd    # N-D vs ducc0
```

| Mode | What it reports |
|------|-----------------|
| `--compare` | 1-D ratio vs ducc0 (and FFTW), per size, both precisions |
| `--compare-nd` / `--compare-2d` | The same for N-D shapes; `--r2c` switches to real transforms |
| `--verify` | Correctness sweep only, nonzero exit on failure |
| `--fftw-ab` | Interleaved Admiral vs FFTW A/B (needs `-DADM_BENCH_FFTW=ON`) |
| `--size=N` | Single-size profiling of the library alone |
| `--factors-ab=A:B` | In-process A/B of two DIF factorizations of the same N |

Every mode takes `--prec=f32|f64|both`, `--reps`, and `--inner`, and most take `--sizes=`/`--shapes=`. The
remaining modes (`--pass`, `--codelet-sweep`, `--base-cost`, `--chain-sweep`, `--factor-sweep`,
`--model-dump`, `--decomp-report`, `--cost-audit`, `--route-ab-dif`, `--fs-split-sweep`) are development
diagnostics. Each block in `bench_fft.cpp` documents its mode in a comment.

`ADM_BENCH_FFTW=ON` adds the FFTW reference arm. It needs the system `fftw3` and `fftw3f`, found through
pkg-config. `ADM_BENCH_THREADS=ON` threads the reference libraries too (ducc0's pool, and FFTW's
`fftw3_threads` companions under `--nthreads=N`).

## Threaded scaling
Measured with `--compare-nd --nthreads=<t>` at t = 1, 2, 4, 8, 16 (`fft_fwd_us`, min across rounds). A figure is
Admiral's single-thread time over its threaded time.

- Shapes under ~16k elements run slower at 16 threads (0.7-1.0×). Threading pays from about 32k elements:
  4.2-6.4× at 32³, 2.6-3.5× at 192².
- Large shapes reach 6-7× (512²), 8-9× (64³), and 10-12× (128³, 256³-scale, 509²).
- Rectangles split on the innermost extent: 64×4096 reaches 9×, 4096×64 only 2×.

1-D is the one late case: the threaded route admits a size only past its byte budget line (~512 KiB at 16
threads), so below ~32k elements a 1-D plan stays serial. Past that line, 16 threads split the wins with
threaded ducc0, which threads 1-D only from ~1M elements (`--compare --nthreads=16 --adm-nthreads=16`, geomean
of ducc0 over Admiral):

| N | f64 | f32 |
|---|-----|-----|
| 1M | 0.82× | 1.36× |
| 2M | 0.33× | 0.42× |
| 4M | 1.56× | 0.90× |
| 8M | 0.36× | 0.36× |
| 16M | 1.82× | 1.72× |

The losing rows are the six-step admission check (`n2 % n1`) rejecting a threaded split, so the size falls to
the serial DIF chain while ducc0 threads. Divisible splits engage the threaded route and win.

## Measurement conventions
- Nanobench, min across repetitions. `--inner=0` lets it auto-tune the epoch.
- The metric is TSC cycles where available. `--nthreads > 1` forces wall clock (cycle counting is per-thread).
- Every timed size passes an accuracy gate against a reference DFT before timing.
- A/B modes alternate the arm order round by round against the measurement-order frequency bias (cold turbo
  flatters the first arm), and report the median plus the round-to-round spread. A delta inside the spread is
  a tie. `--robust` adds an identity control.
- FFTW arms always plan with `FFTW_MEASURE`, with no wisdom caching or planner time limit, tuned the same as
  Admiral's default.

The headline numbers in the README come from the 2026-08-24 sweep on the host named there, GCC 14.2: ducc0
(`c4dda23`) `--compare` over 61 1-D lengths and `--compare-nd` over 27 N-D shapes, and FFTW `FFTW_MEASURE`
over 19 1-D lengths and 9 N-D shapes, all at both precisions.
