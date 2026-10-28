# Benchmarks
`admiral_benchmark` times Admiral against ducc0, and against FFTW wherever pkg-config finds system `fftw3` and `fftw3f` at configure time (the `--fftw-ab` mode). It is part of the `release` preset.

```bash
cmake --preset release
cmake --build --preset release
./build/release/benchmark/admiral_benchmark --compare      # 1-D against ducc0
./build/release/benchmark/admiral_benchmark --compare-nd   # N-D against ducc0
./build/release/benchmark/admiral_benchmark --verify       # correctness sweep
```

Most modes use `--prec=f32|f64|both`, `--reps`, `--inner`, `--sizes=` and `--shapes=`. Other modes: `--fftw-ab` (needs system fftw3/fftw3f, auto-detected at configure), `--factors-ab=A:B`, `--size=N`, `--pass=IP,ido,l1` and `--r2c` with `--compare-nd`. `--nthreads=N` sets the thread count; `--compare` takes `--adm-nthreads=N` for admiral. The ducc0 and FFTW references always thread.
