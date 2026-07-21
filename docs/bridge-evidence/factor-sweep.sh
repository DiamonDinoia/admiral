#!/bin/bash
# Fusion-aware factor-chain re-sweep, band B. A=current DP chain, B=candidate
# with fewer twiddle layers / fewer sweeps under the fused2/fused3 gates.
set -u
B=/home/marco/repos/fft/build-fftw/benchmark/fft_benchmark
S=/tmp/claude-1000/-home-marco-repos-fft/4c09dd1e-8500-4feb-868c-c0bce5897020/scratchpad
out="$S/factor-sweep.txt"; : > "$out"
run() { timeout 600 taskset -c 2 $B --factors-ab="$1" --prec="$2" --rounds=15 >> "$out" 2>&1; }
run 4-4-4-4-4-8:4-4-8-8-8         f64   # 8192
run 4-4-4-4-8-8:4-8-8-8-8         f64   # 16384
run 4-4-4-4-4-4-8:8-8-8-8-8       f64   # 32768
run 4-4-4-4-4-4-8:4-4-4-8-8-8     f64   # 32768 alt
run 4-4-4-4-4-8-8:4-4-8-8-8-8     f64   # 65536
run 4-4-4-4-4-8:4-4-8-8-8         f32   # 8192
run 4-4-4-4-8-8:4-8-8-8-8         f32   # 16384
echo SWEEP-DONE
cat "$out"
