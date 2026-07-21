# Development tools & measurement discipline

Working notes for the tools used to develop/tune yafft on this machine
(Sapphire Rapids w5-3435X, Flatiron workstation). Companion to
`docs/wide-radix-planner.md` (results) — this file is the *how*.

## Per-ISA builds

Four build trees, one per `-march`. gcc ICEs on the AVX-512 codelets, so
everything uses clang-18:

```bash
CLANG=/mnt/sw/nix/store/hdrrg483r5pi6lyravq8fd8idkrd8mc3-llvm-18.1.8/bin/clang++
cmake -B build/v4 -G Ninja -DCMAKE_CXX_COMPILER=$CLANG \
      -DFFT_USE_NATIVE_ARCH=OFF -DCMAKE_CXX_FLAGS="-march=x86-64-v4" \
      -DCMAKE_BUILD_TYPE=Release          # same for v2/v3 with their -march
cmake -B build/native -G Ninja -DCMAKE_CXX_COMPILER=$CLANG   # native: builds tests too
env -u NINJA_STATUS cmake --build build/v4 -j 16
```

- `env -u NINJA_STATUS` is required: the shell's `NINJA_STATUS='%w'` breaks ninja.
- Only `build/native` builds the test suite (114 tests, `ctest --test-dir build/native`).
- v2/v3 binaries run natively on this machine (subset ISA is executable) —
  per-ISA perf comparisons don't need another host.
- Deps are pinned via CPM into `~/.cpm/` (xsimd fork `a2911742` on
  DiamonDinoia/xsimd branch `yafft-feature`, poet `ba54c9ec`).

## Benchmark single-unit drivers (`build/<lvl>/benchmark/bench_fft`)

| mode | what it measures | when to trust it |
|---|---|---|
| `--pass` | one isolated dif_pass (radix, ido, l1), pinned, reps=9 min | L2-resident spans: always. Streaming spans: only vs an interleaved control |
| `--factors-ab` | two factor chains, role-swapped interleaved A/B (ABCMP columns) | the ONLY trustworthy chain comparison at N ≥ 8192 |
| `--compare-nd --robust` | full transform vs ducc0/FFTW, role-swapped with identity gate (ABND) | the ONLY trustworthy library ratio at N ≥ 8192 |
| `--factor-sweep` | exhaustive chain sweep for one N | ranking within one run; absolute numbers drift |
| `--verify --prec=both` | correctness vs reference | always (run at all ISA levels) |

## Measurement discipline (hard-won — violating these produced wrong verdicts)

1. **N ≥ 8192: only role-swapped interleaved A/B counts.** Plain X/ducc anchor
   ratios swing up to 60% across runs (core frequency bimodality). The
   pre-campaign "sole loss f64 8192 = 1.17–1.35" was partly this artifact.
2. **Streaming (>L2 working set) benchmarks NEVER run concurrently** — DRAM/L3
   contention silently doubles some radices and shifts controls. Interleave on
   one pinned core or discard the run.
3. **L2-resident microbenches CAN run in parallel**, one per physical core
   (`taskset -c <n>`), which is how the 72-row calibration matrices were built
   quickly.
4. **Serial runs hours apart are not comparable** (frequency state). A/B means
   both binaries alternating in one process/session.
5. **Instruction count is ground truth for small deltas** (`perf stat`,
   `simdref annotate`); cycles only via min-of-N.
6. `bench-results/` keeps the raw outputs; every claim in the docs cites a file
   there.

## Calibration → fit → verify loop (the WS2 planner cost model)

1. `./scratch_calibrate.sh <lvl>` → `bench-results/calib_<lvl>.txt`: isolated
   `--pass` rows per (radix ∈ {2,3,4,5,7,8,11,16,32} × regime ∈ {vec, valley,
   last}), pinned, L2-resident spans. Footprint variants (128KB/1MB/4MB
   per-side) → `calib_v4_footprint.txt`.
2. Numbers go into `dif_measured_cost` tables in
   `include/fft/detail/twiddles.hpp`, keyed `(sizeof(T), W, regs)` at compile
   time; footprint anchors into `dif_footprint_mult`; the fusion discount γ
   into `dif_fuse_discount`, fitted against role-swapped pairwise chain A/Bs
   only (never against model-predicted numbers).
3. **Propose→verify:** the model's argmin chain for each N is A/B'd against the
   incumbent with `--factors-ab` (`./scratch_wide_ab.sh`). A win becomes the
   new best; a loss becomes a new fit constraint. Iterate.
4. Uncovered `(W, regs)` keys (SVE, RVV, AVX10/256, …) fall back to
   `dif_analytical_cost` — physically-derived traffic/compute/spill terms,
   constants fitted on the six measured x86 tables. It is ±35% absolute (NOT
   good enough as primary; 5/13 ranking violations when tried), but preserves
   the compile-time-knowable structure (register spill walls, lane waste).

## dp_probe — what chain does the planner actually pick?

Standalone harness printing `build_dif_factor_plan<T>(N)` chains without a full
build (sources in the session scratchpad; recreate in ~20 lines: include
`twiddles.hpp`, print plans for a size list):

```bash
$CLANG -std=c++20 -O2 -march=x86-64-v4 -ffast-math -w \
  -I include -I build/v4/src/generated/include \
  -isystem ~/.cpm/poet/<hash>/include -isystem ~/.cpm/xsimd/<hash>/include \
  dp_probe.cpp -o dp_probe_v4
```

Run after every cost-table change; diff the chains before benchmarking anything.

## Asm / PMU pipeline (per the asm-analysis rule)

- Standalone unit: `$CLANG -S -O3 -march=x86-64-v<N> unit.cpp` or
  `objdump -d --disassemble=<symbol>` on the built per-ISA binary.
- `simdref annotate` — per-instruction latency/CPI, spill classification
  (zmm-hot vs xmm-cold), shuffle-port pressure.
- `perf stat -C <core>` pinned via `taskset`: instructions, cycles, IPC,
  `fp_arith_inst_retired.512b_packed_*` (width utilization),
  `ld_blocks.address_alias` (4K aliasing — the r4 streaming signature).
- `llvm-mca` for throughput models of prototype kernels before integration.

## Scratch scripts (repo root, not committed artifacts)

- `scratch_bench.sh <lvl> <tier>` — ratio-table generation vs ducc0/FFTW.
- `scratch_calibrate.sh <lvl>` — the calibration matrix above.
- `scratch_wide_ab.sh` — role-swapped chain A/B batches.
- `scratch_env.sh` — compiler/module env setup.

## Known traps

- IDE clangd diagnostics on the headers (`poet/poet.hpp not found`, etc.) are
  spurious — the IDE lacks the CPM include paths; the builds are the oracle.
- `git stash` to get a "plain" binary for A/B stashes ALL in-flight changes —
  toggle the gate with a one-line const instead.
- Frequency bimodality: prefer instruction counts; cycles only min-of-N pinned.
- Prefetch in dif_pass: do-not-retry (`bench-results/pf_ab_interleaved.txt`).
