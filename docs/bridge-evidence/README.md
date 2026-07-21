# Band-B bridge plan — evidence artifacts (2026-07-04)

Raw measurement artifacts backing `docs/bridge-plan.md` (session 4c09dd1e, pinned
core 2, master 25998d5). Copied from the session scratchpad before it expired.

- `v5-ab-bandB.txt` — headline interleaved `--fftw-ab` band-B ratios (rounds=15).
- `ab-{baseline,candidate}-bandB.txt` — staged-r16 NO-GO A/B.
- `attribution{,2,3}.{txt,sh}` — perf counters ours-vs-FFTW: insn/cycles,
  loads/stores/L2, topdown/fp_arith/TLB (scripts alongside outputs).
- `hot-{ours,fftw}.txt` — perf record hot-symbol tables at 32768.
- `f{64,32}_fused2_census.txt` — objdump instruction census of the fused2 loops.
- `asmwork/` — extracted fused2 hot loops (`.s`), simdref-annotated (`.sa`),
  and llvm-mca (alderlake) reports.
- `factor-sweep.{txt,sh}` — 7-point pow2 factor-order sweep; the W3 enumerator's
  α-calibration data (must reproduce f64 32768 {4,4,4,8,8,8} win + the non-flips).
