#!/bin/bash
# Stage A: topdown classification + flop counting + TLB, ours vs FFTW, f64+f32 32768 & 8192.
set -u
S=/tmp/claude-1000/-home-marco-repos-fft/4c09dd1e-8500-4feb-868c-c0bce5897020/scratchpad
out="$S/attribution3.txt"; : > "$out"
run_ev() { # engine cmd... ; uses $EV $label $reps
  timeout 600 taskset -c 2 perf stat -e "$EV" -x, -o /dev/stdout "$@" 2>/dev/null \
    | awk -F, -v r="$reps" -v e="$label" '$1+0>0 {printf "%s %s %.1f/xf\n", e, $3, $1/r}' >> "$out"
}
for prec in d f; do
  for n in 8192 32768; do
    reps=$(( n == 8192 ? 200000 : 50000 ))
    wf="$S/wis_${prec}_${n}"
    [ -f "$wf" ] || timeout 300 taskset -c 2 "$S"/fftw_perf "$n" 1 "$prec" "$wf" >/dev/null 2>&1
    echo "== N=$n prec=$prec ==" >> "$out"
    # Topdown L1 (slots-based)
    EV="cpu_core/slots/,cpu_core/topdown-retiring/,cpu_core/topdown-bad-spec/,cpu_core/topdown-fe-bound/,cpu_core/topdown-be-bound/"
    label="ours-td"; run_ev "$S"/ours_perf "$n" "$reps" "$prec"
    label="fftw-td"; run_ev "$S"/fftw_perf "$n" "$reps" "$prec" "$wf"
    # FP arithmetic: packed vector ops by width (counts real flops)
    EV="cpu_core/fp_arith_inst_retired.256b_packed_double/,cpu_core/fp_arith_inst_retired.256b_packed_single/,cpu_core/fp_arith_inst_retired.128b_packed_double/,cpu_core/fp_arith_inst_retired.128b_packed_single/,cpu_core/fp_arith_inst_retired.scalar_double/,cpu_core/fp_arith_inst_retired.scalar_single/"
    label="ours-fp"; run_ev "$S"/ours_perf "$n" "$reps" "$prec"
    label="fftw-fp"; run_ev "$S"/fftw_perf "$n" "$reps" "$prec" "$wf"
    # TLB + L1D pressure
    EV="cpu_core/dtlb_load_misses.walk_completed/,cpu_core/mem_load_retired.l1_miss/,cpu_core/mem_load_retired.fb_hit/,cpu_core/mem_load_retired.l2_hit/"
    label="ours-mem"; run_ev "$S"/ours_perf "$n" "$reps" "$prec"
    label="fftw-mem"; run_ev "$S"/fftw_perf "$n" "$reps" "$prec" "$wf"
  done
done
echo ATTR3-DONE
cat "$out"
