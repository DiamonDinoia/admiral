#!/bin/bash
# Follow-up: loads/stores (spill+twiddle traffic) and L2/L3 miss rates.
set -u
S=/tmp/claude-1000/-home-marco-repos-fft/4c09dd1e-8500-4feb-868c-c0bce5897020/scratchpad
EV="cpu_core/mem_inst_retired.all_loads/,cpu_core/mem_inst_retired.all_stores/,cpu_core/mem_load_retired.l2_miss/,cpu_core/mem_load_retired.l3_miss/"
out="$S/attribution2.txt"; : > "$out"
for n in 8192 32768; do
  reps=$(( n == 8192 ? 200000 : 50000 ))
  wf="$S/wis_d_${n}"
  echo "== N=$n prec=d reps=$reps ==" >> "$out"
  for eng in ours fftw; do
    if [ "$eng" = ours ]; then cmd=("$S"/ours_perf "$n" "$reps" d); else cmd=("$S"/fftw_perf "$n" "$reps" d "$wf"); fi
    timeout 600 taskset -c 2 perf stat -e "$EV" -x, -o /dev/stdout "${cmd[@]}" 2>/dev/null \
      | awk -F, -v r="$reps" -v e="$eng" '$3 != "" {printf "%s %s %.0f/xf\n", e, $3, $1/r}' >> "$out"
  done
done
echo ATTR2-DONE
cat "$out"
