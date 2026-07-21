#!/bin/bash
# Step-0 attribution: instructions + cycles per transform, ours vs FFTW MEASURE.
# Run ONLY after core 2 is free. Planning excluded: ours is cheap; FFTW via
# wisdom warm-up run before the measured run.
set -u
S=/tmp/claude-1000/-home-marco-repos-fft/4c09dd1e-8500-4feb-868c-c0bce5897020/scratchpad
EV="cpu_core/instructions/,cpu_core/cycles/"
out="$S/attribution.txt"; : > "$out"
for prec in d f; do
  for n in 8192 32768; do
    reps=$(( n == 8192 ? 200000 : 50000 ))
    wf="$S/wis_${prec}_${n}"
    timeout 300 taskset -c 2 "$S"/fftw_perf "$n" 1 "$prec" "$wf" >/dev/null 2>&1  # warm wisdom
    echo "== N=$n prec=$prec reps=$reps ==" >> "$out"
    timeout 600 taskset -c 2 perf stat -e "$EV" -x, -o /dev/stdout \
      "$S"/ours_perf "$n" "$reps" "$prec" 2>/dev/null | grep -E "instructions|cycles" \
      | awk -F, -v r="$reps" '{printf "ours %s %.0f/xf\n", $3, $1/r}' >> "$out"
    timeout 600 taskset -c 2 perf stat -e "$EV" -x, -o /dev/stdout \
      "$S"/fftw_perf "$n" "$reps" "$prec" "$wf" 2>/dev/null | grep -E "instructions|cycles" \
      | awk -F, -v r="$reps" '{printf "fftw %s %.0f/xf\n", $3, $1/r}' >> "$out"
  done
done
echo ATTRIBUTION-DONE
cat "$out"
