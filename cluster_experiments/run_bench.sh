#!/bin/bash
# GPU speed benchmark on the real data path (24 cached cases, 300 steps per process).  Outputs go to node-local /tmp (auto-cleaned), only the summary stays in logs/.
export SL_RUNS=${TMPDIR:-/tmp}/slbench_$SLURM_JOB_ID; mkdir -p $SL_RUNS
SUM=$SL_ROOT/logs/bench_summary_$SLURM_JOB_ID.txt; echo "benchmark $(date)" > $SUM; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader >> $SUM 2>&1
bench() { name=$1; workers=$2; shift 2; arms="$@"; pids=(); t0=$(date +%s)
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -l 3 > $SL_RUNS/util_$name.csv 2>/dev/null & mon=$!
  for arm in $arms; do python train_cluster.py --arm $arm --seed 1 --tag b_${name}_$arm --split train --limit_cases 24 --steps 300 --workers $workers --log_every 50 --ckpt_every 100000 > $SL_RUNS/log_${name}_$arm.txt 2>&1 & pids+=($!); done
  for p in "${pids[@]}"; do wait $p; done; kill $mon 2>/dev/null; t1=$(date +%s)
  echo "== $name: arms [$arms] workers/proc $workers wall $((t1-t0)) s (includes ~data loading)" >> $SUM
  for arm in $arms; do echo "   $arm: $(grep -E 'it/s' $SL_RUNS/log_${name}_$arm.txt | tail -3 | sed -E 's/.*lr [0-9.e+-]+ //' | tr '\n' '|')" >> $SUM; done
  echo "   GPU util/mem samples (mean util): $(awk -F', ' '{gsub(/ %/,"",$1); s+=$1; n++} END {if (n) printf "%.0f%% over %d samples", s/n, n}' $SL_RUNS/util_$name.csv)  last mem: $(tail -1 $SL_RUNS/util_$name.csv)" >> $SUM; }
bench 1proc_w4 4 A6
bench 1proc_w8 8 A6
bench 3procs_w4 4 A0 A6 D0
bench 4procs_w3 3 A0 A6 D0 A7
bench 6procs_w2 2 A0 A6 D0 A7 C2 F1
cat $SUM
