#!/bin/bash
# Train several arms CONCURRENTLY on the one allocated GPU (one process per arm).  Args: "ARMS" SEED STEPS WORKERS TAGSUF   (env MAXSEC = per-process time guard in seconds)
ARMS=$1; SEED=${2:-17}; STEPS=${3:-12000}; WORKERS=${4:-4}; SUF=${5:-_full}; MAXSEC=${MAXSEC:-80000}
pids=(); names=()
for arm in $ARMS; do
  python train_cluster.py --arm $arm --seed $SEED --tag ${arm}_s${SEED}${SUF} --split train --steps $STEPS --workers $WORKERS --max_seconds $MAXSEC >> $SL_ROOT/logs/train_${arm}_s${SEED}${SUF}.log 2>&1 &
  pids+=($!); names+=($arm); sleep 5
done
worst=0
for i in "${!pids[@]}"; do wait ${pids[$i]}; rc=$?; echo "arm ${names[$i]} exit $rc"; [ $rc -gt $worst ] && worst=$rc; done
echo "worst exit $worst (0 done, 2 NaN stop, 3 time guard -> resubmit to resume)"; exit $worst
