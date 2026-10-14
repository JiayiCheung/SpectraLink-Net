#!/bin/bash
# Inference for several finished tags concurrently on one GPU.  Args: "TAGS" SPLIT
TAGS=$1; SPLIT=${2:-dev}; pids=()
for t in $TAGS; do python infer_cluster.py --tag $t --split $SPLIT >> $SL_ROOT/logs/infer_${t}_${SPLIT}.log 2>&1 & pids+=($!); sleep 3; done
worst=0; for p in "${pids[@]}"; do wait $p; rc=$?; [ $rc -gt $worst ] && worst=$rc; done; echo "worst exit $worst"; exit $worst
