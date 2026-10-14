# Cluster experiments (candidate-head study) — code only

Branch `cluster-experiments`; nothing here touches `main`. Scripts for the n20 pilot and its cluster-scale validation:
`prep_cluster.py` (ROI + resample cache, CPU) -> `make_splits.py` (203 train / 40 dev / 60 sealed test) -> `train_cluster.py` / `run_arms.sh` (several arms share one GPU)
-> `infer_cluster.py` (GPU inference + whole-ROI recall budget) -> `metrics_cluster.py` (CPU metrics) -> `analyze_n20.py` (paired bootstrap).
`pilot.py` holds the network arms (A0 spatial baseline, A6 + candidate head, D0 matched generic auxiliary loss, A7 BCE-only, C1/C2 early heads, F0/F1 mixers, R2 routed decoder).
Paths are taken from `SL_ROOT`, `SL_RUNS`, `SL_RAW` or `$HOME`; submit the Slurm scripts from the base directory (relative log paths).
