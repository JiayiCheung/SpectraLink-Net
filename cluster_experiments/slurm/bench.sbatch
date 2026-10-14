#!/bin/bash
# Submit from the SpectraLink_v2 base directory so that the relative log paths resolve:  cd $HOME/run/CSUqx/SpectraLink_v2 && sbatch code/slurm/bench.sbatch
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=bench
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/12.1
source activate fsf
export SL_BASE=${SL_BASE:-$HOME/run/CSUqx/SpectraLink_v2}
export SL_ROOT=$SL_BASE SL_RUNS=$SL_BASE/runs SL_RAW=$HOME/run/CSUqx/DataBase/Task08_HepaticVessel
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 SL_LOAD_THREADS=8
cd $SL_BASE/code
echo "job $SLURM_JOB_ID node $SLURM_NODELIST start $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -2
bash run_bench.sh
echo "end $(date)"
