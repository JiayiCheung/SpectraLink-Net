#!/bin/bash
#SBATCH -x paraai-n32-h-01-agent-1,paraai-n32-h-01-agent-4,paraai-n32-h-01-agent-8,paraai-n32-h-01-agent-16,paraai-n32-h-01-agent-17,paraai-n32-h-01-agent-25,paraai-n32-h-01-agent-27,paraai-n32-h-01-agent-28,paraai-n32-h-01-agent-29,paraai-n32-h-01-agent-30,paraai-n32-h-01-agent-31
#SBATCH --job-name=SLinkNet_train
#SBATCH --output=SLinkNet_train_%j.out
#SBATCH --error=SLinkNet_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00

module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x
source activate fsf

export CUDA_VISIBLE_DEVICES=0,1,


# 切换到项目目录
cd /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet

# 打印环境信息
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU Count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "============================================"

# 启动分布式训练
torchrun \
    --nproc_per_node=2 \
    scripts/train.py \
    --config=config/config.yaml

