#!/bin/bash
#SBATCH --job-name=SLinkNet_vis3d
#SBATCH --output=SLinkNet_vis3d_%j.out
#SBATCH --error=SLinkNet_vis3d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

# ============================================
# 3D血管可视化 - PyVista渲染
# ============================================

module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/12.1
source activate fsf

export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到项目目录
cd /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet



# 设置离屏渲染环境变量
export PYVISTA_OFF_SCREEN=true
export DISPLAY=:99



echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "============================================"

# 定义路径
PRED_DIR="/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/validation/predictions"
GT_DIR="/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess/processed/data"
OUTPUT_DIR="/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/outputs/visualization_3d"


# 创建输出目录
mkdir -p $OUTPUT_DIR


python scripts/3d.py \
    --pred_dir $PRED_DIR \
    --gt_dir $GT_DIR \
    --output_dir $OUTPUT_DIR \
    --max_cases 60

echo "============================================"
echo "完成时间: $(date)"
echo "结果保存在: $OUTPUT_DIR"
echo "============================================"
