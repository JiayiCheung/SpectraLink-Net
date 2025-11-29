#!/bin/bash
#SBATCH --job-name=SLinkNet_preprocess
#SBATCH --output=SLinkNet_preprocess_%j.out
#SBATCH --error=SLinkNet_preprocess_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

# ============================================
# SpectraLink-Net 数据预处理作业脚本
# 包含两个阶段：裁剪 + 重采样归一化
# ============================================

# 清理并加载模块
module purge
module load miniforge3/24.1 compilers/gcc/13.2.0 compilers/cuda/11.8 cudnn/8.8.1.3_cuda11.x

# 激活conda环境
source activate vessel

# 切换到项目目录
cd /home/bingxing2/home/scx7776/run/CSUqx/SLinkNet

# 打印环境信息
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "============================================"

# 定义路径
RAW_IMAGES_DIR="/home/bingxing2/home/scx7776/run/CSUqx/DataBase/Task08_HepaticVessel/imagesTr"
RAW_LABELS_DIR="/home/bingxing2/home/scx7776/run/CSUqx/DataBase/Task08_HepaticVessel/labelsTr"
PREPROCESS_DIR="/home/bingxing2/home/scx7776/run/CSUqx/SLinkNet/data/preprocess"

# 创建预处理目录
mkdir -p $PREPROCESS_DIR

echo "============================================"
echo "Stage 1: 非零区域裁剪"
echo "============================================"

python -m data.crop \
    --images_dir $RAW_IMAGES_DIR \
    --labels_dir $RAW_LABELS_DIR \
    --output_dir $PREPROCESS_DIR \
    --num_workers 8

echo "============================================"
echo "Stage 2: 重采样和归一化"
echo "============================================"

python -m data.preprocess \
    --cropped_dir $PREPROCESS_DIR/cropped \
    --output_dir $PREPROCESS_DIR \
    --target_spacing 1.0 1.0 1.0 \
    --num_workers 8

echo "============================================"
echo "预处理完成!"
echo "End Time: $(date)"
echo "============================================"

# 显示生成的文件
echo "生成的文件:"
ls -la $PREPROCESS_DIR/
echo ""
echo "处理后的数据:"
ls -la $PREPROCESS_DIR/processed/data/ | head -20
echo ""
echo "数据集划分:"
cat $PREPROCESS_DIR/splits.json
