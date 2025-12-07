#!/bin/bash
# ============================================
# SpectraLink-Net 代码同步脚本
# 用法: ./sync.sh
# ============================================

# ========== 配置区域（请修改为你的信息）==========
REMOTE_USER="your_username"           # 并行云用户名
REMOTE_HOST="login.paratera.com"      # 并行云登录节点（根据实际修改）
REMOTE_PATH="~/SpectraLink-Net"       # 远程目标路径
SSH_PORT="22"                         # SSH 端口（并行云可能不同）
# ================================================

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}  SpectraLink-Net 代码同步工具${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo -e "本地路径: ${GREEN}${SCRIPT_DIR}${NC}"
echo -e "远程路径: ${GREEN}${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}${NC}"
echo ""

# 检查配置
if [ "$REMOTE_USER" = "your_username" ]; then
    echo -e "${RED}错误: 请先编辑 sync.sh，配置你的并行云账号信息！${NC}"
    echo ""
    echo "需要修改的配置项："
    echo "  REMOTE_USER - 你的用户名"
    echo "  REMOTE_HOST - 登录节点地址"
    echo "  REMOTE_PATH - 远程目标目录"
    exit 1
fi

# 排除不需要同步的文件
EXCLUDES=(
    ".git"
    "__pycache__"
    "*.pyc"
    ".DS_Store"
    "outputs"
    "checkpoints"
    "logs"
    "*.pth"
    "*.pt"
    "wandb"
    ".venv"
    "venv"
    "data/raw"        # 原始数据通常已在服务器上
    "data/processed"  # 处理后的数据
)

# 构建 rsync 排除参数
EXCLUDE_ARGS=""
for item in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$item"
done

# 先做 dry-run 预览
echo -e "${YELLOW}正在预览将要同步的文件...${NC}"
echo ""

rsync -avzn --delete \
    -e "ssh -p ${SSH_PORT}" \
    $EXCLUDE_ARGS \
    "${SCRIPT_DIR}/" \
    "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

echo ""
echo -e "${YELLOW}----------------------------------------${NC}"
read -p "确认同步以上文件? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo -e "${GREEN}开始同步...${NC}"

    rsync -avz --delete --progress \
        -e "ssh -p ${SSH_PORT}" \
        $EXCLUDE_ARGS \
        "${SCRIPT_DIR}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ 同步完成！${NC}"
        echo ""
        echo -e "现在可以登录服务器运行训练："
        echo -e "  ${YELLOW}ssh -p ${SSH_PORT} ${REMOTE_USER}@${REMOTE_HOST}${NC}"
        echo -e "  ${YELLOW}cd ${REMOTE_PATH} && python scripts/train.py${NC}"
    else
        echo -e "${RED}✗ 同步失败，请检查网络连接和配置${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}已取消同步${NC}"
fi
