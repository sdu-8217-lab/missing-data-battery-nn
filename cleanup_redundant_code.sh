#!/bin/bash
# 删除冗余代码脚本
# 执行前请确认已备份重要代码

echo "================================================"
echo "删除冗余代码 - 基于 Hydra + Lightning 的简化架构"
echo "================================================"
echo ""

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}准备删除以下过度设计的代码：${NC}"
echo ""

echo "1. src/core/ - 自己实现的注册中心（Hydra 已提供）"
echo "   - src/core/__init__.py"
echo "   - src/core/interfaces.py"
echo "   - src/core/registry.py"
echo ""

echo "2. src/missing/imputers/ - 过度拆分的插补器"
echo "   - base.py, zero.py, mean.py, knn.py, iterative.py"
echo "   合并为：src/missing/missing_data.py"
echo ""

echo "3. 其他冗余文件"
echo "   - src/missing/imputation_utils.py"
echo "   - src/config/pydantic_config.py（用 Hydra 替代）"
echo "   - src/config/loader.py"
echo ""

echo -e "${GREEN}保留并推广的简化版代码：${NC}"
echo "   ✅ src/missing/missing_data_simple.py → src/missing/missing_data.py"
echo "   ✅ src/models/factory_simple.py → src/models/factory.py"
echo "   ✅ src/experiments/simple_runner.py（新建）"
echo ""

read -p "确认删除？ (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo -e "${YELLOW}执行删除...${NC}"
    
    # 备份
    BACKUP_DIR=".cleanup_backup_$(date +%Y%m%d)"
    mkdir -p $BACKUP_DIR
    
    if [ -d "src/core" ]; then
        cp -r src/core $BACKUP_DIR/
        rm -rf src/core/
        echo -e "${GREEN}✓ 删除 src/core/（已备份）${NC}"
    fi
    
    if [ -d "src/missing/imputers" ]; then
        cp -r src/missing/imputers $BACKUP_DIR/
        rm -rf src/missing/imputers/
        echo -e "${GREEN}✓ 删除 src/missing/imputers/（已备份）${NC}"
    fi
    
    if [ -f "src/missing/imputation_utils.py" ]; then
        cp src/missing/imputation_utils.py $BACKUP_DIR/
        rm -f src/missing/imputation_utils.py
        echo -e "${GREEN}✓ 删除 src/missing/imputation_utils.py（已备份）${NC}"
    fi
    
    if [ -f "src/config/pydantic_config.py" ]; then
        cp src/config/pydantic_config.py $BACKUP_DIR/
        rm -f src/config/pydantic_config.py
        echo -e "${GREEN}✓ 删除 src/config/pydantic_config.py（已备份）${NC}"
    fi
    
    if [ -f "src/config/loader.py" ]; then
        cp src/config/loader.py $BACKUP_DIR/
        rm -f src/config/loader.py
        echo -e "${GREEN}✓ 删除 src/config/loader.py（已备份）${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}================================================${NC}"
    echo -e "${GREEN}冗余代码已删除！备份位置：$BACKUP_DIR/${NC}"
    echo -e "${GREEN}================================================${NC}"
    echo ""
    echo "下一步："
    echo "  1. 移动简化版代码到正式位置："
    echo "     mv src/missing/missing_data_simple.py src/missing/missing_data.py"
    echo "     mv src/models/factory_simple.py src/models/factory.py"
    echo ""
    echo "  2. 测试简化版："
    echo "     python tests/test_simplified.py"
    echo ""
    echo "  3. 使用 Hydra + Lightning 运行实验："
    echo "     python src/experiments/simple_runner.py"
    
else
    echo -e "${YELLOW}已取消删除${NC}"
    exit 0
fi
