#!/bin/bash

# 设置环境变量（您需要在实际运行前设置这些值）
# export RAW_DATASET_PATH="your_dataset_path"
# export MODEL_PATH="your_model_path"
PROJECT_ROOT="/exp/share_302035829/hunyuan/holdenlu/project/I_COA"

# 检查必需的环境变量是否已设置
if [ -z "$RAW_DATASET_PATH" ] || [ -z "$MODEL_PATH" ]; then
    echo "错误：请先设置 RAW_DATASET_PATH 和 MODEL_PATH 环境变量"
    echo "示例："
    echo "export RAW_DATASET_PATH=\"your_dataset_path\""
    echo "export MODEL_PATH=\"your_model_path\""
    exit 1
fi

if [ ! -d "$PROJECT_ROOT" ]; then
    echo "错误：项目根目录不存在: $PROJECT_ROOT"
    exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 指定运行顺序
DIRECTORIES=(
  "baseline"
  "coa"
)

# 为每个目录定义运行顺序
declare -A RUN_ORDER=(
    ["baseline"]="gen_pred.py"
    ["coa"]="gen_coa.py eval_coa.py"
)

# 运行所有脚本
for dir in "${DIRECTORIES[@]}"; do
    echo "正在处理目录: $dir"
    cd "$dir" || { echo "无法进入目录 $dir"; exit 1; }

    for script in ${RUN_ORDER[$dir]}; do
        if [ -f "$script" ]; then
            echo "运行脚本: $script"
            python "$script" || { echo "脚本 $script 执行失败"; exit 1; }
        else
            echo "警告: 脚本 $script 不存在"
        fi
    done

    cd .. || { echo "无法返回上级目录"; exit 1; }
    echo "完成目录: $dir"
    echo "------------------------"
done

echo "所有脚本执行完成"
