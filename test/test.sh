#!/bin/bash

# 设置环境变量（您需要在实际运行前设置这些值）
export MODEL_PATH=/exp/share_302035829/hunyuan/holdenlu/ckpt
export RAW_DATASET_PATH=/exp/share_302035829/hunyuan/holdenlu/data/UnsafeBench/ori_data
export PYTHONPATH=/exp/share_302035829/hunyuan/holdenlu/project/I_COA:$PYTHONPATH

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
