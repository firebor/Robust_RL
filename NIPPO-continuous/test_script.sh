#!/bin/bash

# NIPPO算法鲁棒性测试脚本
# 在运行此脚本之前，请确保已经训练好了模型并提取了最佳模型

# 设置测试参数
seeds=(42 123 0 2025 7 100 999 314 73 66)
exp_name="PPO-continuous"  # 实验名称，需要与训练时的exp_name一致
# env_names=("Walker2d-v2" "Hopper-v2" "HalfCheetah-v2" "InvertedDoublePendulum-v2")
env_names=("Walker2d-v2")

# 创建测试结果目录
test_result_dir="test_result"
mkdir -p "${test_result_dir}"

echo "开始NIPPO算法鲁棒性测试..."
echo "实验名称: ${exp_name}"
echo "测试环境: ${env_names[@]}"
echo "测试种子: ${seeds[@]}"
echo ""

# 检查并提取最佳模型
echo "检查最佳模型..."
for env_name in "${env_names[@]}"; do
    best_model_dir="./best_model/${exp_name}/${env_name}"
    if [ ! -d "$best_model_dir" ]; then
        echo "警告：${env_name} 的最佳模型不存在，正在提取..."
        python get_best_model.py --env_name="$env_name" --exp_name="$exp_name"
    else
        echo "✓ ${env_name} 的最佳模型已存在"
    fi
done
echo ""

# 计数器
count=0

for env_name in "${env_names[@]}"; do
    for seed in "${seeds[@]}"; do
        echo "===================="
        echo "开始测试 $((++count)): ${env_name} seed=${seed}"
        echo "===================="
        
        # 检查最佳模型文件是否存在
        model_dir="./best_model/${exp_name}/${env_name}"
        if [ ! -d "$model_dir" ]; then
            echo "警告：最佳模型目录不存在 $model_dir"
            echo "请先运行 get_best_model.py 提取最佳模型"
            continue
        fi
        
        # 运行测试
        python main_test_multi_cross.py \
            --env_name="$env_name" \
            --exp_name="$exp_name" \
            --seed=$seed
        
        if [ $? -eq 0 ]; then
            echo "✓ 测试完成: ${env_name} seed=${seed}"
        else
            echo "✗ 测试失败: ${env_name} seed=${seed}"
        fi
        
        echo ""
    done
done

echo "===================="
echo "所有测试完成！"
echo "结果保存在: ${test_result_dir}/${exp_name}_cross/"
echo "====================" 