#!/bin/bash

# 设置实验参数
seeds=(42 123 0 2025 7 100 999 314 73 66)
max_processes=3

# 创建日志目录
Path_prefix="train_result"
env_name="Walker2d-v2"
exp_name="Dynamic weights"
log_dir="script_logs"
mkdir -p "${Path_prefix}/${exp_name}/${env_name}/${log_dir}"

# 计数器
count=0

for seed in "${seeds[@]}"; do
    # 检查当前运行的进程数，如果达到最大值则等待
    while (( $(jobs -p | wc -l) >= max_processes )); do
        sleep 5
    done
    
    # 为每个实验创建独立的日志文件
    log_file="${Path_prefix}/${exp_name}/${env_name}/${log_dir}/experiment_s${seed}.log"
    
    # 启动实验并将输出重定向到日志文件
    nohup python main.py --seed=$seed > "$log_file" 2>&1 &
    
    echo "启动实验 $((++count)): seed=$seed"
    
    # 短暂延迟以避免同时启动
    sleep 2
done

# 等待所有实验完成
wait
echo "所有实验已完成" 