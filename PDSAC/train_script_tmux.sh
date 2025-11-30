#!/bin/bash

# 设置实验参数
seeds=(42 123 0 2025 7 100 999 314 73 66)
noises=(0.2 0.1 0.15)
max_processes=3

# 创建日志目录
Path_prefix="train_result"
env_name="Walker2d-v2"
exp_name="grad-norm"  # 添加实验名称参数
log_dir="script_logs"
mkdir -p "${Path_prefix}/${exp_name}/${env_name}/${log_dir}"

# 计数器
count=0

# 创建tmux会话
tmux new-session -d -s experiments

for noise in "${noises[@]}"; do
    for seed in "${seeds[@]}"; do
        # 检查当前运行的进程数
        while (( $(tmux list-panes -t experiments | wc -l) >= max_processes )); do
            sleep 5
        done
        
        # 为每个实验创建独立的日志文件
        log_file="${Path_prefix}/${exp_name}/${env_name}/${log_dir}/experiment_s${seed}_n${noise}.log"
        
        # 创建新的tmux窗口并运行实验，先激活虚拟环境
        tmux new-window -t experiments -n "exp_${count}" "conda activate RobustRL && python main.py --seed=$seed --noise_multiplier=$noise --env_name=$env_name --exp_name=$exp_name > $log_file 2>&1"
        
        echo "启动实验 $((++count)): seed=$seed, noise=$noise"
        
        # 短暂延迟以避免同时启动
        sleep 2
    done
done

# 等待所有窗口完成
while (( $(tmux list-panes -t experiments | wc -l) > 1 )); do
    sleep 5
done

# 关闭tmux会话
tmux kill-session -t experiments

echo "所有实验已完成" 