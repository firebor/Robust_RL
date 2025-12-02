#!/bin/bash

# Set experiment parameters
seeds=(42 123 0 2025 7 100 999 314 73 66)
# seeds=(42)
noises=(1.0)
# switch_steps=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200 210 220 230 240 250 260 270 280 290 300 310 320 330 340 350 360 370 380 390 400 410 420 430 440 450 460 470 480 490 500)
switch_steps=(70)
max_processes=3
exp_name="RLE"  # Experiment name
env_name="Walker2d-v2"     # Environment name

Path_prefix="train_result"
log_dir="script_logs"
mkdir -p "${Path_prefix}/${exp_name}/${env_name}/${log_dir}"

# 计数器
count=0

for noise in "${noises[@]}"; do
    for seed in "${seeds[@]}"; do
        for switch_step in "${switch_steps[@]}"; do
            # 检查当前运行的进程数，如果达到最大值则等待
            while (( $(jobs -p | wc -l) >= max_processes )); do
                sleep 5
            done

            # 为每个实验创建独立的日志文件
            log_file="${Path_prefix}/${exp_name}/${env_name}/${log_dir}/experiment_s${seed}_n${noise}_sw${switch_step}.log"
            
            # 启动实验并将输出重定向到日志文件
            nohup python main.py \
                --seed=$seed \
                --noise_multiplier=$noise \
                --env_name=$env_name \
                --exp_name=$exp_name \
                --switch_steps=$switch_step > "$log_file" 2>&1 &
            
            echo "启动实验 $((++count)): seed=$seed, switch_steps=$switch_step"
            
            # 短暂延迟以避免同时启动
            sleep 2
        done
    done
done

# 等待所有实验完成
wait
echo "所有实验已完成"
