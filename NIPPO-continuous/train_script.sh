#!/bin/bash

# Set experiment parameters
seeds=(42 123 0 2025 7 100 999 314 73 66 61209 9834 77025 14563 30298 5571 84602 21937 69840 10526 43781 96215 2804 71359 52468 8901 63724 17890 94532 36107 5824 75619 20483 67390 1125 82976 40158 95723 26431 7806 59342 13874 86509 32167 4790 70258 25613 91470 6482 38795 53061 19247 85630 4108 76924 29571 6038 97145 33620 5489 81276 16039 72584 23910 68472 1075 49361 88027 35492 5718 79603 22845 65170 18396 90214 44753 3106 73829 27158 5640 83971 15284 62739 9406 48512 37690 5083 86147 24905 71268 13579 99999 42069 66666 10001 88888 55555 22222 77777 33333)
noises=(1.0)
max_processes=3
exp_name="default-exp"  # Experiment name
env_name="Hopper-v2"     # Environment name

Path_prefix="train_result"
log_dir="script_logs"
mkdir -p "${Path_prefix}/${exp_name}/${env_name}/${log_dir}"

# 计数器
count=0

for noise in "${noises[@]}"; do
    for seed in "${seeds[@]}"; do
        # 检查当前运行的进程数，如果达到最大值则等待
        while (( $(jobs -p | wc -l) >= max_processes )); do
            sleep 5
        done

        # 为每个实验创建独立的日志文件
        log_file="${Path_prefix}/${exp_name}/${env_name}/${log_dir}/experiment_s${seed}_n${noise}.log"
        
        # 启动实验并将输出重定向到日志文件
        nohup python main.py \
            --seed=$seed \
            --noise_multiplier=$noise \
            --env_name=$env_name \
            --exp_name=$exp_name  > "$log_file" 2>&1 &
        
        echo "启动实验 $((++count)): seed=$seed"
        
        # 短暂延迟以避免同时启动
        sleep 2
    done
done

# 等待所有实验完成
wait
echo "所有实验已完成"
