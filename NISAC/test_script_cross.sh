#!/bin/bash


# 设置实验参数
# seeds=(123 0 2025 7 100 999 314 73 66)
seeds=(42 123 0 2025 7 100 999 314 73 66)
# seeds=(7 73 314 999)
# seeds=(0 7 42 100 123 999 2025)
# noises=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
noises=(1.0)
max_processes=1

# 创建日志目录
Path_prefix="test_result"
env_name="HalfCheetah-v2"
exp_name="polynomial_grad_norm_scaled_decay"
log_dir="script_logs"
mkdir -p "${Path_prefix}/${exp_name}_cross/${env_name}/${log_dir}"

# 计数器
count=0

# 创建一个数组来存储所有后台进程的PID
declare -a pids

for noise in "${noises[@]}"; do
    for seed in "${seeds[@]}"; do
        # 检查当前运行的进程数，如果达到最大值则等待
        while (( ${#pids[@]} >= max_processes )); do
            # echo "当前运行进程数: ${#pids[@]}, 等待进程完成..."
            # 检查是否有进程已经完成
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    echo "进程 ${pids[$i]} 已完成"
                    unset 'pids[$i]'
                    pids=("${pids[@]}")  # 重新索引数组
                fi
            done
            sleep 5
        done

        # 为每个实验创建独立的日志文件
        log_file="${Path_prefix}/${exp_name}_cross/${env_name}/${log_dir}/experiment_s${seed}_n${noise}.log"
        
        # 启动实验并将输出重定向到日志文件
        nohup python main_test_multi_cross.py \
            --exp_name=$exp_name \
            --seed=$seed \
            --noise_multiplier=$noise \
            --env_name=$env_name > "$log_file" 2>&1 &
        
        # 保存进程PID
        pids+=($!)
        
        echo "启动实验 $((++count)): seed=$seed, noise=$noise, PID=$!"
        
        # 短暂延迟以避免同时启动
        sleep 2
    done
done

# 等待所有进程完成
echo "等待所有进程完成..."
for pid in "${pids[@]}"; do
    wait $pid
    echo "进程 $pid 已完成"
done

echo "所有实验已完成"