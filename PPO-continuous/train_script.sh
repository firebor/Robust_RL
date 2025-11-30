#!/bin/bash

# Set experiment parameters
seeds=(42 123 0 2025 7 100 999 314 73 66)
max_processes=10
exp_name="PPO-continuous"  # Experiment name
env_name="Walker2d-v2"     # Environment name

# Create log directory
log_dir="experiment_logs"
mkdir -p $log_dir

# Counter for tracking experiments
count=0

for seed in "${seeds[@]}"; do
    # Check number of running processes, wait if maximum reached
    while (( $(jobs -p | wc -l) >= max_processes )); do
        sleep 5
    done
    
    # Create individual log file for each experiment
    log_file="${log_dir}/experiment_s${seed}.log"
    
    # Start experiment and redirect output to log file
    nohup python main.py \
        --seed=$seed \
        --exp_name=$exp_name \
        --env_name=$env_name \
        --use_cuda=True \
        > "$log_file" 2>&1 &
    
    echo "Started experiment $((++count)): seed=$seed, exp_name=$exp_name"
    
    # Short delay to avoid simultaneous starts
    sleep 2
done

# Wait for all experiments to complete
wait
echo "All experiments completed" 