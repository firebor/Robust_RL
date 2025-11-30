import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def draw_convergence_curves(env, algorithms, seeds):
    """
    绘制多个算法的收敛曲线对比图
    
    参数:
    env: 环境名称
    algorithms: 算法名称列表
    seeds: 种子列表
    """
    colors = plt.cm.tab10.colors
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    for algo_idx, algo in enumerate(algorithms):
        file_path = f'./json_file/{algo}/{env}'
        
        # 存储所有种子的数据
        all_data = []
        
        # 收集所有种子的数据
        for seed in seeds[algo_idx]:
            # 查找包含特定seed的文件
            pattern = f'seed{seed}-'
            matching_files = [f for f in os.listdir(file_path) if pattern in f]
            
            if not matching_files:
                print(f"Warning: No file found for {algo} seed {seed}")
                continue
                
            json_file = matching_files[0]  # 取第一个匹配的文件
            with open(f'{file_path}/{json_file}', 'r') as f:
                data = [json.loads(line) for line in f]
                data = data[0]
            
            # 提取数据
            x_data = [item[1] for item in data]  # 第二列作为横坐标
            y_data = [item[2] for item in data]  # 第三列作为纵坐标
            all_data.append((x_data, y_data))
        
        # 找到最短的x_data范围作为基准
        min_length = min(len(x) for x, _ in all_data)
        base_x_data = None
        
        # 找到最短的x_data
        for x_data, _ in all_data:
            if len(x_data) == min_length:
                base_x_data = x_data
                break
        
        # 计算平均值和标准差
        mean_values = []
        std_values = []
        
        for x in base_x_data:
            # 收集所有种子在该x点附近的y值
            y_values = []
            for x_data, y_data in all_data:
                # 找到最接近的x值
                idx = np.argmin(np.abs(np.array(x_data) - x))
                y_values.append(y_data[idx])
            
            mean_values.append(np.mean(y_values))
            std_values.append(np.std(y_values))
        
        # 绘制平均值曲线和标准差带
        plt.plot(base_x_data, mean_values, 
                 color=colors[algo_idx], 
                 linewidth=2,
                 label=algo)
        plt.fill_between(base_x_data, 
                         np.array(mean_values) - np.array(std_values), 
                         np.array(mean_values) + np.array(std_values), 
                         color=colors[algo_idx], 
                         alpha=0.1)
    
    # 设置图形属性
    plt.title(f'{env} Training Curves (Mean ± Std)', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # 保存图形
    os.makedirs('./bujiaSmooth_convergence_curves', exist_ok=True)
    plt.savefig(f'./bujiaSmooth_convergence_curves/{env}_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
if __name__ == "__main__":
    # HalfCheetah-v2、Hopper-v2、Walker2d-v2、InvertedDoublePendulum-v2
    env = 'InvertedDoublePendulum-v2'
    algorithms = ['SCSAC','SAC']  # 算法列表
    # [3407,1997,2023,3141,65537,1337,1009,2718,1001,233]
    seeds = [[0,2025,7,100,999,314,73,66],
             [0,100,999,314,73]]
    
    draw_convergence_curves(env, algorithms, seeds)
