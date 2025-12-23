import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def smooth_data(data, smoothing_factor=0.6): #默认0.6
    """
    使用TensorBoard底层算法对数据进行平滑处理
    
    参数:
    data: 要平滑的数据
    smoothing_factor: 平滑因子，范围0-0.999，值越大平滑效果越明显
    """
    x = data.copy()
    weight = smoothing_factor  # 权重 (动态规划)
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smoothing_factor  # `* smooth` 是为了让下一元素 权重为1
    return x

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
            pattern = f'seed{seed}'
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
        
        # 对数据进行平滑处理，使用TensorBoard底层算法
        smoothing_factor = 0.99
        smoothed_mean = smooth_data(mean_values, smoothing_factor=smoothing_factor)
        smoothed_std = smooth_data(std_values, smoothing_factor=smoothing_factor)
        
        # 绘制平滑后的平均值曲线和标准差带
        plt.plot(base_x_data, smoothed_mean, 
                 color=colors[algo_idx], 
                 linewidth=2,
                 label=algo)
        plt.fill_between(base_x_data, 
                         np.array(smoothed_mean) - np.array(smoothed_std), 
                         np.array(smoothed_mean) + np.array(smoothed_std), 
                         color=colors[algo_idx], 
                         alpha=0.1)
    
    # 设置图形属性
    # plt.title(f'{env} Training Curves (TensorBoard Smoothed Mean ± Std)', fontsize=16)
    plt.xlabel('Step', fontsize=18, fontweight='bold')
    plt.ylabel('Score', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=18)
    
    # 保存图形
    os.makedirs('./convergence_curves', exist_ok=True)
    plt.savefig(f'./convergence_curves/{env}_convergence_curves.pdf', dpi=300, bbox_inches='tight')
    plt.close()

# 使用示例
if __name__ == "__main__":
    # HalfCheetah-v2、Hopper-v2、Walker2d-v2、InvertedDoublePendulum-v2
    env = 'Walker2d-v2'
    # algorithms = ['SAC','SCSAC','NISAC']  # 算法列表
    algorithms = ['RLE']  # 算法列表
    # seeds = [[5571,6038]]
    seeds = [[42, 123, 0, 2025, 7, 100, 999, 314, 73, 66]]
    # [3407,1997,2023,3141,65537,1337,1009,2718,1001,233]
    # seeds = [[66, 123, 7],
    #          [66, 123, 7],
    #          [66, 123, 7]]
    
    draw_convergence_curves(env, algorithms, seeds)
