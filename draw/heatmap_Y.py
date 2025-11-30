import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置参数
# seeds = [42,123,0,2025,7,100,999,73,66,314]
# seeds = [42,123,0]
# algorithm = "NISAC"
# noise = 1.0
# exp_name = 'Adam-optimizer_cross'
# exp_name = 'self_noise_polynomial_decay_1to0.1_cross'
# env_name = 'Walker2d-v2'

def get_reward_filename(algo_name, params, env_name, seed):
    """根据参数构建 reward 数据文件名"""
    base_path = f'../{algo_name}/test_result/{params["exp-name"]}/{env_name}/result_numpy/{env_name}_seed{seed}'
    # AIGC START
    # 如果是 RLE 算法，添加 sw 参数
    if params.get('sw') is not None:
        base_path += f'_sw{params["sw"]}'
    # AIGC END
    if params.get('noise') is not None:
        base_path += f'_noise{params["noise"]}'
    base_path += '.npy'
    return base_path

def draw_heatmap(env_name, algorithm_params, vmin=None, vmax=None):
    """
    为 algorithm_params 中的每个算法和其指定的每个种子绘制热力图。
    同时为每个算法绘制所有种子的平均值热力图。
    所有热力图使用统一的颜色标尺。

    参数:
    env_name: 环境名称
    algorithm_params: 包含所有算法参数的字典。
    vmin: 颜色范围的最小值（可选，如果为None则自动计算）
    vmax: 颜色范围的最大值（可选，如果为None则自动计算）
    """
    print("开始绘制热力图...")
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 16
    # AIGC START
    # 如果指定了固定的颜色范围，直接使用；否则收集数据计算
    if vmin is not None and vmax is not None:
        global_min = vmin
        global_max = vmax
        print(f"使用固定的颜色范围: vmin={vmin}, vmax={vmax}")
    else:
        # 首先收集所有数据以确定全局最大最小值
        all_data = []
        for algo_name, params in algorithm_params.items():
            algo_seeds = params.get('seed', [])
            if not algo_seeds:
                continue
                
            for seed in algo_seeds:
                try:
                    filename = get_reward_filename(algo_name, params, env_name, seed)
                    if not os.path.exists(filename):
                        continue
                        
                    reward_data = np.load(filename)
                    if reward_data.shape[0] >= 51 and reward_data.shape[1] >= 51:
                        indices = np.linspace(0, 50, 11, dtype=int)
                        reward_data_sampled = reward_data[indices,:][:,indices]
                        all_data.append(reward_data_sampled)
                except Exception as e:
                    print(f"      错误: 加载算法 {algo_name} 的种子 {seed} 数据时出错: {e}")
        
        if not all_data:
            print("警告: 没有找到任何有效数据，无法绘制热力图。")
            return
            
        # 计算全局最大最小值
        global_min = min(data.min() for data in all_data)
        global_max = max(data.max() for data in all_data)
        print(f"自动计算的颜色范围: vmin={global_min}, vmax={global_max}")
    # AIGC END
    
    # 为每个算法绘制热力图
    for algo_name, params in algorithm_params.items():
        print(f"  处理算法: {algo_name}")
        exp_name = params.get('exp-name', 'default_exp')
        algo_seeds = params.get('seed', [])
        noise = params.get('noise')

        if not algo_seeds:
            print(f"    警告: 算法 {algo_name} 没有指定种子，跳过热力图绘制。")
            continue

        # 创建保存路径
        file_path_base = f'./res/heatmap/{env_name}/{algo_name}'
        if noise is not None:
            file_path_base += f'_n{noise}'
        file_path = f'{file_path_base}/{exp_name}'
        os.makedirs(file_path, exist_ok=True)

        # 收集该算法的所有种子数据
        algo_data = []
        for seed in algo_seeds:
            print(f"    处理种子: {seed}")
            try:
                filename = get_reward_filename(algo_name, params, env_name, seed)
                if not os.path.exists(filename):
                    print(f"      警告: 无法找到文件 {filename}，跳过种子 {seed} 的热力图。")
                    continue
                
                reward_data = np.load(filename)
                
                if reward_data.shape[0] >= 51 and reward_data.shape[1] >= 51:
                    indices = np.linspace(0, 50, 11, dtype=int)
                    reward_data_sampled = reward_data[indices,:][:,indices]
                    algo_data.append(reward_data_sampled)
                    
                    # 绘制单个种子的热力图
                    plt.figure(figsize=(12, 10))
                    ax = sns.heatmap(reward_data_sampled,
                                    annot=False,
                                    fmt='.0f',
                                    cmap='YlOrRd',
                                    square=True,
                                    linewidths=.5,
                                    linecolor='lightgray',
                                    cbar_kws={"shrink": .7},
                                    vmin=global_min,
                                    vmax=global_max)

                    tick_labels = np.linspace(0.5, 1.5, num=11)
                    ax.set_xticks(np.arange(11) + 0.5)
                    ax.set_xticklabels([f'{x:.1f}' for x in tick_labels], rotation=0, ha="right")
                    ax.set_yticks(np.arange(11) + 0.5)
                    ax.set_yticklabels([f'{x:.1f}' for x in tick_labels], rotation=0)

                    plt.xlabel('Friction')
                    plt.ylabel('Mass')
                    title = f'{algo_name} - {env_name} - Seed={seed}'
                    if noise is not None:
                        title += f' Noise={noise}'
                    plt.title(title, fontweight='bold')

                    plt.tight_layout()
                    save_filename = f'{file_path}/MassAndFriction_S{seed}.pdf'
                    plt.savefig(save_filename, bbox_inches='tight')
                    plt.close()
                    print(f"      热力图已保存: {save_filename}")

            except Exception as e:
                print(f"      错误: 绘制算法 {algo_name} 的种子 {seed} 热力图时出错: {e}")
        
        # 绘制平均值热力图
        if algo_data:
            mean_data = np.mean(algo_data, axis=0)
            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(mean_data,
                            annot=False,
                            fmt='.0f',
                            cmap='YlOrRd',
                            square=True,
                            linewidths=.5,
                            linecolor='lightgray',
                            cbar_kws={"shrink": .7},
                            vmin=global_min,
                            vmax=global_max)

            tick_labels = np.linspace(0.5, 1.5, num=11)
            ax.set_xticks(np.arange(11) + 0.5)
            ax.set_xticklabels([f'{x:.1f}' for x in tick_labels], rotation=0, ha="right", fontsize=16)
            ax.set_yticks(np.arange(11) + 0.5)
            ax.set_yticklabels([f'{x:.1f}' for x in tick_labels], rotation=0, fontsize=16)

            plt.xlabel('Friction')
            plt.ylabel('Mass')
            title = f'{algo_name} - {env_name}'
            if algo_name == "NISAC":
                title = f'FLAGSAC - {env_name}'
            # if noise is not None:
            #     title += f' Noise={noise}'
            # plt.title(title, fontsize=16, fontweight='bold')

            plt.tight_layout()
            save_filename = f'{file_path}/{env_name.lower()}_heatmap_{algo_name.lower()}.pdf'
            plt.savefig(save_filename, bbox_inches='tight')
            plt.close()
            print(f"      平均值热力图已保存: {save_filename}")

    print("热力图绘制完成。")