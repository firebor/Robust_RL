import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图像的目录
os.makedirs('./res', exist_ok=True)

# 数据
mass = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]

seeds=[42,123,0,2025,7,100,999,73,66,314]
# seeds=[7,42,66,73,2025]
# noises=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
noise=0.5

env_name = 'Walker2d-v2'
hssac_exp_name='Dynamic_weights'
pdsac_exp_name='grad-norm-r5000-n0.5_cross'
sac_exp_name='Adam-optimizer'
scsac_exp_name='Adam-optimizer'
# seed = 42
# noise= 0.3

def get_reward_filename(algo_name, params, env_name, seed):
    """根据参数构建 reward 数据文件名"""
    base_path = f'../{algo_name}/test_result/{params["exp-name"]}/{env_name}/result_numpy/{env_name}_seed{seed}'
    if params.get('noise') is not None:
        base_path += f'_noise{params["noise"]}'
    base_path += '.npy'
    return base_path

def get_axis_range(env_name):
    """根据环境名称返回对应的坐标轴范围"""
    if env_name in ['Hopper-v2', 'Walker2d-v2']:
        return np.linspace(0.5, 1.5, 11)
    elif env_name in ['HalfCheetah-v2', 'InvertedDoublePendulum-v2']:
        return np.linspace(0.1, 1.9, 11)
    else:
        raise ValueError(f"未知的环境名称: {env_name}")

def draw_msr(env_name, algorithm_params):
    """
    绘制均值和标准差折线图（残影图），对比不同算法。
    每个算法的结果是其在指定种子上的聚合。

    参数:
    env_name: 环境名称
    algorithm_params: 包含所有算法参数的字典。
    """
    print("开始绘制 MSR (Mean ± Std) 图...")
    # 确定保存路径 (使用 env_name 区分)
    prefix_path = f'./res/msr/{env_name}'
    os.makedirs(f'{prefix_path}/Mass', exist_ok=True)
    os.makedirs(f'{prefix_path}/Friction', exist_ok=True)

    # 根据环境获取坐标轴范围
    mass_friction_values = get_axis_range(env_name)

    # 定义线条样式列表
    markers = ['o', 's', '^', '*', 'D', 'p', 'h', 'v', '<', '>', '8', 'd']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.cm.tab10.colors

    # 初始化存储所有种子数据的字典
    all_rewards = {algo_name: {'mass': [], 'friction': []} for algo_name in algorithm_params.keys()}

    # 加载每个算法的数据
    for algo_name, params in algorithm_params.items():
        print(f"  处理算法: {algo_name}")
        algo_seeds = params.get('seed', [])
        if not algo_seeds:
            print(f"    警告: 算法 {algo_name} 没有指定种子，跳过。")
            continue

        for seed in algo_seeds:
            try:
                filename = get_reward_filename(algo_name, params, env_name, seed)
                if not os.path.exists(filename):
                    print(f"    警告: 无法找到文件 {filename}，跳过算法 {algo_name} 的种子 {seed}")
                    continue

                reward = np.load(filename)

                # 提取 Mass 和 Friction 数据
                if reward.shape[0] > 25 and reward.shape[1] > 25:
                    # Mass 数据
                    flattened_mass = reward[:, 25].flatten()
                    n_mass = len(flattened_mass)
                    indices_mass = np.linspace(0, n_mass - 1, 11, dtype=int)
                    reward_mass = flattened_mass[indices_mass]
                    all_rewards[algo_name]['mass'].append(reward_mass)

                    # Friction 数据
                    flattened_friction = reward[25, :].flatten()
                    n_friction = len(flattened_friction)
                    indices_friction = np.linspace(0, n_friction - 1, 11, dtype=int)
                    reward_friction = flattened_friction[indices_friction]
                    all_rewards[algo_name]['friction'].append(reward_friction)
                else:
                    print(f"    警告: 文件 {filename} 的形状 {reward.shape} 不符合预期，无法提取数据。")

            except Exception as e:
                print(f"    错误: 加载或处理算法 {algo_name} 的种子 {seed} 时出错: {e}")

    # 转换为numpy数组
    valid_algorithms = []
    for algo_name in algorithm_params.keys():
        if all_rewards[algo_name]['mass']: # 检查是否有数据
            all_rewards[algo_name]['mass'] = np.array(all_rewards[algo_name]['mass'])
            all_rewards[algo_name]['friction'] = np.array(all_rewards[algo_name]['friction'])
            valid_algorithms.append(algo_name)
        else:
             # 如果某个算法没有加载到任何数据，从字典中移除，避免绘图时出错
            del all_rewards[algo_name]
            print(f"  算法 {algo_name} 没有加载到有效数据，将不参与绘图。")

    if not valid_algorithms:
        print("错误: 没有为任何算法加载到有效数据，无法绘制 MSR 图。")
        return

    # --- 绘制 Mass 图表 --- 
    plt.figure(figsize=(12, 7))
    plot_count = 0
    for algo_name in valid_algorithms:
        rewards = all_rewards[algo_name]
        if rewards['mass'].size > 0: # 再次确认有数据
            # 计算均值和标准差
            mean = np.mean(rewards['mass'], axis=0)
            std = np.std(rewards['mass'], axis=0)

            params = algorithm_params[algo_name]
            label = f"{algo_name}"
            if algo_name == "NISAC":
                label = "FLAGSAC"
            # if params.get('noise') is not None:
            #     label += f" (n={params['noise']})"
            # label += f" ({params['exp-name']})" # MSR图例通常只标算法名

            # 绘制曲线和阴影
            plt.plot(mass_friction_values, mean,
                    marker=markers[plot_count % len(markers)],
                    linestyle=linestyles[plot_count % len(linestyles)],
                    color=colors[plot_count % len(colors)],
                    label=label)
            plt.fill_between(mass_friction_values, mean - std, mean + std,
                            color=colors[plot_count % len(colors)],
                            alpha=0.1) # 降低透明度
            plot_count += 1

    if plot_count > 0:
        # plt.title(f'{env_name} - Reward vs Mass (Mean ± Std across seeds)', fontsize=18, fontweight='bold')
        plt.xlabel('Mass', fontsize=18, fontweight='bold')
        plt.ylabel('Reward', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        save_path_mass = f'{prefix_path}/Mass/reward_vs_mass_mean_std.pdf'
        plt.savefig(save_path_mass, bbox_inches='tight')
        print(f"  图像已保存到 {save_path_mass}")
    plt.close()

    # --- 绘制 Friction 图表 --- 
    plt.figure(figsize=(12, 7))
    plot_count = 0
    for algo_name in valid_algorithms:
        rewards = all_rewards[algo_name]
        if rewards['friction'].size > 0:
            # 计算均值和标准差
            mean = np.mean(rewards['friction'], axis=0)
            std = np.std(rewards['friction'], axis=0)

            params = algorithm_params[algo_name]
            label = f"{algo_name}"
            if algo_name == "NISAC":
                label = "FLAGSAC"
            # if params.get('noise') is not None:
                # label += f" (n={params['noise']})"
            # label += f" ({params['exp-name']})"

            # 绘制曲线和阴影
            plt.plot(mass_friction_values, mean,
                    marker=markers[plot_count % len(markers)],
                    linestyle=linestyles[plot_count % len(linestyles)],
                    color=colors[plot_count % len(colors)],
                    label=label)
            plt.fill_between(mass_friction_values, mean - std, mean + std,
                            color=colors[plot_count % len(colors)],
                            alpha=0.1)
            plot_count += 1

    if plot_count > 0:
        # plt.title(f'{env_name} - Reward vs Friction (Mean ± Std across seeds)', fontsize=18, fontweight='bold')
        plt.xlabel('Friction', fontsize=18, fontweight='bold')
        plt.ylabel('Reward', fontsize=18, fontweight='bold')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=18)
        save_path_friction = f'{prefix_path}/Friction/reward_vs_friction_mean_std.pdf'
        plt.savefig(save_path_friction, bbox_inches='tight')
        print(f"  图像已保存到 {save_path_friction}")
    plt.close()
    print("MSR 图绘制完成。")