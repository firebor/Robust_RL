import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图像的目录
# os.makedirs('./res', exist_ok=True) # 由 draw_all.py 创建或在函数内部创建

# 数据 - 这些全局变量不再需要，由调用者传入或在函数内定义
# mass = [0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
# seeds=[42,123,0,2025,7,100,999,73,66,314]
# noise=0.5
# env_name = 'Walker2d-v2'
# hssac_exp_name='Dynamic_weights'
# pdsac_exp_name='grad-norm-r5000-n0.5_cross'
# sac_exp_name='Adam-optimizer'
# scsac_exp_name='Adam-optimizer'


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

def draw_per_seed_comparison_lines(env_name, algorithm_params):
    """
    绘制折线图比较不同算法的性能，为每个种子生成一张对比图。

    参数:
    env_name: 环境名称
    algorithm_params: 包含所有算法参数的字典，格式如下:
        {
            'Algo1': {'exp-name': 'exp1', 'seed': [s1, s2], 'noise': n1},
            'Algo2': {'exp-name': 'exp2', 'seed': [s2, s3], 'noise': None},
            ...
        }
    """
    print("开始绘制按种子对比的折线图 (Per-seed Comparison Lines)...")
    # 收集所有算法的所有种子
    all_seeds = set()
    for params in algorithm_params.values():
        all_seeds.update(params.get('seed', []))

    if not all_seeds:
        print("警告: 在 algorithm_params 中没有找到任何种子。")
        return

    # 根据环境获取坐标轴范围
    mass_friction_values = get_axis_range(env_name)

    # 定义线条样式列表
    markers = ['o', 's', '^', '*', 'D', 'p', 'h', 'v', '<', '>', '8', 'd']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']
    colors = plt.cm.tab10.colors # 使用tab10颜色映射获取区分度更高的颜色

    # 确定根保存路径 (基于第一个算法的noise或默认值)
    # 注意：如果不同算法有不同的noise，图例需要清晰标示
    # 为了简化，我们创建一个顶层目录，然后按种子分子目录
    base_save_path = f'./res/zhexian/{env_name}'
    os.makedirs(f'{base_save_path}/Mass', exist_ok=True)
    os.makedirs(f'{base_save_path}/Friction', exist_ok=True)

    for seed in sorted(list(all_seeds)):
        print(f"  处理种子: {seed}")
        rewards_mass_for_seed = {}
        rewards_friction_for_seed = {}
        algorithms_in_plot = []

        # 加载包含当前种子的算法数据
        for algo_name, params in algorithm_params.items():
            if seed in params.get('seed', []):
                try:
                    filename = get_reward_filename(algo_name, params, env_name, seed)
                    if not os.path.exists(filename):
                        print(f"    警告: 无法找到文件 {filename}，跳过算法 {algo_name} 的种子 {seed}")
                        continue

                    reward = np.load(filename)
                    algorithms_in_plot.append(algo_name) # 记录成功加载数据的算法

                    # 提取 Mass 和 Friction 数据 (假设reward形状是固定的)
                    # 如果 reward 形状可变，需要更健壮的提取方式
                    if reward.shape[0] > 25 and reward.shape[1] > 25: # 基本检查
                        # Mass: 第26行(:)或第26列(25) - 根据实际数据结构调整
                        # 假设是沿列变化取第26列 (index 25)
                        flattened_mass = reward[:, 25].flatten()
                        n_mass = len(flattened_mass)
                        indices_mass = np.linspace(0, n_mass - 1, 11, dtype=int)
                        rewards_mass_for_seed[algo_name] = flattened_mass[indices_mass]

                        # Friction: 第26行(25)或第26列(:) - 根据实际数据结构调整
                        # 假设是沿行变化取第26行 (index 25)
                        flattened_friction = reward[25, :].flatten()
                        n_friction = len(flattened_friction)
                        indices_friction = np.linspace(0, n_friction - 1, 11, dtype=int)
                        rewards_friction_for_seed[algo_name] = flattened_friction[indices_friction]
                    else:
                         print(f"    警告: 文件 {filename} 的形状 {reward.shape} 不符合预期，无法提取数据。")


                except Exception as e:
                    print(f"    错误: 加载或处理算法 {algo_name} 的种子 {seed} 时出错: {e}")

        # 如果该种子有至少一个算法的数据，则绘图
        if algorithms_in_plot:
            # 绘制质量比较图
            plt.figure(figsize=(12, 7))
            plot_count = 0
            for i, algo_name in enumerate(algorithms_in_plot):
                 if algo_name in rewards_mass_for_seed:
                    params = algorithm_params[algo_name]
                    label = f"{algo_name}"
                    if algo_name == "NISAC":
                        label = "FLAGSAC"
                    # if params.get('noise') is not None:
                    #     label += f" (n={params['noise']})"
                    # label += f" ({params['exp-name']})" # 添加 exp-name 到图例

                    plt.plot(mass_friction_values, rewards_mass_for_seed[algo_name],
                             marker=markers[plot_count % len(markers)],
                             linestyle=linestyles[plot_count % len(linestyles)],
                             color=colors[plot_count % len(colors)],
                             label=label)
                    plot_count += 1

            if plot_count > 0: # 确保有数据绘制
                plt.title(f'{env_name} - Reward vs Mass (Seed={seed})', fontsize=30, fontweight='bold')
                plt.xlabel('Mass', fontsize=30, fontweight='bold')
                plt.ylabel('Reward', fontsize=30, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=30) # 调整图例字体大小
                save_path_mass = f'{base_save_path}/Mass/reward_vs_mass_seed{seed}.pdf'
                plt.savefig(save_path_mass, bbox_inches='tight')
                print(f"    图像已保存到 {save_path_mass}")
            plt.close()

            # 绘制摩擦比较图
            plt.figure(figsize=(12, 7))
            plot_count = 0
            for i, algo_name in enumerate(algorithms_in_plot):
                 if algo_name in rewards_friction_for_seed:
                    params = algorithm_params[algo_name]
                    label = f"{algo_name}"
                    if algo_name == "NISAC":
                        label = "FLAGSAC"
                    # if params.get('noise') is not None:
                    #     label += f" (n={params['noise']})"
                    # label += f" ({params['exp-name']})" # 添加 exp-name 到图例

                    plt.plot(mass_friction_values, rewards_friction_for_seed[algo_name],
                             marker=markers[plot_count % len(markers)],
                             linestyle=linestyles[plot_count % len(linestyles)],
                             color=colors[plot_count % len(colors)],
                             label=label)
                    plot_count += 1

            if plot_count > 0: # 确保有数据绘制
                plt.title(f'{env_name} - Reward vs Friction (Seed={seed})', fontsize=16, fontweight='bold')
                plt.xlabel('Friction', fontsize=14, fontweight='bold')
                plt.ylabel('Reward', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=10) # 调整图例字体大小
                save_path_friction = f'{base_save_path}/Friction/reward_vs_friction_seed{seed}.pdf'
                plt.savefig(save_path_friction, bbox_inches='tight')
                print(f"    图像已保存到 {save_path_friction}")
            plt.close()
        else:
            print(f"  种子 {seed} 没有成功加载任何算法的数据，跳过绘图。")

    print("按种子对比的折线图绘制完成。")


# 移除 draw_zhexian_with_seed 函数，因为现在 draw_zhexian 处理所有情况
# def draw_zhexian_with_seed(...):
#     pass

# 移除旧的 draw_zhexian 函数定义（如果存在两个）
