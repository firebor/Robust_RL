import numpy as np
import plotly.graph_objects as go
import os

# 移除全局设置参数
# seeds=[42,123,0]
# algorithm= "NISAC"
# noise = 1.0
# exp_name = 'self_noise_polynomial_decay_1to0.1_cross'
# env_name = 'Walker2d-v2'

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
        return np.linspace(0.5, 1.5, 51)  # 注意这里用51个点
    elif env_name in ['HalfCheetah-v2', 'InvertedDoublePendulum-v2']:
        return np.linspace(0.1, 1.9, 51)  # 注意这里用51个点
    else:
        raise ValueError(f"未知的环境名称: {env_name}")

def draw_surface_html(env_name, algorithm_params):
    """
    为 algorithm_params 中的每个算法和其指定的每个种子绘制交互式3D曲面图。
    所有曲面图使用统一的颜色标尺。

    参数:
    env_name: 环境名称
    algorithm_params: 包含所有算法参数的字典。
    """
    print("开始绘制交互式3D曲面图...")
    
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
                    all_data.append(reward_data)
            except Exception as e:
                print(f"      错误: 加载算法 {algo_name} 的种子 {seed} 数据时出错: {e}")
    
    if not all_data:
        print("警告: 没有找到任何有效数据，无法绘制交互式3D曲面图。")
        return
        
    # 计算全局最大最小值
    global_min = min(data.min() for data in all_data)
    global_max = max(data.max() for data in all_data)
    
    for algo_name, params in algorithm_params.items():
        print(f"  处理算法: {algo_name}")
        exp_name = params.get('exp-name', 'default_exp')
        algo_seeds = params.get('seed', [])
        noise = params.get('noise')

        if not algo_seeds:
            print(f"    警告: 算法 {algo_name} 没有指定种子，跳过 HTML 曲面图绘制。")
            continue

        # 创建保存路径
        file_path_base = f'./res/surfacemap/{env_name}/{algo_name}'
        if noise is not None:
            file_path_base += f'_n{noise}'
        file_path = f'{file_path_base}/{exp_name}/html'
        os.makedirs(file_path, exist_ok=True)

        for seed in algo_seeds:
            print(f"    处理种子: {seed}")
            try:
                filename = get_reward_filename(algo_name, params, env_name, seed)
                if not os.path.exists(filename):
                    print(f"      警告: 无法找到文件 {filename}，跳过种子 {seed} 的 HTML 曲面图。")
                    continue

                reward_data = np.load(filename)

                # 检查数据维度是否为 51x51，plotly surface 需要匹配的 X, Y 维度
                if reward_data.shape != (51, 51):
                    print(f"      警告: 文件 {filename} 的形状为 {reward_data.shape}，期望 (51, 51)。跳过种子 {seed} 的 HTML 曲面图。")
                    continue

                # 生成网格坐标
                x = get_axis_range(env_name)
                y = get_axis_range(env_name)
                X, Y = np.meshgrid(x, y)

                # 颜色映射 (保持不变)
                coolwarm_plotly = [
                    [0.0, 'rgb(59, 76, 192)'],    # 冷色 (深蓝)
                    [0.5, 'rgb(221, 221, 221)'],  # 中性 (灰)
                    [1.0, 'rgb(180, 4, 38)']      # 暖色 (深红)
                ]

                # 创建交互式3D曲面图，使用全局最大最小值
                fig = go.Figure(data=[go.Surface(
                    x=X,
                    y=Y,
                    z=reward_data,
                    colorscale=coolwarm_plotly,
                    showscale=True,
                    colorbar=dict(title='Reward'),
                    cmin=global_min,  # 设置全局最小值
                    cmax=global_max   # 设置全局最大值
                )])

                # 更新布局
                title = f'{algo_name} - {env_name}'
                if noise is not None:
                    title += f' Noise={noise}'
                
                fig.update_layout(
                    title=dict(
                        text=title,
                        font=dict(size=20, weight='bold')
                    ),
                    scene=dict(
                        xaxis_title=dict(
                            text='Friction',
                            font=dict(size=14, weight='bold')
                        ),
                        yaxis_title=dict(
                            text='Mass',
                            font=dict(size=14, weight='bold')
                        ),
                        zaxis_title=dict(
                            text='Episode Reward',
                            font=dict(size=14, weight='bold')
                        ),
                        xaxis=dict(
                            ticktext=[f'{val:.1f}' for val in np.linspace(x[0], x[-1], 11)],
                            tickvals=np.linspace(x[0], x[-1], 11)
                        ),
                        yaxis=dict(
                            ticktext=[f'{val:.1f}' for val in np.linspace(y[0], y[-1], 11)],
                            tickvals=np.linspace(y[0], y[-1], 11)
                        )
                    ),
                    width=1000,
                    height=800,
                    margin=dict(l=65, r=50, b=65, t=90)
                )

                # 保存为交互式HTML文件
                save_filename = f'{file_path}/3D_Reward_Surface_S{seed}.html'
                fig.write_html(save_filename)
                print(f"      交互式 3D 曲面图已保存: {save_filename}")

            except Exception as e:
                print(f"      错误: 绘制算法 {algo_name} 的种子 {seed} HTML 曲面图时出错: {e}")

    print("3D Surface (HTML) 图绘制完成。")