import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# seeds = [73]
# seeds=[42,123,0,2025,7,100,999,73,66,314]
seeds=[42,123,0]
algorithm= "NISAC"
noise = 1.0
exp_name = 'self_noise_polynomial_decay_1to0.1_cross'
env_name = 'Walker2d-v2'
file_path=''
if noise:
    file_path=f'./res/surfacemap/{algorithm}_n{noise}/{exp_name}/{env_name}/png'
else:
    file_path=f'./res/surfacemap/{algorithm}/{exp_name}/{env_name}/png'
os.makedirs(file_path, exist_ok=True)
for seed in seeds:
    if noise:
        reward_pdsac = np.load(f'../{algorithm}/test_result/{exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')
    else:
        reward_pdsac = np.load(f'../{algorithm}/test_result/{exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')
    # reward_pdsac = reward_pdsac[4:15, 4:15]  # 11x11 数据
    # 创建 3D 曲面图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 生成网格坐标（-0.5~0.5）
    x = np.linspace(0.5, 1.5, 51)  # 修改为 -0.5 到 0.5
    y = np.linspace(0.5, 1.5, 51)  # 修改为 -0.5 到 0.5
    X, Y = np.meshgrid(x, y)

    # 绘制 3D 曲面
    surf = ax.plot_surface(
        X, Y, reward_pdsac,
        cmap='coolwarm',       # 颜色映射
        edgecolor='none',     # 无边框
        alpha=0.8,           # 透明度
        rstride=1, cstride=1  # 曲面网格密度
    )

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=10, label='Reward')

    # 设置坐标轴标签
    ax.set_xlabel('Friction', fontsize=12)
    ax.set_ylabel('Mass', fontsize=12)
    ax.set_zlabel('Episode Reward', fontsize=12)
    ax.set_title(f'3D Reward Surface (Seed={seed})', fontsize=14)

    # 设置刻度
    ax.set_xticks(np.linspace(0.5, 1.5, 11))  # 11个刻度
    ax.set_yticks(np.linspace(0.5, 1.5, 11))  # 11个刻度
    ax.set_xticklabels([f'{x:.1f}' for x in np.linspace(0.5, 1.5, 11)], rotation=10)  # 旋转标签避免重叠
    ax.set_yticklabels([f'{y:.1f}' for y in np.linspace(0.5, 1.5, 11)], rotation=-10)

    # 调整视角（俯仰角30°，方位角120°）
    ax.view_init(elev=30, azim=120)

    # 保存图片
    plt.tight_layout()
    plt.savefig(f'{file_path}/3D_Reward_Surface_S{seed}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"3D曲面图已保存: 3D_Reward_Surface_S{seed}.png")