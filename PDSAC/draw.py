import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图像的目录
os.makedirs('./res', exist_ok=True)

# 数据
mass = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
# seeds=[42,123,0,2025,7,100,999,314,73,66]
seeds=[42,123,0,2025,7,100,999,73,66] #314 42
# noises=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
noises=[0.2]

env_name = 'Walker2d-v2'
hssac_exp_name='Dynamic_weights'
pdsac_exp_name='grad-norm-r5000'
sac_exp_name='Adam-optimizer'
scsac_exp_name='Adam-optimizer'
# seed = 42
# noise= 0.3

os.makedirs('./res/Friction', exist_ok=True)
os.makedirs('./res/Mass', exist_ok=True)

for seed in seeds:
    for noise in noises:
        # SAC 数据
        reward_sac = np.load(f'../SAC/test_result/{sac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]

        # PDSAC 数据
        reward_pdsac = np.load(f'../PDSAC/test_result/{pdsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')[0]
        # reward_pdsac = [216, 636, 1454, 2070, 2957, 3531, 4089, 4077, 4720, 4885, 3702, 948, 469, 370, 333, 307, 291, 301, 312, 310]

        # SAC-HS 数据
        reward_sac_hs = np.load(f'../HSSAC/test_result/{hssac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]

        # SCSAC 数据
        reward_scsac = np.load(f'../SCSAC/test_result/{scsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]


        # 创建图表
        plt.figure(figsize=(10, 6))  # 设置图表大小

        # 绘制 SAC 曲线
        plt.plot(mass, reward_sac, marker='o', linestyle='-', color='b', label='SAC')

        # 绘制 PDSAC 曲线
        plt.plot(mass, reward_pdsac, marker='s', linestyle='--', color='r', label='PDSAC')

        # 绘制 SAC-HS 曲线
        plt.plot(mass, reward_sac_hs, marker='^', linestyle='-.', color='g', label='HSSAC')

        # 绘制 SCSAC 曲线
        plt.plot(mass, reward_scsac, marker='*', linestyle=':', color='m', label='SCSAC')

        # 添加标题和标签
        plt.title('Reward vs Mass (Comparison of SAC, PDSAC, and SAC-HS)', fontsize=16)
        plt.xlabel('Mass', fontsize=14)
        plt.ylabel('Reward', fontsize=14)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图例
        plt.legend()

        # 保存图像到./res文件夹
        plt.savefig(f'./res/Mass/reward_vs_mass_comparison_seed{seed}_noise{noise}.png', dpi=300, bbox_inches='tight')
        print(f"图像已保存到 ./res/Mass/reward_vs_mass_comparison_seed{seed}_noise{noise}.png")
        # 关闭图形
        plt.close()


        # Friction
                # SAC 数据
        reward_sac = np.load(f'../SAC/test_result/{sac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]

        # PDSAC 数据
        reward_pdsac = np.load(f'../PDSAC/test_result/{pdsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')[1]
        # reward_pdsac = [216, 636, 1454, 2070, 2957, 3531, 4089, 4077, 4720, 4885, 3702, 948, 469, 370, 333, 307, 291, 301, 312, 310]

        # SAC-HS 数据
        reward_sac_hs = np.load(f'../HSSAC/test_result/{hssac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]

        # SCSAC 数据
        reward_scsac = np.load(f'../SCSAC/test_result/{scsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]


        # 创建图表
        plt.figure(figsize=(10, 6))  # 设置图表大小

        # 绘制 SAC 曲线
        plt.plot(mass, reward_sac, marker='o', linestyle='-', color='b', label='SAC')

        # 绘制 PDSAC 曲线
        plt.plot(mass, reward_pdsac, marker='s', linestyle='--', color='r', label='PDSAC')

        # 绘制 SAC-HS 曲线
        plt.plot(mass, reward_sac_hs, marker='^', linestyle='-.', color='g', label='HSSAC')

        # 绘制 SCSAC 曲线
        plt.plot(mass, reward_scsac, marker='*', linestyle=':', color='m', label='SCSAC')

        # 添加标题和标签
        plt.title('Reward vs Friction', fontsize=16)
        plt.xlabel('Friction', fontsize=14)
        plt.ylabel('Reward', fontsize=14)

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图例
        plt.legend()

        # 保存图像到./res文件夹
        plt.savefig(f'./res/Friction/reward_vs_Friction_comparison_seed{seed}_noise{noise}.png', dpi=300, bbox_inches='tight')
        print(f"图像已保存到 ./res/Friction/reward_vs_Friction_comparison_seed{seed}_noise{noise}.png")
        # 关闭图形
        plt.close()