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
pdsac_exp_name='grad-norm'
sac_exp_name='Adam-optimizer'
scsac_exp_name='Adam-optimizer'
# seed = 42
# noise= 0.3

os.makedirs('./res/msr_norm/Friction', exist_ok=True)
os.makedirs('./res/msr_norm/Mass', exist_ok=True)

# 初始化存储所有种子数据的数组
all_reward_sac_mass = []
all_reward_pdsac_mass = []
all_reward_sac_hs_mass = []
all_reward_scsac_mass = []

all_reward_sac_friction = []
all_reward_pdsac_friction = []
all_reward_sac_hs_friction = []
all_reward_scsac_friction = []

for seed in seeds:
    for noise in noises:
        # Mass 数据收集
        reward_sac = np.load(f'../SAC/test_result/{sac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]
        reward_pdsac = np.load(f'../PDSAC/test_result/{pdsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')[0]
        reward_sac_hs = np.load(f'../HSSAC/test_result/{hssac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]
        reward_scsac = np.load(f'../SCSAC/test_result/{scsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[0]
        
        all_reward_sac_mass.append(reward_sac)
        all_reward_pdsac_mass.append(reward_pdsac)
        all_reward_sac_hs_mass.append(reward_sac_hs)
        all_reward_scsac_mass.append(reward_scsac)
        
        # Friction 数据收集
        reward_sac = np.load(f'../SAC/test_result/{sac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]
        reward_pdsac = np.load(f'../PDSAC/test_result/{pdsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise}.npy')[1]
        reward_sac_hs = np.load(f'../HSSAC/test_result/{hssac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]
        reward_scsac = np.load(f'../SCSAC/test_result/{scsac_exp_name}/{env_name}/result_numpy/{env_name}_seed{seed}.npy')[1]
        
        all_reward_sac_friction.append(reward_sac)
        all_reward_pdsac_friction.append(reward_pdsac)
        all_reward_sac_hs_friction.append(reward_sac_hs)
        all_reward_scsac_friction.append(reward_scsac)

# 转换为numpy数组并计算统计量
all_reward_sac_mass = np.array(all_reward_sac_mass)
all_reward_pdsac_mass = np.array(all_reward_pdsac_mass)
all_reward_sac_hs_mass = np.array(all_reward_sac_hs_mass)
all_reward_scsac_mass = np.array(all_reward_scsac_mass)

all_reward_sac_friction = np.array(all_reward_sac_friction)
all_reward_pdsac_friction = np.array(all_reward_pdsac_friction)
all_reward_sac_hs_friction = np.array(all_reward_sac_hs_friction)
all_reward_scsac_friction = np.array(all_reward_scsac_friction)

# 计算归一化基准值
baseline_sac_mass = np.mean(all_reward_sac_mass[:, 9])
baseline_pdsac_mass = np.mean(all_reward_pdsac_mass[:, 9])
baseline_sac_hs_mass = np.mean(all_reward_sac_hs_mass[:, 9])
baseline_scsac_mass = np.mean(all_reward_scsac_mass[:, 9])

baseline_sac_friction = np.mean(all_reward_sac_friction[:, 9])
baseline_pdsac_friction = np.mean(all_reward_pdsac_friction[:, 9])
baseline_sac_hs_friction = np.mean(all_reward_sac_hs_friction[:, 9])
baseline_scsac_friction = np.mean(all_reward_scsac_friction[:, 9])

# 计算均值和标准差
mean_sac_mass = np.mean(all_reward_sac_mass, axis=0)
std_sac_mass = np.std(all_reward_sac_mass, axis=0)
mean_pdsac_mass = np.mean(all_reward_pdsac_mass, axis=0)
std_pdsac_mass = np.std(all_reward_pdsac_mass, axis=0)
mean_sac_hs_mass = np.mean(all_reward_sac_hs_mass, axis=0)
std_sac_hs_mass = np.std(all_reward_sac_hs_mass, axis=0)
mean_scsac_mass = np.mean(all_reward_scsac_mass, axis=0)
std_scsac_mass = np.std(all_reward_scsac_mass, axis=0)

mean_sac_friction = np.mean(all_reward_sac_friction, axis=0)
std_sac_friction = np.std(all_reward_sac_friction, axis=0)
mean_pdsac_friction = np.mean(all_reward_pdsac_friction, axis=0)
std_pdsac_friction = np.std(all_reward_pdsac_friction, axis=0)
mean_sac_hs_friction = np.mean(all_reward_sac_hs_friction, axis=0)
std_sac_hs_friction = np.std(all_reward_sac_hs_friction, axis=0)
mean_scsac_friction = np.mean(all_reward_scsac_friction, axis=0)
std_scsac_friction = np.std(all_reward_scsac_friction, axis=0)

# 绘制 Mass 归一化图表
plt.figure(figsize=(10, 6))

# 计算归一化后的均值和标准差
norm_mean_sac_mass = mean_sac_mass / baseline_sac_mass
norm_std_sac_mass = std_sac_mass / baseline_sac_mass
norm_mean_pdsac_mass = mean_pdsac_mass / baseline_pdsac_mass
norm_std_pdsac_mass = std_pdsac_mass / baseline_pdsac_mass
norm_mean_sac_hs_mass = mean_sac_hs_mass / baseline_sac_hs_mass
norm_std_sac_hs_mass = std_sac_hs_mass / baseline_sac_hs_mass
norm_mean_scsac_mass = mean_scsac_mass / baseline_scsac_mass
norm_std_scsac_mass = std_scsac_mass / baseline_scsac_mass

# 绘制归一化曲线和阴影
plt.plot(mass, norm_mean_sac_mass, color='b', label='SAC')
plt.fill_between(mass, norm_mean_sac_mass - norm_std_sac_mass, norm_mean_sac_mass + norm_std_sac_mass, color='b', alpha=0.2)

plt.plot(mass, norm_mean_pdsac_mass, color='r', label='PDSAC')
plt.fill_between(mass, norm_mean_pdsac_mass - norm_std_pdsac_mass, norm_mean_pdsac_mass + norm_std_pdsac_mass, color='r', alpha=0.2)

plt.plot(mass, norm_mean_sac_hs_mass, color='g', label='HSSAC')
plt.fill_between(mass, norm_mean_sac_hs_mass - norm_std_sac_hs_mass, norm_mean_sac_hs_mass + norm_std_sac_hs_mass, color='g', alpha=0.2)

plt.plot(mass, norm_mean_scsac_mass, color='m', label='SCSAC')
plt.fill_between(mass, norm_mean_scsac_mass - norm_std_scsac_mass, norm_mean_scsac_mass + norm_std_scsac_mass, color='m', alpha=0.2)

plt.title('Normalized Reward vs Mass (Mean ± Std)', fontsize=16)
plt.xlabel('Mass', fontsize=14)
plt.ylabel('Normalized Reward', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('./res/msr_norm/Mass/normalized_reward_vs_mass_mean_std.png', dpi=300, bbox_inches='tight')
plt.close()

# 绘制 Friction 归一化图表
plt.figure(figsize=(10, 6))

# 计算归一化后的均值和标准差
norm_mean_sac_friction = mean_sac_friction / baseline_sac_friction
norm_std_sac_friction = std_sac_friction / baseline_sac_friction
norm_mean_pdsac_friction = mean_pdsac_friction / baseline_pdsac_friction
norm_std_pdsac_friction = std_pdsac_friction / baseline_pdsac_friction
norm_mean_sac_hs_friction = mean_sac_hs_friction / baseline_sac_hs_friction
norm_std_sac_hs_friction = std_sac_hs_friction / baseline_sac_hs_friction
norm_mean_scsac_friction = mean_scsac_friction / baseline_scsac_friction
norm_std_scsac_friction = std_scsac_friction / baseline_scsac_friction

# 绘制归一化曲线和阴影
plt.plot(mass, norm_mean_sac_friction, color='b', label='SAC')
plt.fill_between(mass, norm_mean_sac_friction - norm_std_sac_friction, norm_mean_sac_friction + norm_std_sac_friction, color='b', alpha=0.2)

plt.plot(mass, norm_mean_pdsac_friction, color='r', label='PDSAC')
plt.fill_between(mass, norm_mean_pdsac_friction - norm_std_pdsac_friction, norm_mean_pdsac_friction + norm_std_pdsac_friction, color='r', alpha=0.2)

plt.plot(mass, norm_mean_sac_hs_friction, color='g', label='HSSAC')
plt.fill_between(mass, norm_mean_sac_hs_friction - norm_std_sac_hs_friction, norm_mean_sac_hs_friction + norm_std_sac_hs_friction, color='g', alpha=0.2)

plt.plot(mass, norm_mean_scsac_friction, color='m', label='SCSAC')
plt.fill_between(mass, norm_mean_scsac_friction - norm_std_scsac_friction, norm_mean_scsac_friction + norm_std_scsac_friction, color='m', alpha=0.2)

plt.title('Normalized Reward vs Friction (Mean ± Std)', fontsize=16)
plt.xlabel('Friction', fontsize=14)
plt.ylabel('Normalized Reward', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('./res/msr_norm/Friction/normalized_reward_vs_friction_mean_std.png', dpi=300, bbox_inches='tight')
plt.close()