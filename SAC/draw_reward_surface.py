import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import copy
import os
import torch.nn as nn
from configparser import ConfigParser
from utils.utils import Dict
from sac import SAC
from typing import Dict as TypeDict, OrderedDict

# --- 配置参数 ---
ENV_ID = "Walker2d-v2"  # 环境ID
MODEL_PATH = "/home/admin1/code/Robust-RL/SAC/best_model/Adam-optimizer/Walker2d-v2/Walker2d-v2_sac_seed7/agent_1500000_5453"  # <<<=== 在此加载预训练的SAC模型
N_GRID_POINTS = 21  # 网格分辨率（例如：论文中某些环境使用31x31 [引用: 256]）
RANGE = 1.0         # alpha和beta的范围（例如：Walker2d在论文中使用[-1, 1] [引用: 256]）
N_EVAL_EPISODES = 10 # 每个点评估的回合数（增加以提高精度 [引用: 256]）
SEED = 42           # 用于复现随机方向的种子

# --- 辅助函数 ---

def get_params_dict(model: SAC) -> OrderedDict:
    """提取模型参数到有序字典中。"""
    params = collections.OrderedDict()
    # 使用深拷贝创建参数的独立副本
    for name, param in model.actor.named_parameters():
        params[name] = copy.deepcopy(param)
    # 包含critic参数
    # for name, param in model.q_1.named_parameters():
    #     params[name] = copy.deepcopy(param)
    # for name, param in model.q_2.named_parameters():
    #     params[name] = copy.deepcopy(param)
    # 包含目标critic参数（如果需要）
    # for name, param in model.target_q_1.named_parameters():
    #     params[name] = copy.deepcopy(param)
    # for name, param in model.target_q_2.named_parameters():
    #     params[name] = copy.deepcopy(param)
    return params

def set_params_dict(model: SAC, params_dict: OrderedDict):
    """从有序字典中设置模型参数。"""
    # 更新actor参数
    actor_params = {k: v for k, v in params_dict.items() if k in dict(model.actor.named_parameters())}
    model.actor.load_state_dict(actor_params, strict=False)
    
    # 更新critic参数
    # q1_params = {k: v for k, v in params_dict.items() if k in dict(model.q_1.named_parameters())}
    # model.q_1.load_state_dict(q1_params, strict=False)
    
    # q2_params = {k: v for k, v in params_dict.items() if k in dict(model.q_2.named_parameters())}
    # model.q_2.load_state_dict(q2_params, strict=False)
    
    # 更新目标critic参数
    # target_q1_params = {k: v for k, v in params_dict.items() if k in dict(model.target_q_1.named_parameters())}
    # model.target_q_1.load_state_dict(target_q1_params, strict=False)
    
    # target_q2_params = {k: v for k, v in params_dict.items() if k in dict(model.target_q_2.named_parameters())}
    # model.target_q_2.load_state_dict(target_q2_params, strict=False)
    
    # 确保目标网络正确同步
    # model.soft_update(model.q_1, model.target_q_1, 1.0)
    # model.soft_update(model.q_2, model.target_q_2, 1.0)

def get_filter_normalized_direction(model: SAC, params_dict: OrderedDict) -> OrderedDict:
    """生成滤波归一化的随机方向[引用: 30, 31]。"""
    direction = collections.OrderedDict()
    # 使用与模型参数相同的设备
    device = next(model.actor.parameters()).device

    for name, param in params_dict.items():
        # 生成具有相同形状的高斯噪声
        noise = torch.randn(param.size(), device=device)

        # 计算噪声和参数张量的范数
        # 对矩阵/张量使用Frobenius范数
        noise_norm = torch.linalg.norm(noise)
        param_norm = torch.linalg.norm(param)

        # 避免当noise_norm为零时的除零错误
        if noise_norm > 1e-10:
            # 应用滤波归一化：缩放噪声以匹配参数范数 [引用: 31]
            normalized_noise = noise * (param_norm / noise_norm)
        else:
            normalized_noise = torch.zeros_like(noise) # 或者适当处理

        direction[name] = normalized_noise

    return direction

def evaluate_policy_reward(model: SAC, env: gym.Env, n_eval_episodes: int) -> float:
    """评估策略并返回平均奖励。"""
    # 重要：根据论文使用非折扣奖励 [引用: 36]
    
    all_episode_rewards = []
    for _ in range(n_eval_episodes):
        episode_rewards = []
        done = False
        truncated = False
        obs, _ = env.reset()
        
        try:
            while not done and not truncated:
                # 使用SAC的get_action方法
                state_tensor = torch.tensor(obs, dtype=torch.float, device=model.device).unsqueeze(0)
                action, _ = model.get_action(state_tensor)
                action = action.cpu().detach().numpy().reshape(-1)
                
                # 确保动作在有效范围内
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, done, truncated, info = env.step(action)
                # 累积回合的非折扣奖励
                episode_rewards.append(reward)
        except Exception as e:
            print(f"评估过程中出现错误: {e}")
            print(f"当前状态: {obs}")
            print(f"当前动作: {action}")
            return 0.0
            
        all_episode_rewards.append(sum(episode_rewards))

    mean_reward = np.mean(all_episode_rewards)
    return mean_reward

# --- 加载模型的辅助函数 ---
def load_sac_model(model_path, env, device='cuda'):
    """从保存的权重文件加载SAC模型"""
    # 加载配置文件
    config = ConfigParser()
    config.read('config.ini')
    agent_args = Dict(config, 'sac')
    
    # 获取环境信息
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建模型但不使用TensorBoard记录器
    model = SAC(None, device, state_dim, action_dim, agent_args)
    
    # 加载模型权重
    # 注意：这里保持与原始方法相同的加载方式，但实际上我们需要根据文件格式调整
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"模型从 {model_path} 加载成功")
    
    return model

# --- 主脚本 ---

if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 创建环境
    env = gym.make(ENV_ID)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载预训练的SAC模型
    if not os.path.exists(MODEL_PATH):
        print(f"错误：在 {MODEL_PATH} 未找到模型文件")
        print("请为Walker2d-v2训练一个SAC智能体并保存。")
        exit()

    print(f"正在从 {MODEL_PATH} 加载模型")
    # 使用自定义加载函数替代SB3的加载函数
    model = load_sac_model(MODEL_PATH, env, device)
    print("模型已加载。")

    # 获取原始参数
    original_params = get_params_dict(model)
    # print("--------------------------------")
    # print(original_params['logstd'])
    # print(original_params['layers.0.weight'])
    # 生成两个随机的滤波归一化方向 [引用: 73]
    print("正在生成方向...")
    direction1 = get_filter_normalized_direction(model, original_params)
    direction2 = get_filter_normalized_direction(model, original_params)
    
    # 创建网格
    alphas = np.linspace(-RANGE, RANGE, N_GRID_POINTS)
    betas = np.linspace(-RANGE, RANGE, N_GRID_POINTS)
    # alphas = [-0.1, 0, 0.1]
    # betas = [-0.1, 0, 0.1]
    reward_surface = np.zeros((N_GRID_POINTS, N_GRID_POINTS))

    # 在每个网格点评估奖励
    print(f"正在评估奖励曲面（{N_GRID_POINTS}x{N_GRID_POINTS} 网格）...")
    total_points = N_GRID_POINTS * N_GRID_POINTS
    current_point = 0
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            current_point += 1
            print(f"  正在评估点 ({i+1}, {j+1}) / ({N_GRID_POINTS}, {N_GRID_POINTS}) - {current_point}/{total_points}")

            # 计算扰动参数: theta' = theta + alpha*d1 + beta*d2 [引用: 73]
            perturbed_params = collections.OrderedDict()
            # print(original_params['logstd'])
            for name, param in original_params.items():
                # 确保计算在正确的设备上进行
                # print(f"original_params参数 {name} 的形状: {param.shape}")
                d1_tensor = direction1[name].to(param.device)
                d2_tensor = direction2[name].to(param.device)
                # perturbed_params[name] = param + a * d1_tensor + b * d2_tensor
                perturbed_params[name] = param + alpha * d1_tensor + beta * d2_tensor
                # print(f"perturbed_params参数 {name} 的形状: {perturbed_params[name].shape}")
            # print(original_params['logstd'])
            # 设置参数
            set_params_dict(model, perturbed_params)

            # 评估策略
            mean_reward = evaluate_policy_reward(model, env, n_eval_episodes=N_EVAL_EPISODES)
            reward_surface[i, j] = mean_reward
            print(f"    alpha={alpha:.2f}, beta={beta:.2f} -> 奖励={mean_reward:.2f}")

    # 将原始参数恢复到模型（重要！）
    set_params_dict(model, original_params)
    env.close()
    print("评估完成。")

    # 绘制奖励曲面
    print("正在绘制曲面...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(alphas, betas)

    # 使用plot_surface
    surf = ax.plot_surface(X, Y, reward_surface.T, cmap='viridis', edgecolor='none') # 由于循环顺序，需要转置

    ax.set_xlabel('Alpha (方向 1)')
    ax.set_ylabel('Beta (方向 2)')
    ax.set_zlabel(f'平均回合奖励（非折扣）在 {N_EVAL_EPISODES} 个回合上')
    ax.set_title(f'SAC在{ENV_ID}上的奖励曲面')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=30., azim=120) # 调整视角
    plt.tight_layout()
    plt.savefig("reward_surface_sac_walker2d.png")
    print("图表已保存为 reward_surface_sac_walker2d.png")
    # plt.show()