import time
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
from Pd_sac import PDSAC
from typing import Dict as TypeDict, OrderedDict
from multiprocessing import Pool, set_start_method
import multiprocessing
import logging
import tempfile
import shutil
import plotly.graph_objects as go  # 添加plotly库
gym.logger.set_level(logging.ERROR)
# --- 配置参数 ---
ENV_ID = "Walker2d-v2"  # 环境ID
MODEL_PATH = "/home/wsl/code/Robust_RL/PDSAC/best_model/grad-norm/Walker2d-v2/Walker2d-v2_sac_seed2025_noise0.2/agent_1380000_6023"  # <<<=== 在此加载预训练的SAC模型
N_GRID_POINTS = 50  # 网格分辨率（例如：论文中某些环境使用31x31 [引用: 256]）
RANGE = 0.2         # alpha和beta的范围（例如：Walker2d在论文中使用[-1, 1] [引用: 256]）
N_EVAL_EPISODES = 10 # 每个点评估的回合数（增加以提高精度 [引用: 256]）
SEED = 1024           # 用于复现随机方向的种子
NUM_PROCESSES = 16   # 并行进程数，根据CPU核心数调整

# --- 辅助函数 ---

def get_params_dict(model: PDSAC) -> OrderedDict:
    """提取模型参数到有序字典中。"""
    params = collections.OrderedDict()
    # 使用深拷贝创建参数的独立副本，并分离梯度信息
    for name, param in model.actor.named_parameters():
        params[name] = copy.deepcopy(param.detach())
    # 包含critic参数
    # for name, param in model.q_1.named_parameters():
    #     params[name] = copy.deepcopy(param.detach())
    # for name, param in model.q_2.named_parameters():
    #     params[name] = copy.deepcopy(param.detach())
    # 包含目标critic参数（如果需要）
    # for name, param in model.target_q_1.named_parameters():
    #     params[name] = copy.deepcopy(param.detach())
    # for name, param in model.target_q_2.named_parameters():
    #     params[name] = copy.deepcopy(param.detach())
    return params

def set_params_dict(model: PDSAC, params_dict: OrderedDict):
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

def get_filter_normalized_direction(model: PDSAC, params_dict: OrderedDict) -> OrderedDict:
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
            normalized_noise = (noise * (param_norm / noise_norm)).detach()
        else:
            normalized_noise = torch.zeros_like(noise).detach() # 或者适当处理

        direction[name] = normalized_noise

    return direction

def get_orthogonal_direction(model: PDSAC, params_dict: OrderedDict, direction1: OrderedDict) -> OrderedDict:
    """生成与第一个方向正交的第二个方向（使用Gram-Schmidt正交化）"""
    # 首先生成一个随机方向
    direction2 = collections.OrderedDict()
    device = next(model.actor.parameters()).device
    
    for name, param in params_dict.items():
        # 生成随机噪声
        noise = torch.randn(param.size(), device=device)
        
        # 计算与第一个方向的点积
        dot_product = torch.sum(noise * direction1[name])
        
        # Gram-Schmidt正交化：从第二个方向减去它在第一个方向上的投影
        orthogonal_noise = noise - dot_product * direction1[name]
        
        # 计算范数
        noise_norm = torch.linalg.norm(orthogonal_noise)
        param_norm = torch.linalg.norm(param)
        
        # 归一化
        if noise_norm > 1e-10:
            normalized_noise = (orthogonal_noise * (param_norm / noise_norm)).detach()
        else:
            normalized_noise = torch.zeros_like(noise).detach()
            
        direction2[name] = normalized_noise
        
    return direction2

@torch.no_grad()
def evaluate_policy_reward(model: PDSAC, env: gym.Env, n_eval_episodes: int) -> float:
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
                # 使用PDSAC的get_action方法
                state_tensor = torch.tensor(obs, dtype=torch.float, device=model.device).unsqueeze(0)
                action, _ = model.get_action(state_tensor)
                action = action.cpu().detach().numpy().reshape(-1)
                
                # 确保动作在有效范围内
                action = np.clip(action, env.action_space.low, env.action_space.high)
                
                obs, reward, done, truncated, info = env.step(action)
                # 累积回合的非折扣奖励
                episode_rewards.append(reward)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print(f"Current state: {obs}")
            print(f"Current action: {action}")
            return 0.0
            
        all_episode_rewards.append(sum(episode_rewards))

    mean_reward = np.mean(all_episode_rewards)
    # print(f"Mean reward: {mean_reward}")
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
    model = PDSAC(None, device, state_dim, action_dim, 0.1, agent_args)
    
    # 加载模型权重
    # 注意：这里保持与原始方法相同的加载方式，但实际上我们需要根据文件格式调整
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model successfully loaded from {model_path}")
    
    return model, config

def save_directions_to_temp(direction1, direction2):
    """将方向保存到临时目录中"""
    temp_dir = tempfile.mkdtemp(prefix="sac_directions_")
    print(f"创建临时目录: {temp_dir}")
    
    # 保存方向1
    dir1_path = os.path.join(temp_dir, "direction1.pt")
    torch.save(direction1, dir1_path)
    
    # 保存方向2
    dir2_path = os.path.join(temp_dir, "direction2.pt")
    torch.save(direction2, dir2_path)
    
    return temp_dir

def load_directions_from_temp(temp_dir):
    """从临时目录加载方向"""
    dir1_path = os.path.join(temp_dir, "direction1.pt")
    dir2_path = os.path.join(temp_dir, "direction2.pt")
    
    direction1 = torch.load(dir1_path)
    direction2 = torch.load(dir2_path)
    
    return direction1, direction2

# --- 主脚本 ---

@torch.no_grad()
def worker(args):
    """
    优化后的Worker函数：从临时文件加载方向，然后计算扰动参数
    """
    try:
        # 解包参数
        env_id, model_path, alpha, beta, n_eval_episodes, temp_dir = args
        
        # 创建环境
        env = gym.make(env_id)
        
        # 获取设备
        device = 'cpu'  # 为了简化多进程通信，我们使用CPU
        
        # 加载配置文件
        config = ConfigParser()
        config.read('config.ini')
        agent_args = Dict(config, 'sac')
        
        # 获取环境维度
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # 创建新模型并加载原始权重
        model = PDSAC(None, device, state_dim, action_dim, 0.1, agent_args)
        model.load_state_dict(torch.load(model_path, map_location=device))

        
        # 获取原始参数
        original_params = get_params_dict(model)
        
        
        # 从临时文件加载方向
        direction1, direction2 = load_directions_from_temp(temp_dir)
        
        # 计算扰动参数
        perturbed_params = collections.OrderedDict()
        for name, param in original_params.items():
            if name in direction1 and name in direction2:
                d1_tensor = direction1[name]
                d2_tensor = direction2[name]
                perturbed_params[name] = param.detach() + alpha * d1_tensor + beta * d2_tensor
            else:
                perturbed_params[name] = param.detach()
        
        # 设置扰动后的参数到模型
        set_params_dict(model, perturbed_params)
        
        # 评估策略
        mean_reward = evaluate_policy_reward(model, env, n_eval_episodes)
        
        # 清理资源
        env.close()
        
        return (alpha, beta, mean_reward)
    except Exception as e:
        print(f"Worker错误: {e}")
        import traceback
        traceback.print_exc()
        return (alpha, beta, 0.0)  # 错误时返回零奖励
    finally:
        # 确保资源正确清理
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    # 设置随机种子以确保可复现性
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 设置多进程启动方法
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 创建环境
    env = gym.make(ENV_ID)

    # 设置设备
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 在 {MODEL_PATH} 未找到模型文件")
        print(f"请为 {ENV_ID} 训练一个SAC智能体并保存。")
        exit()

    print(f"正在从 {MODEL_PATH} 加载模型")
    
    # 加载模型
    model, config = load_sac_model(MODEL_PATH, env, device)
    print("模型已加载。")
    
    # 获取原始参数
    original_params = get_params_dict(model)
    
    # 生成两个随机的滤波归一化方向
    print("正在生成方向...")
    direction1 = get_filter_normalized_direction(model, original_params)
    direction2 = get_filter_normalized_direction(model, original_params)
    # direction2 = get_orthogonal_direction(model, original_params, direction1)
    # print(direction1["fc1.weight"])
    # print(direction2["fc1.weight"])
    # print(direction1["fc1.weight"] @ direction2["fc1.weight"].transpose(0,1))
    
    # 将方向保存到临时文件
    temp_dir = save_directions_to_temp(direction1, direction2)
    
    try:
        # 创建网格
        alphas = np.linspace(-RANGE, RANGE, N_GRID_POINTS)
        betas = np.linspace(-RANGE, RANGE, N_GRID_POINTS)
        reward_surface = np.zeros((N_GRID_POINTS, N_GRID_POINTS))

        # 在每个网格点评估奖励 - 使用并行处理
        print(f"正在评估奖励曲面 ({N_GRID_POINTS}x{N_GRID_POINTS} 网格) 使用 {NUM_PROCESSES} 个进程...")
        
        # 准备任务列表 - 只传递必要的信息和临时目录路径
        tasks = []
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                # 将任务添加到列表
                tasks.append((ENV_ID, MODEL_PATH, alpha, beta, N_EVAL_EPISODES, temp_dir))
        
        # 使用多进程池并行处理
        total_tasks = len(tasks)
        with Pool(processes=NUM_PROCESSES) as pool:
            results = []
            # 使用imap_unordered以获得更好的性能，结果会以完成顺序返回
            for idx, result in enumerate(pool.imap_unordered(worker, tasks)):
                alpha, beta, reward = result
                # 找到对应的索引
                i = np.where(alphas == alpha)[0][0]
                j = np.where(betas == beta)[0][0]
                reward_surface[i, j] = reward
                
                # 显示进度
                print(f"  已评估点 {idx+1}/{total_tasks}: alpha={alpha:.2f}, beta={beta:.2f} -> reward={reward:.2f}")
        
        env.close()
        print("评估完成。")

        # 绘制奖励曲面
        print("正在绘制曲面...")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(alphas, betas)

        # 使用plot_surface，更改颜色映射为coolwarm
        surf = ax.plot_surface(X, Y, reward_surface.T, cmap='coolwarm', edgecolor='none') # 由于循环顺序，需要转置

        ax.set_xlabel('Alpha (Direction 1)')
        ax.set_ylabel('Beta (Direction 2)')
        ax.set_zlabel(f'Average Episode Reward (undiscounted) over {N_EVAL_EPISODES} episodes')
        ax.set_title(f'Reward Surface of SAC on {ENV_ID}')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.view_init(elev=30., azim=120) # 调整视角
        plt.tight_layout()
        plt.savefig("reward_surface_sac_walker2d.png")
        print("绘图已保存为 reward_surface_sac_walker2d.png")
        plt.show()
            # 保存原始数据为npz文件
        np.savez("reward_surface_data.npz", 
                alphas=alphas, 
                betas=betas, 
                reward_surface=reward_surface)
        print("原始数据已保存为 reward_surface_data.npz")
        
        # 使用plotly创建交互式3D可视化
        X_mesh, Y_mesh = np.meshgrid(alphas, betas)
        fig_plotly = go.Figure(data=[go.Surface(
            x=X_mesh, 
            y=Y_mesh, 
            z=reward_surface.T,
            colorscale='Viridis')])
        
        fig_plotly.update_layout(
            title=f'奖励曲面 - {ENV_ID}',
            scene=dict(
                xaxis_title='Alpha (方向 1)',
                yaxis_title='Beta (方向 2)',
                zaxis_title=f'平均回合奖励 (未折扣, {N_EVAL_EPISODES}回合)'
            ),
            width=800,
            height=800
        )
        
        # 保存为交互式HTML文件
        fig_plotly.write_html("reward_surface_interactive.html")
        print("交互式3D可视化已保存为 reward_surface_interactive.html")
            
            
    finally:
        # 清理临时目录
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)