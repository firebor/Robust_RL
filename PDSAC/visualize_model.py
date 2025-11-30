from configparser import ConfigParser
from argparse import ArgumentParser
import os
import numpy as np
import torch
import gym
import time
from Pd_sac import PDSAC
from utils.utils import Dict, make_transition
from log import setup_logger

def load_model(model_path, env_name, device, noise_multiplier, agent_args):
    """
    加载预训练的PDSAC模型
    
    Args:
        model_path: 模型路径
        env_name: 环境名称
        device: 设备（CPU/GPU）
        noise_multiplier: 噪声乘数
        agent_args: 代理参数
        
    Returns:
        agent: 加载了模型参数的PDSAC代理
        env: 创建的gym环境
    """
    env = gym.make(env_name, render_mode="human")
    env.reset()
    
    # 获取状态和动作空间维度
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建PDSAC代理
    writer = None  # 测试时不需要tensorboard写入器
    agent = PDSAC(writer, device, state_dim, action_dim, noise_multiplier, agent_args)
    
    # 加载模型参数
    print(f"正在加载模型: {model_path}")
    agent.load_state_dict(torch.load(model_path, map_location=device))
    
    return agent, env

def visualize_policy(env, agent, episodes=5, max_steps=1000, render_delay=0.01, logger=None):
    """
    在环境中可视化策略
    
    Args:
        env: gym环境
        agent: PDSAC代理
        episodes: 运行的回合数
        max_steps: 每个回合的最大步数
        render_delay: 渲染延迟（控制可视化速度）
        logger: 日志记录器
        
    Returns:
        avg_reward: 平均奖励
    """
    total_reward = 0
    
    for episode in range(episodes):
        state = env.reset()[0]
        episode_reward = 0
        steps = 0
        done = False
        truncated = False
        
        print(f"\n开始回合 {episode+1}/{episodes}")
        
        while not done and not truncated and steps < max_steps:
            # 获取动作
            with torch.no_grad():
                action, _ = agent.get_action(
                    torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=agent.device), 0))
            action = action.cpu().detach().numpy()
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action.reshape(-1))
            
            # 更新状态和奖励
            state = next_state
            episode_reward += reward
            steps += 1
            
            # 延迟渲染（控制可视化速度）
            time.sleep(render_delay)
            
            # 每100步打印一次信息
            if steps % 100 == 0:
                print(f"步数: {steps}, 当前奖励: {episode_reward:.2f}")
        
        print(f"回合 {episode+1} 结束，总步数: {steps}, 总奖励: {episode_reward:.2f}")
        if logger:
            logger.info(f"回合 {episode+1} 结束，总步数: {steps}, 总奖励: {episode_reward:.2f}")
        
        total_reward += episode_reward
    
    avg_reward = total_reward / episodes
    print(f"\n{episodes} 个回合的平均奖励: {avg_reward:.2f}")
    if logger:
        logger.info(f"{episodes} 个回合的平均奖励: {avg_reward:.2f}")
    
    return avg_reward

def visualize_with_disturbance(env_name, agent, device, disturbance_type='mass', factor=1.0, episodes=3, logger=None):
    """
    在有扰动的环境中可视化策略
    
    Args:
        env_name: 环境名称
        agent: PDSAC代理
        device: 设备（CPU/GPU）
        disturbance_type: 扰动类型（'mass'或'friction'）
        factor: 扰动因子
        episodes: 运行的回合数
        logger: 日志记录器
        
    Returns:
        avg_reward: 平均奖励
    """
    # 创建环境
    env = gym.make(env_name, render_mode="human")
    env.reset()
    
    # 应用扰动
    if disturbance_type == 'mass':
        cur_mass = env.model.body_mass.copy()
        new_mass = factor * cur_mass
        env.model.body_mass[:] = new_mass
        print(f"应用质量扰动，扰动因子: {factor}")
        if logger:
            logger.info(f"应用质量扰动，扰动因子: {factor}")
    elif disturbance_type == 'friction':
        cur_friction = env.model.geom_friction.copy()
        new_friction = factor * cur_friction
        env.model.geom_friction[:] = new_friction
        print(f"应用摩擦力扰动，扰动因子: {factor}")
        if logger:
            logger.info(f"应用摩擦力扰动，扰动因子: {factor}")
    
    # 可视化策略
    return visualize_policy(env, agent, episodes=episodes, logger=logger)

def main():
    parser = ArgumentParser('PDSAC模型可视化')
    parser.add_argument("--env_name", type=str, default='Walker2d-v2', 
                      help="环境名称，例如: 'Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2'等")
    parser.add_argument("--model_path", type=str, required=True, 
                      help="模型文件的路径")
    parser.add_argument("--algo", type=str, default='sac', 
                      help="算法名称 (默认: sac)")
    parser.add_argument("--episodes", type=int, default=5, 
                      help="测试回合数 (默认: 5)")
    parser.add_argument("--render_delay", type=float, default=0.01, 
                      help="渲染延迟，控制可视化速度 (默认: 0.01)")
    parser.add_argument("--noise_multiplier", type=float, default=0.2, 
                      help="高斯噪声的标准差 (默认: 0.2)")
    parser.add_argument("--disturbance", action="store_true", 
                      help="是否添加环境扰动")
    parser.add_argument("--disturbance_type", type=str, default='mass', 
                      help="扰动类型，可选：'mass'或'friction' (默认: mass)")
    parser.add_argument("--disturbance_factor", type=float, default=1.5, 
                      help="扰动因子 (默认: 1.5)")
    
    args = parser.parse_args()
    
    # 配置日志
    os.makedirs('./visualize_logs', exist_ok=True)
    logger = setup_logger(log_dir=f'./visualize_logs/visualize_{args.env_name}_{int(time.time())}.log')
    
    # 打印参数
    print("参数配置:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
        logger.info(f"{arg}: {getattr(args, arg)}")
    
    # 读取配置文件
    config_parser = ConfigParser()
    config_parser.read('./config.ini')
    agent_args = Dict(config_parser, args.algo)
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    agent, env = load_model(args.model_path, args.env_name, device, args.noise_multiplier, agent_args)
    
    # 运行可视化
    if args.disturbance:
        visualize_with_disturbance(
            args.env_name, 
            agent, 
            device, 
            disturbance_type=args.disturbance_type, 
            factor=args.disturbance_factor, 
            episodes=args.episodes,
            logger=logger
        )
    else:
        visualize_policy(env, agent, episodes=args.episodes, render_delay=args.render_delay, logger=logger)

if __name__ == "__main__":
    main() 