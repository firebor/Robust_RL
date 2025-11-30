import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
import os
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
import logging
from datetime import datetime
import json


# Create save directories
os.makedirs('./model', exist_ok=True)
os.makedirs('./data_train', exist_ok=True)


def load_model(model_path, agent, state_norm=None):
    """
    Load saved model and state normalizer
    
    Args:
        model_path: Path to the model file
        agent: Agent instance
        state_norm: State normalizer instance, if None create a new one
        
    Returns:
        agent: Agent with loaded weights
        state_norm: State normalizer with loaded statistics
    """
    # Load model
    save_dict = torch.load(model_path, map_location=agent.device)
    agent.load_state_dict(save_dict['agent_state_dict'])
    print(f"Loaded model weights: {model_path}")
    
    # Load state normalizer parameters
    if state_norm is not None:
        # Infer normalizer parameters path from model path
        model_dir = os.path.dirname(model_path)
        norm_stats_dir = os.path.join(os.path.dirname(os.path.dirname(model_dir)), 'normalization_stats')
        env_name = os.path.basename(os.path.dirname(model_dir))
        seed = model_path.split('seed')[-1].split('/')[0]
        norm_stats_path = os.path.join(norm_stats_dir, f'{env_name}_PPO_seed{seed}', 'normalization_stats.npy')
        
        if os.path.exists(norm_stats_path):
            # Load normalizer parameters
            stats = np.load(norm_stats_path, allow_pickle=True).item()
            state_norm.running_ms.n = stats['sample_count']
            state_norm.running_ms.mean = np.array(stats['mean'])
            state_norm.running_ms.S = np.array(stats['sum_squared_diff'])
            state_norm.running_ms.std = np.array(stats['std'])
            print(f"Loaded state normalizer parameters: {norm_stats_path}")
        else:
            print(f"Warning: State normalizer parameters file not found: {norm_stats_path}")
    
    return agent, state_norm


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        state = env.reset()[0]
        if args.use_state_norm:
            state = state_norm(state, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            action = agent.evaluate(state)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                scaled_action = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                scaled_action = action
            next_state, reward, terminated, truncated, _ = env.step(scaled_action)
            done = terminated or truncated
            if args.use_state_norm:
                next_state = state_norm(next_state, update=False)
            episode_reward += reward
            state = next_state
        evaluate_reward += episode_reward

    return evaluate_reward / times


def setup_logger(log_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_normalization_stats(state_norm, save_dir):
    """
    Save state normalizer statistics
    
    Args:
        state_norm: State normalizer instance
        save_dir: Directory to save the statistics
    """
    stats = {
        'sample_count': state_norm.running_ms.n,
        'mean': state_norm.running_ms.mean.tolist(),
        'sum_squared_diff': state_norm.running_ms.S.tolist(),
        'std': state_norm.running_ms.std.tolist()
    }
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Save as JSON file
    with open(os.path.join(save_dir, 'normalization_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    # Save as NumPy file
    np.save(os.path.join(save_dir, 'normalization_stats.npy'), stats)


def main(args):
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    # args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    args.device = torch.device("cpu")
    print(f"Using device: {args.device}")

    # Create save directories
    os.makedirs(f'./train_result/{args.exp_name}/{args.env_name}', exist_ok=True)
    
    # Create TensorBoard log directory
    log_dir_temp = f'./train_result/{args.exp_name}/{args.env_name}/tensorboard_logs/{args.env_name}_PPO_seed{args.seed}_noise'
    log_dir = log_dir_temp
    counter = 0
    while os.path.exists(log_dir):
        counter += 1
        log_dir = f"{log_dir_temp}({counter})"
    
    os.makedirs(name=log_dir, exist_ok=True)
    os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs', exist_ok=True)
    
    # Set log file paths
    if counter == 0:
        log_log_dir = f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_PPO_seed{args.seed}.log'
        model_dir = f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_PPO_seed{args.seed}'
    else:
        log_log_dir = f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_PPO_seed{args.seed}({counter}).log'
        model_dir = f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_PPO_seed{args.seed}({counter})'
    
    # Create model weights directory
    os.makedirs(name=model_dir, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(log_log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize environment
    if args.use_rle:
        # RLE模式：创建多个环境实例
        envs = []
        for i in range(args.num_workers):
            env = gym.make(args.env_name)
            envs.append(env)
        env_evaluate = gym.make(args.env_name)
    else:
        # 单环境模式
        env = gym.make(args.env_name)
        env_evaluate = gym.make(args.env_name)
    
    args.state_dim = env.observation_space.shape[0] if not args.use_rle else envs[0].observation_space.shape[0]
    args.action_dim = env.action_space.shape[0] if not args.use_rle else envs[0].action_space.shape[0]
    args.max_action = float(env.action_space.high[0]) if not args.use_rle else float(envs[0].action_space.high[0])
    args.max_episode_steps = env._max_episode_steps if not args.use_rle else envs[0]._max_episode_steps
    
    logger.info(f"Environment: {args.env_name}")
    logger.info(f"State dimension: {args.state_dim}")
    logger.info(f"Action dimension: {args.action_dim}")
    logger.info(f"Maximum action value: {args.max_action}")
    logger.info(f"Maximum episode steps: {args.max_episode_steps}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Random seed: {args.seed}")
    if args.use_rle:
        logger.info(f"RLE enabled with {args.num_workers} workers")
        logger.info(f"RLE feature size: {args.rle_feature_size}")
        logger.info(f"RLE switch steps: {args.rle_switch_steps}")
        logger.info(f"RLE intrinsic coefficient: {args.rle_int_coef}")
    
    evaluate_num = 0
    evaluate_rewards = []
    total_steps = 0
    
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
    # Load model if specified
    if args.load_model:
        agent, state_norm = load_model(args.load_model, agent)
        logger.info(f"Loaded model: {args.load_model}")
    
    # Initialize state normalizer
    state_norm = Normalization(shape=args.state_dim)
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_normalizer = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaler = RewardScaling(shape=1, gamma=args.gamma)
    
    # Training loop
    while total_steps < args.max_train_steps:
        if args.use_rle:
            # RLE模式：多worker并行训练
            states = []
            dones = []
            
            # 重置所有环境
            for i, env in enumerate(envs):
                state = env.reset()[0]
                if args.use_state_norm:
                    state = state_norm(state)
                states.append(state)
                dones.append(False)
            
            episode_steps = 0
            all_done = False
            
            while not all_done and episode_steps < args.max_episode_steps:
                episode_steps += 1
                total_steps += args.num_workers
                
                # 为每个worker选择动作
                actions = []
                action_logprobs = []
                intrinsic_rewards = []
                goals = []
                worker_ids = []
                rle_features = []
                
                for i, (env, state) in enumerate(zip(envs, states)):
                    if not dones[i]:
                        action, action_logprob = agent.choose_action(state)
                        if args.policy_dist == "Beta":
                            action = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
                        
                        # RLE: 计算内在奖励
                        if args.use_rle:
                            state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(args.device)
                            int_reward, features = agent.rle_manager.compute_intrinsic_reward(state_tensor, torch.tensor([i]))
                            goal = agent.rle_manager.current_goals[i].cpu().numpy()
                        else:
                            int_reward = 0.0
                            features = np.zeros(args.rle_feature_size)
                            goal = np.zeros(args.rle_feature_size)
                        
                        actions.append(action)
                        action_logprobs.append(action_logprob)
                        intrinsic_rewards.append(int_reward.item())
                        goals.append(goal)
                        worker_ids.append(i)
                        rle_features.append(features.detach().cpu().numpy())
                    else:
                        # 如果环境已完成，使用零填充
                        actions.append(np.zeros(args.action_dim))
                        action_logprobs.append(np.zeros(args.action_dim))
                        intrinsic_rewards.append(0.0)
                        goals.append(np.zeros(args.rle_feature_size))
                        worker_ids.append(i)
                        rle_features.append(np.zeros(args.rle_feature_size))
                
                # 执行环境步骤
                next_states = []
                rewards = []
                new_dones = []
                dead_or_wins = []
                
                for i, (env, action) in enumerate(zip(envs, actions)):
                    if not dones[i]:
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        dead_or_win = terminated
                        
                        if args.use_reward_norm:
                            reward = reward_normalizer(reward)
                        elif args.use_reward_scaling:
                            reward = reward_scaler(reward)
                        
                        if args.use_state_norm:
                            next_state = state_norm(next_state)
                        
                        next_states.append(next_state)
                        rewards.append(reward)
                        new_dones.append(done)
                        dead_or_wins.append(dead_or_win)
                    else:
                        next_states.append(states[i])
                        rewards.append(0.0)
                        new_dones.append(True)
                        dead_or_wins.append(False)
                
                # 存储经验
                for i in range(args.num_workers):
                    if not dones[i]:
                        replay_buffer.store(
                            states[i], actions[i], action_logprobs[i], rewards[i], 
                            next_states[i], dead_or_wins[i], new_dones[i],
                            intrinsic_rewards[i], goals[i], worker_ids[i], rle_features[i]
                        )
                
                # 更新状态和完成标志
                states = next_states
                dones = new_dones
                all_done = all(dones)
                
                # RLE: 更新目标向量
                if args.use_rle:
                    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=args.device)
                    agent.rle_manager.step(dones_tensor)
                
                # 更新策略
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer, total_steps)
                    replay_buffer.count = 0
                
                # 评估策略
                if is_evl_reward(total_steps, args):
                    evaluate_num += 1
                    with torch.no_grad():
                        evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    logger.info(f"Evaluation {evaluate_num}, Total steps: {total_steps}, Reward: {evaluate_reward}")
                    writer.add_scalar('evaluate_reward', evaluate_reward, total_steps)
                    torch.save(agent.state_dict(), model_dir+'/agent_' + str(total_steps) + "_" + str(evaluate_reward))
        else:
            # 单环境模式（原始代码）
            state = env.reset()[0]
            if args.use_state_norm:
                state = state_norm(state)
            episode_steps = 0
            done = False
            
            while not done and episode_steps < args.max_episode_steps:
                episode_steps += 1
                total_steps += 1
                
                # Choose action
                action, action_logprob = agent.choose_action(state)
                if args.policy_dist == "Beta":
                    action = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
                next_state, reward, terminated, truncated, _ = env.step(action)
                if args.use_reward_norm:
                    reward = reward_normalizer(reward)
                elif args.use_reward_scaling:
                    reward = reward_scaler(reward)
                done = terminated or truncated
                
                if args.use_state_norm:
                    next_state = state_norm(next_state)
                
                done_without_next_state = terminated or truncated
                # Store experience
                replay_buffer.store(state, action, action_logprob, reward, next_state, done_without_next_state, done)
                state = next_state
                
                # Update policy
                if replay_buffer.count == args.batch_size:
                    agent.update(replay_buffer,total_steps)
                    replay_buffer.count = 0
                
                # Evaluate policy
                if is_evl_reward(total_steps, args):
                    evaluate_num += 1
                    with torch.no_grad():
                        evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                    evaluate_rewards.append(evaluate_reward)
                    logger.info(f"Evaluation {evaluate_num}, Total steps: {total_steps}, Reward: {evaluate_reward}")
                    writer.add_scalar('evaluate_reward', evaluate_reward, total_steps)
                    torch.save(agent.state_dict(), model_dir+'/agent_' + str(total_steps) + "_" + str(evaluate_reward))
    
    # Save state normalizer parameters after training
    if args.use_state_norm:
        norm_stats_dir = f'./train_result/{args.exp_name}/{args.env_name}/normalization_stats/{args.env_name}_PPO_seed{args.seed}'
        save_normalization_stats(state_norm, norm_stats_dir)
        logger.info(f"Saved state normalizer parameters to: {norm_stats_dir}")
    
    # 关闭环境
    if args.use_rle:
        for env in envs:
            env.close()
    else:
        env.close()
    env_evaluate.close()
    
    writer.close()
    logger.info("Training completed")

def is_evl_reward(steps, args):
    eval_interval = args.evaluate_freq
    if steps > 2000000:
        eval_interval = eval_interval/10
    
    if steps % eval_interval == 0 and steps != 0:
        return True
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e5, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=bool, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load model")
    parser.add_argument("--update_freq", type=int, default=100, help="Frequency of policy updates")
    parser.add_argument("--exp_name", type=str, default="default_experiment", help="Name of the experiment")
    parser.add_argument("--env_name", type=str, default='Walker2d-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Whether to use CUDA")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--noise_multiplier", type=float, default=0.2, help="设置高斯噪声的标准差(默认0.2)")
    
    # RLE相关参数
    parser.add_argument("--use_rle", type=bool, default=True, help="是否启用RLE探索")
    parser.add_argument("--num_workers", type=int, default=8, help="RLE并行worker数量")
    parser.add_argument("--rle_feature_size", type=int, default=16, help="RLE特征维度")
    parser.add_argument("--rle_switch_steps", type=int, default=500, help="RLE目标切换步数")
    parser.add_argument("--rle_tau", type=float, default=0.005, help="RLE软更新参数")
    parser.add_argument("--rle_int_coef", type=float, default=0.01, help="RLE内在奖励系数")

    args = parser.parse_args()

    main(args)
