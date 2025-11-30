from configparser import ConfigParser
from argparse import ArgumentParser
from log import setup_logger

import torch
import gym
import numpy as np
import os
from sac import SAC
from utils.utils import make_transition, Dict, RunningMeanStd
from torch.utils.tensorboard import SummaryWriter
import random

# os.environ['LD_LIBRARY_PATH'] = "/home/cmdout/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# os.makedirs('./model_weights', exist_ok=True)
opio = [1774, 3118, 3609, 4888, 5480, 5463, 5302, 5115, 4968, 4591, 4118]




def evaluate_policy(env, agent, eval_seed=None):
    times = 50  # 执行50次评估并计算平均值
    evaluate_reward = 0
    
    for i in range(times):
        # 如果提供了评估种子，则每次使用不同但确定性的种子
        if eval_seed is not None:
            # 为每次评估生成不同但确定性的种子
            episode_seed = eval_seed + i * 1000
            state = env.reset(seed=episode_seed)[0]
        else:
            state = env.reset()[0]
            
        done = False
        truncated = False
        episode_reward = 0
        
        while not done and not truncated:
            action, _ = agent.get_action(
                torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=agent.device), 0))
            action = action.cpu().detach().numpy()
            state_, r, done, truncated, *_ = env.step(action.reshape(-1))
            episode_reward += r
            state = state_
            
        evaluate_reward += episode_reward
        
    return int(evaluate_reward / times)


def test_policy(env_name, agent, DISTURBANCE_DICT, writer, seed=42, logger=None):
    # 设置固定的测试种子
    TEST_SEED = seed
    
    mass_list = DISTURBANCE_DICT[env_name]['mass_list']
    friction_list = DISTURBANCE_DICT[env_name]['friction_list']
    reward_lists = []
    reward_list = []
    
    # 测试不同质量的影响
    for mass in mass_list:
        env = gym.make(env_name)
        # 设置环境种子确保一致性
        env.reset(seed=TEST_SEED)
        env.action_space.seed(TEST_SEED)
        
        cur_mass = env.model.body_mass
        new_mass = mass * cur_mass
        env.model.body_mass[:] = new_mass
        reward = evaluate_policy(env, agent, eval_seed=TEST_SEED)
        writer.add_scalar("reward/mass", reward, mass)
        reward_list.append(reward)
        print(f"Mass: {mass}, Reward: {reward}")
        logger.info(f"Mass: {mass}, Reward: {reward}")
        
    # 保存质量测试结果
    reward_lists.append(reward_list)
    reward_list = []
    # 测试不同摩擦力的影响
    for friction in friction_list:
        env = gym.make(env_name)
        # 设置环境种子确保一致性
        env.reset(seed=TEST_SEED)
        env.action_space.seed(TEST_SEED)
        
        cur_friction = env.model.geom_friction
        new_friction = friction * cur_friction
        env.model.geom_friction[:] = new_friction
        reward = evaluate_policy(env, agent, eval_seed=TEST_SEED)
        writer.add_scalar("reward/friction", reward, friction)
        reward_list.append(reward)
        print(f"Friction: {friction}, Reward: {reward}")
        logger.info(f"Friction: {friction}, Reward: {reward}")
        

    # 保存摩擦力测试结果
    reward_lists.append(reward_list)
    # 创建保存结果的目录
    os.makedirs(f'./test_result/{env_name}/result_numpy', exist_ok=True)
    np.save(f'./test_result/{env_name}/result_numpy/{env_name}_seed{seed}.npy',
            np.array(reward_lists))
            
    print("测试完成，结果已保存")
    logger.info("测试完成，结果已保存")


def get_model_path(env_name, seed):
    # 在best_model下找对应的文件夹
    model_dir = f'./approximate_model/{env_name}'
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        return None
    
    # 查找匹配的文件夹
    target_folder_name = f"_seed{seed}"
    for folder_name in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder_name)
        if os.path.isdir(folder_path) and target_folder_name in folder_name:
            # 找到匹配的文件夹，返回其中的第一个文件
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if files:
                model_path = os.path.join(folder_path, files[0])
                print(f"找到模型文件: {model_path}")
                return model_path
            else:
                print(f"在文件夹 {folder_path} 中没有找到文件")
    
    print(f"在 {model_dir} 中没有找到匹配的文件夹 {target_folder_name}")
    return None


def main_test(args, agent_args):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建测试结果目录    
    prefix_dir = f'./test_result/{args.env_name}'
    os.makedirs(prefix_dir, exist_ok=True)

    # 获取模型路径
    model_path = get_model_path(args.env_name, args.seed)
    if model_path is None:
        print(f"未找到模型文件，请检查模型名称是否正确")
        return
        
    DISTURBANCE_DICT = {
        "Hopper-v2": {"mass_list": np.linspace(0.2, 1.6, 11), "friction_list": np.linspace(0.7, 1.3, 11)},
        "Walker2d-v2": {"mass_list": np.linspace(0.5, 1.5, 11), "friction_list": np.linspace(0.4, 1.8, 11)},
        "HalfCheetah-v2": {"mass_list": np.linspace(0.6, 1.3, 11), "friction_list": np.linspace(0.1, 2.7, 11)},
        "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.6, 2.6, 11),
                                      "friction_list": np.linspace(0.4, 2.4, 11)}}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建测试日志目录
    log_dir_temp=f'{prefix_dir}/test_tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}'
    log_dir=f'{prefix_dir}/test_tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}'
    counter=0
    while os.path.exists(log_dir):
        counter+=1
        log_dir=f"{log_dir_temp}({counter})"
    
    # 新建日志
    os.makedirs(name=log_dir,exist_ok=True)
    os.makedirs(name=f'{prefix_dir}/test_logs_logs',exist_ok=True)
    log_log_dir=""
    if counter==0:
        log_log_dir=f'{prefix_dir}/test_logs_logs/{args.env_name}_{args.algo}_seed{args.seed}.log'
    else:
        log_log_dir=f'{prefix_dir}/test_logs_logs/{args.env_name}_{args.algo}_seed{args.seed}({counter}).log'
    logger = setup_logger(log_dir=log_log_dir)
    
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    agent = SAC(writer, device, state_dim, action_dim, agent_args)
    print(f"使用标准SAC算法，seed={args.seed}")
    logger.info(f"使用标准SAC算法，seed={args.seed}")
        
    agent.load_state_dict(torch.load(model_path , map_location=torch.device('cpu')))
    logger.info(f"加载模型: {model_path}")
    test_policy(args.env_name, agent, DISTURBANCE_DICT, writer, seed=args.seed, logger=logger)


if __name__ == '__main__':
    parser = ArgumentParser('parameters')


    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=2501, help='number of epochs, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='agent_1870_5415', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=20, help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.2, help='reward scaling(default : 0.1)')


    parser.add_argument("--env_name", type=str, default='Walker2d-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--algo", type=str, default='sac', help='algorithm to adjust (default : ppo)')
    parser.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")


    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('./config.ini')
    print(parser)
    agent_args = Dict(parser, args.algo)
    main_test(args, agent_args)
    # main(args, agent_args)