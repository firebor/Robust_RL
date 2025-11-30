from configparser import ConfigParser
from argparse import ArgumentParser
from log import setup_logger

import torch
import gym
import numpy as np
import os
from NISAC.noise_injected_sac_old import NISAC
from utils.utils import make_transition, Dict, RunningMeanStd
from torch.utils.tensorboard import SummaryWriter
import random

# os.environ['LD_LIBRARY_PATH'] = "/home/cmdout/.mujoco/mujoco210/bin:/usr/lib/nvidia"
os.environ
# os.makedirs('./model_weights', exist_ok=True)

opio = [1774, 3118, 3609, 4888, 5480, 5463, 5302, 5115, 4968, 4591, 4118]


def evaluate_policy(env, agent):
    times=100
    evaluate_reward = 0
    
    for i in range(times):
        # 使用完全随机的种子
        state = env.reset()[0]
        done = False
        truncated = False
        episode_reward = 0
        while not done and not truncated:
            action, _ = agent.get_action(
                torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=agent.device), 0))
            action = action.cpu().detach().numpy()
            state_, r, done, truncated,*_ = env.step(action.reshape(-1))
            episode_reward += r
            state = state_
            
        evaluate_reward += episode_reward
        
    return int(evaluate_reward / times)


# 测试策略函数
def test_policy(env_name, agent, DISTURBANCE_DICT, writer, seed=42,noise_multiplier=0.1,logger=None):
    # 设置固定的测试种子
    
    mass_list = DISTURBANCE_DICT[env_name]['mass_list']
    friction_list = DISTURBANCE_DICT[env_name]['friction_list']
    reward_lists=[]
    reward_list = []
    
    # 测试不同质量的影响
    for mass in mass_list:
        env = gym.make(env_name)
        # 设置环境种子确保一致性
        env.reset()
        env.action_space.seed()
        
        cur_mass = env.model.body_mass
        new_mass = mass * cur_mass
        env.model.body_mass[:] = new_mass
        reward = evaluate_policy(env, agent)
        # 将质量保留1位小数，用于日志显示
        mass_rounded = round(mass, 1)
        writer.add_scalar("reward/mass", reward, mass_rounded*10)
        reward_list.append(reward)
        print(f"Mass: {mass_rounded}, Reward: {reward}")
        logger.info(f"Mass: {mass_rounded}, Reward: {reward}")
    # 测试不同摩擦力的影响
    reward_lists.append(reward_list)
    reward_list = []
    for friction in friction_list:
        env = gym.make(env_name)
        # 设置环境种子确保一致性
        env.reset()
        env.action_space.seed()
        
        cur_friction = env.model.geom_friction
        new_friction = friction * cur_friction
        env.model.geom_friction[:] = new_friction
        reward = evaluate_policy(env, agent)
        # 将摩擦系数保留1位小数，用于日志显示
        friction_rounded = round(friction, 1)
        writer.add_scalar("reward/friction", reward, friction_rounded*10)
        reward_list.append(reward)
        print(f"Friction: {friction_rounded}, Reward: {reward}")
        logger.info(f"Friction: {friction_rounded}, Reward: {reward}")
    reward_lists.append(reward_list)
    os.makedirs(f'./test_result/{env_name}/result_numpy', exist_ok=True)
    np.save(f'./test_result/{env_name}/result_numpy/{env_name}_seed{seed}_noise{noise_multiplier}.npy',
            np.array(reward_lists))
    print("测试完成，结果已保存")

def get_model_path(env_name,seed, noise_multiplier,exp_name):
    # 在best_model下找对应的文件夹
    model_dir = f'./best_model/{exp_name}/{env_name}'
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        return None
    
    # 查找匹配的文件夹
    target_folder_name = f"seed{seed}_noise{noise_multiplier}"
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
    prefix_dir=f'./test_result/{args.exp_name}/{args.env_name}'
    # 创建测试数据保存目录
    os.makedirs(prefix_dir, exist_ok=True)

    model_path = get_model_path(args.env_name,args.seed, args.noise_multiplier,args.exp_name)
    # model_path='/home/admin1/code/Robust-RL/PDSAC/train_result/grad-norm/Walker2d-v2/model_weights/Walker2d-v2_sac_seed7_noise0.5/agent_1500000_4181'
    if model_path is None:
        print(f"未找到模型文件，请检查模型名称是否正确")
        return
    
    
    DISTURBANCE_DICT = {
        "Hopper-v2": {"mass_list": np.linspace(0.2, 1.6, 11), "friction_list": np.linspace(0.7, 1.3, 11)},
        "Walker2d-v2": {"mass_list": np.linspace(0.1,2, 20), "friction_list": np.linspace(0.1,2,20)},
        "HalfCheetah-v2": {"mass_list": np.linspace(0.6, 1.3, 11), "friction_list": np.linspace(0.1, 2.7, 11)},
        "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.6, 2.6, 11),
                                      "friction_list": np.linspace(0.4, 2.4, 11)}}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    

    log_dir_temp=f'{prefix_dir}/test_tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}'
    log_dir=f'{prefix_dir}/test_tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}'
    counter=0
    while os.path.exists(log_dir):
        counter+=1
        log_dir=f"{log_dir_temp}({counter})"
    
    # 新建日志
    os.makedirs(name=log_dir,exist_ok=True)
    os.makedirs(name=f'{prefix_dir}/test_logs_logs',exist_ok=True)
    log_log_dir=""
    if counter==0:
        log_log_dir=f'{prefix_dir}/test_logs_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}.log'
    else:
        log_log_dir=f'{prefix_dir}/test_logs_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}({counter}).log'
    logger = setup_logger(log_dir=log_log_dir)
    logger.info(f"使用模型: {model_path}")
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)[0]
    env.action_space.seed(args.seed)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    agent = NISAC(writer, device, state_dim, action_dim, args.noise_multiplier, agent_args)
    agent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print(f"============使用PDSAC算法，seed={args.seed}, noise_multiplier={args.noise_multiplier}")
    test_policy(args.env_name, agent, DISTURBANCE_DICT, writer, seed=args.seed,logger=logger,noise_multiplier=args.noise_multiplier)


if __name__ == '__main__':
    parser = ArgumentParser('parameters')
    parser.add_argument("--env_name", type=str, default='Walker2d-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--algo", type=str, default='sac', help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--steps', type=int, default=1500000, help='number of steps, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='agent_1500_5527', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=20, help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.2, help='reward scaling(default : 0.1)')

    parser.add_argument("--exp_name", type=str, default='grad-norm', help='实验名称，用于保存结果')
    parser.add_argument("--noise_multiplier", type=float, default=0.2, help="设置高斯噪声的标准差(默认0.2)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")
    args = parser.parse_args()

    parser = ConfigParser()
    parser.read('./config.ini')
    # print(parser.sections())
    agent_args = Dict(parser, args.algo)
    main_test(args, agent_args)
    # main(args, agent_args)
    