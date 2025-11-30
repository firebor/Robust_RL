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
import warnings
import logging
import re

# 全面屏蔽警告
warnings.filterwarnings('ignore')
# 设置日志级别
logging.getLogger().setLevel(logging.ERROR)
# 禁用gym的警告
gym.logger.set_level(logging.ERROR)

# os.environ['LD_LIBRARY_PATH'] = "/home/cmdout/.mujoco/mujoco210/bin:/usr/lib/nvidia"

# os.makedirs('./model_weights', exist_ok=True)
opio = [1774, 3118, 3609, 4888, 5480, 5463, 5302, 5115, 4968, 4591, 4118]

def worker(args):
    """
    工作进程函数，用于并行评估
    每个worker运行10次评估并返回平均奖励
    """
    try:
        env_name, model_state_dict, mass, friction, state_dim, action_dim = args
        
        # 将agent_args转换回Dict类
        from configparser import ConfigParser
        config = ConfigParser()
        config.read('./config.ini')
        agent_args = Dict(config, 'sac')  # 使用默认的'sac'配置
        
        # 设置设备
        device = 'cpu'
        # 创建环境
        env = gym.make(env_name)
        env.reset()[0]
        
        # 设置质量和摩擦力
        if mass is not None:
            cur_mass = env.model.body_mass
            new_mass = mass * cur_mass
            env.model.body_mass[:] = new_mass
        
        if friction is not None:
            cur_friction = env.model.geom_friction
            new_friction = friction * cur_friction
            env.model.geom_friction[:] = new_friction
        
        # 创建智能体
        agent = SAC(None, device, state_dim, action_dim, agent_args)
        agent.load_state_dict(model_state_dict)
        agent.eval()
        
        # 运行10次评估并计算平均奖励
        total_reward = 0
        n_eval_episodes = 10
        
        for _ in range(n_eval_episodes):
            state = env.reset()[0]
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                with torch.no_grad():
                    action, _ = agent.get_action(
                        torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=device), 0))
                action = action.cpu().detach().numpy()
                state_, r, done, truncated, *_ = env.step(action.reshape(-1))
                episode_reward += r
                state = state_
            
            total_reward += episode_reward
        
        # 计算平均奖励
        mean_reward = total_reward / n_eval_episodes
        
        # 清理资源
        env.close()
        del agent
        return mass, friction, mean_reward
    except Exception as e:
        print(f"Worker错误: {e}")
        import traceback
        traceback.print_exc()
        return mass, friction, 0.0  # 出错时返回零奖励

def test_policy(env_name, model_state_dict, state_dim, action_dim, agent_args, 
                DISTURBANCE_DICT, writer, seed=42, logger=None, exp_name='mutil_test'):
    """
    使用多进程测试不同的质量和摩擦力组合
    """
    mass_list = DISTURBANCE_DICT[env_name]['mass_list']
    friction_list = DISTURBANCE_DICT[env_name]['friction_list']
    reward_lists = []
    
    # 准备任务列表
    tasks = []
    for mass in mass_list:
        for friction in friction_list:
            tasks.append((env_name, model_state_dict, mass, friction, state_dim, action_dim))
    
    # 创建进程池并行处理所有任务
    from multiprocessing import Pool
    
    print(f"开始并行测试 ({len(mass_list)}x{len(friction_list)} 网格点)，使用最多30个进程...")
    
    # 使用进程池
    total_tasks = len(tasks)
    with Pool(processes=15) as pool:  # 使用30个进程
        # 并行执行评估
        results = []
        for idx, result in enumerate(pool.imap_unordered(worker, tasks)):
            mass, friction, reward = result
            
            # 找到对应的索引
            i = np.where(mass_list == mass)[0][0]
            j = np.where(friction_list == friction)[0][0]
            
            # 存储结果
            mass_rounded = round(mass, 2)
            friction_rounded = round(friction, 2)
            print(f"已评估点 {idx+1}/{total_tasks}: Mass: {mass_rounded}, Friction: {friction_rounded}, Reward: {int(reward)}")
            logger.info(f"Mass: {mass_rounded}, Friction: {friction_rounded}, Reward: {int(reward)}")
            
            # 将结果添加到列表
            results.append((i, j, int(reward)))
    
    # 构建完整的reward_lists
    reward_matrix = np.zeros((len(mass_list), len(friction_list)))
    for i, j, reward in results:
        reward_matrix[i, j] = reward
    
    # 按质量分割成reward_lists
    for i in range(len(mass_list)):
        reward_lists.append(reward_matrix[i].tolist())
    
    # 保存结果
    os.makedirs(f'./test_result/{exp_name}_cross/{env_name}/result_numpy', exist_ok=True)
    np.save(f'./test_result/{exp_name}_cross/{env_name}/result_numpy/{env_name}_seed{seed}.npy',
            np.array(reward_lists))
    
    print("并行测试完成，结果已保存")

def get_model_path(env_name,seed,exp_name):
    # 在best_model下找对应的文件夹
    model_dir = f'./best_model/{exp_name}/{env_name}'
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        return None
    
    # 查找匹配的文件夹
    target_folder_name = f"seed{seed}"
    
    for folder_name in os.listdir(model_dir):
        
        folder_path = os.path.join(model_dir, folder_name)
        match = re.search(r'seed(\d+)', folder_name)
        if os.path.isdir(folder_path) and match and match.group(1) == str(seed):
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
    # 设置多进程启动方法
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    prefix_dir=f'./test_result/{args.exp_name}_cross/{args.env_name}'
    # 创建测试数据保存目录
    os.makedirs(prefix_dir, exist_ok=True)

    model_path = get_model_path(args.env_name,args.seed,args.exp_name)
    if model_path is None:
        print(f"未找到模型文件，请检查模型名称是否正确")
        return
    
    # 预先加载模型参数到内存
    device = 'cpu'
    model_state_dict = torch.load(model_path, map_location=torch.device(device))
    
    DISTURBANCE_DICT = {
        "Hopper-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "Walker2d-v2": {"mass_list": np.linspace(0.5,1.5,51), "friction_list": np.linspace(0.5,1.5,51)},
        "HalfCheetah-v2": {"mass_list": np.linspace(0.1, 1.9, 51), "friction_list": np.linspace(0.1, 1.9, 51)},
        "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.5, 1.5, 51),
                                      "friction_list": np.linspace(0.5, 1.5, 51)}}
    
    # 设置随机种子
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.empty_cache()  # 清理CUDA缓存

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
    logger.info(f"使用模型: {model_path}")
    writer = SummaryWriter(log_dir=log_dir)
    
    # 获取环境维度
    env = gym.make(args.env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    env.close()  # 关闭这个环境，因为我们将在子进程中创建环境

    # 直接调用test_policy进行并行测试
    test_policy(args.env_name, model_state_dict, state_dim, action_dim, agent_args, 
                DISTURBANCE_DICT, writer, seed=args.seed, logger=logger, exp_name=args.exp_name)
    logger.info(f"并行测试完成，结果已保存")


if __name__ == '__main__':
    parser = ArgumentParser('parameters')

    parser.add_argument("--env_name", type=str, default='Walker2d-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--exp_name", type=str, default='Adam-optimizer', help='实验名称，用于保存结果')
    parser.add_argument("--algo", type=str, default='sac', help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--epochs', type=int, default=2501, help='number of epochs, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='auto', help='load network name in ./model_weights or "auto" to find automatically')
    parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=20, help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.2, help='reward scaling(default : 0.1)')
    parser.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")

    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('./config.ini')
    print(parser)
    agent_args = Dict(parser, args.algo)
    main_test(args, agent_args)