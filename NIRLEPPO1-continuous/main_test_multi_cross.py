import torch
import gym
import numpy as np
import os
from ppo_continuous import PPO_continuous
from normalization import Normalization
from multiprocessing import Pool
import argparse
import logging
import re
import warnings
from datetime import datetime

# 全面屏蔽警告
warnings.filterwarnings('ignore')
# 设置日志级别
logging.getLogger().setLevel(logging.ERROR)
# 禁用gym的警告
gym.logger.set_level(logging.ERROR)


class Args:
    """参数类，模拟argparse.Namespace"""
    def __init__(self):
        # PPO算法参数
        self.policy_dist = "Gaussian"  # Beta or Gaussian
        self.max_action = 1.0
        self.batch_size = 2048
        self.mini_batch_size = 64
        self.max_train_steps = 3e6
        self.lr_a = 3e-4  # Learning rate of actor
        self.lr_c = 3e-4  # Learning rate of critic
        self.gamma = 0.99  # Discount factor
        self.lamda = 0.95  # GAE parameter
        self.epsilon = 0.2  # PPO clip parameter
        self.K_epochs = 10  # PPO parameter
        self.entropy_coef = 0.01  # Entropy coefficient
        self.set_adam_eps = True
        self.use_grad_clip = True
        self.use_lr_decay = True
        self.use_adv_norm = True
        self.use_state_norm = True
        self.use_reward_norm = False
        self.use_reward_scaling = False
        self.use_orthogonal_init = True
        self.use_tanh = True
        self.hidden_width = 64
        self.noise_multiplier = 1.0
        self.device = 'cpu'


def setup_logger(log_dir):
    """设置日志记录器"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def worker(args_tuple):
    """
    工作进程函数，用于并行评估
    每个worker运行10次评估并返回平均奖励
    """
    try:
        env_name, model_state_dict, norm_stats, mass, friction, state_dim, action_dim = args_tuple
        
        # 创建参数对象
        args = Args()
        args.state_dim = state_dim
        args.action_dim = action_dim
        args.device = 'cpu'
        args.use_state_norm = True
        # 创建环境
        env = gym.make(env_name)
        env.reset()
        
        # 设置环境参数
        args.max_action = float(env.action_space.high[0])
        
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
        agent = PPO_continuous(args)
        agent.load_state_dict(model_state_dict)
        agent.eval()
        
        # 创建状态归一化器
        state_norm = None
        if args.use_state_norm and norm_stats is not None:
            print(f"使用状态归一化器norm_stats")
            state_norm = Normalization(shape=state_dim)
            state_norm.running_ms.n = norm_stats['sample_count']
            state_norm.running_ms.mean = np.array(norm_stats['mean'])
            state_norm.running_ms.S = np.array(norm_stats['sum_squared_diff'])
            state_norm.running_ms.std = np.array(norm_stats['std'])
        
        # 运行10次评估并计算平均奖励
        total_reward = 0
        n_eval_episodes = 10
        
        for _ in range(n_eval_episodes):
            state = env.reset()[0]
            if state_norm is not None:
                state = state_norm(state, update=False)
            done = False
            truncated = False
            episode_reward = 0
            
            while not done and not truncated:
                with torch.no_grad():
                    action = agent.evaluate(state)
                    if args.policy_dist == "Beta":
                        scaled_action = 2 * (action - 0.5) * args.max_action  # [0,1]->[-max,max]
                    else:
                        scaled_action = action
                
                next_state, r, done, truncated, *_ = env.step(scaled_action)
                if state_norm is not None:
                    next_state = state_norm(next_state, update=False)
                episode_reward += r
                state = next_state
            
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


def test_policy(env_name, model_state_dict, norm_stats, state_dim, action_dim, 
                DISTURBANCE_DICT, seed=42, logger=None, exp_name='nippo_test', switch_steps=60):
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
            tasks.append((env_name, model_state_dict, norm_stats, mass, friction, state_dim, action_dim))
    
    print(f"开始并行测试 ({len(mass_list)}x{len(friction_list)} 网格点)，使用最多20个进程...")
    
    # 使用进程池
    total_tasks = len(tasks)
    with Pool(processes=20) as pool:
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
            if logger:
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
    np.save(f'./test_result/{exp_name}_cross/{env_name}/result_numpy/{env_name}_seed{seed}_sw{switch_steps}.npy',
            np.array(reward_lists))
    
    print("并行测试完成，结果已保存")


def get_model_path(env_name, seed, exp_name, switch_steps):
    """获取模型路径"""
    # 在best_model下找对应的文件夹
    model_dir = f'./best_model/{exp_name}/{env_name}'
    if not os.path.exists(model_dir):
        print(f"错误：目录 {model_dir} 不存在")
        print("请先运行 get_best_model.py 提取最佳模型")
        return None, None
    
    # 查找匹配的文件夹
    target_folder_name = f"seed{seed}_sw{switch_steps}"
    for folder_name in os.listdir(model_dir):
        folder_path = os.path.join(model_dir, folder_name)
        # 使用正则表达式匹配seed和switch_steps
        match = re.search(r'seed(\d+)_sw(\d+)', folder_name)
        if os.path.isdir(folder_path) and match and match.group(1) == str(seed) and match.group(2) == str(switch_steps):
            # 找到匹配的文件夹，返回其中的第一个文件
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            if files:
                model_path = os.path.join(folder_path, files[0])
                print(f"找到模型文件: {model_path}")
                
                # 查找对应的归一化统计文件（仍然从train_result中获取）
                norm_stats_dir = f'./train_result/{exp_name}/{env_name}/normalization_stats/{env_name}_PPO_seed{seed}_sw{switch_steps}'
                norm_stats_path = os.path.join(norm_stats_dir, 'normalization_stats.npy')
                
                norm_stats = None
                if os.path.exists(norm_stats_path):
                    norm_stats = np.load(norm_stats_path, allow_pickle=True).item()
                    print(f"找到归一化统计文件: {norm_stats_path}")
                else:
                    print(f"警告：归一化统计文件不存在: {norm_stats_path}")
                
                return model_path, norm_stats
            else:
                print(f"在文件夹 {folder_path} 中没有找到文件")
    
    print(f"在 {model_dir} 中没有找到匹配的文件夹 {target_folder_name}")
    return None, None


def main_test(test_args):
    """主测试函数"""
    # 设置多进程启动方法
    from multiprocessing import set_start_method
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    prefix_dir = f'./test_result/{test_args.exp_name}_cross/{test_args.env_name}'
    # 创建测试数据保存目录
    os.makedirs(prefix_dir, exist_ok=True)
    print(f"创建目录: {prefix_dir}")

    model_path, norm_stats = get_model_path(test_args.env_name, test_args.seed, test_args.exp_name, test_args.switch_steps)
    if model_path is None:
        print(f"未找到模型文件，请检查模型名称是否正确")
        return
    
    # 预先加载模型参数到内存
    device = 'cpu'
    model_state_dict = torch.load(model_path, map_location=torch.device(device))
    
    DISTURBANCE_DICT = {
        "Hopper-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "Walker2d-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "HalfCheetah-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "Ant-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)},
        "Humanoid-v2": {"mass_list": np.linspace(0.5, 1.5, 51), "friction_list": np.linspace(0.5, 1.5, 51)}
    }
    
    # 设置随机种子
    np.random.seed(test_args.seed)
    torch.manual_seed(test_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(test_args.seed)
        torch.cuda.empty_cache()

    # 创建日志目录
    log_dir_temp = f'{prefix_dir}/test_tensorboard_logs/{test_args.env_name}_PPO_seed{test_args.seed}_sw{test_args.switch_steps}'
    log_dir = log_dir_temp
    counter = 0
    while os.path.exists(log_dir):
        counter += 1
        log_dir = f"{log_dir_temp}({counter})"
    
    os.makedirs(name=log_dir, exist_ok=True)
    os.makedirs(name=f'{prefix_dir}/test_logs_logs', exist_ok=True)
    
    if counter == 0:
        log_log_dir = f'{prefix_dir}/test_logs_logs/{test_args.env_name}_PPO_seed{test_args.seed}_sw{test_args.switch_steps}.log'
    else:
        log_log_dir = f'{prefix_dir}/test_logs_logs/{test_args.env_name}_PPO_seed{test_args.seed}_sw{test_args.switch_steps}({counter}).log'
    
    logger = setup_logger(log_log_dir)
    logger.info(f"使用模型: {model_path}")
    
    # 获取环境维度
    env = gym.make(test_args.env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    env.close()

    # 直接调用test_policy进行并行测试
    test_policy(test_args.env_name, model_state_dict, norm_stats, state_dim, action_dim, 
                DISTURBANCE_DICT, seed=test_args.seed, logger=logger, exp_name=test_args.exp_name,
                switch_steps=test_args.switch_steps)
    logger.info(f"并行测试完成，结果已保存")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('NIPPO鲁棒性测试参数')

    parser.add_argument("--env_name", type=str, default='Walker2d-v2', 
                       help="环境名称: 'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2',\
                            'InvertedDoublePendulum-v2' (默认: Walker2d-v2)")
    parser.add_argument("--exp_name", type=str, default='RLE', 
                       help='实验名称，用于保存结果')
    parser.add_argument("--seed", type=int, default=42, 
                       help="随机种子(默认42)")
    parser.add_argument("--switch_steps", type=int, default=70, help="加噪声的步数")

    test_args = parser.parse_args()
    main_test(test_args) 