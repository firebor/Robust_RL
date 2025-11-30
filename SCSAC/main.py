from configparser import ConfigParser
from argparse import ArgumentParser
from log import setup_logger

import torch
import gym
import numpy as np
import os
from sc_sac import SCSAC
from utils.utils import make_transition, Dict, RunningMeanStd
from torch.utils.tensorboard import SummaryWriter
import random

# os.environ['LD_LIBRARY_PATH'] = "/home/cmdout/.mujoco/mujoco210/bin:/usr/lib/nvidia"
# os.makedirs('./model_weights', exist_ok=True)


def main(args, agent_args):
    # 设置所有随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.use_cuda:
        device = 'cpu'

    os.makedirs(f'./train_result/{args.exp_name}/{args.env_name}', exist_ok=True)

    log_dir_temp=f'./train_result/{args.exp_name}/{args.env_name}/tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}'
    log_dir=f'./train_result/{args.exp_name}/{args.env_name}/tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}'
    counter=0
    while os.path.exists(log_dir):
        counter+=1
        log_dir=f"{log_dir_temp}({counter})"
        
    
    os.makedirs(name=log_dir,exist_ok=True)
    os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs',exist_ok=True)
    log_log_dir=""
    model_dir=""
    if counter==0:
        log_log_dir=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_{args.algo}_seed{args.seed}.log'
        os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}',exist_ok=True)
        model_dir=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}'
    else:
        log_log_dir=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_{args.algo}_seed{args.seed}({counter}).log'
        os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}({counter})',exist_ok=True)
        model_dir=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}({counter})'
    # 新建日志
    logger = setup_logger(log_dir=log_log_dir)


    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    env_evaluate = gym.make(args.env_name)
    env_evaluate.reset(seed=args.seed)
    env_evaluate.action_space.seed(args.seed)

    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = SCSAC(writer, device, state_dim, action_dim, agent_args)
    if args.load != 'no':
        agent.load_state_dict(torch.load("./model_weights/" + args.load))
    
    steps = 0
    evaluate_reward = 0

    while steps < args.steps:
        score = 0.0
        state = env.reset()[0]
        done = False
        truncated = False
        while not done and not truncated:
            if args.render:
                env.render()
            if steps < 5000:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = agent.get_action(torch.unsqueeze(torch.tensor(state, dtype=torch.float, device=device), 0))
                action = action.cpu().numpy()
            
            steps = steps + 1
            next_state, reward, done, truncated, *_ = env.step(action.reshape(-1))
            transition = make_transition(state,
                                         action,
                                         np.array([reward * args.reward_scaling]),
                                         next_state,
                                         np.array([done])
                                         )
            agent.put_data(transition)
            state = next_state
            score += reward
            if agent.data.data_idx > agent_args.learn_start_size:
                agent.train_net(agent_args.batch_size, steps)
            
            if is_evl_reward(steps, args):
                with torch.no_grad():
                    evaluate_reward = evaluate_policy(env_evaluate, agent, args.seed)
                logger.info("avg score : {:.1f}, of steps :{}".format(evaluate_reward, steps))
                torch.save(agent.state_dict(), model_dir+'/agent_' + str(steps) + "_" + str(evaluate_reward))
        
        if args.tensorboard and steps > 0:
            writer.add_scalar("score/score", score, steps)

def is_evl_reward(steps, args):
    eval_interval = args.eval_interval
    if steps > 1000000:
        eval_interval = eval_interval/10
    elif steps > 500000:
        eval_interval = eval_interval/5
    
    if steps % eval_interval == 0 and steps != 0:
        return True
    return False

def is_save_model(n_epi,score,args):
    if score>5200:
        return True
    if n_epi % args.save_interval == 0 and n_epi != 0 and n_epi > 500:
        return True
    elif n_epi > 1200 and n_epi%10==0:
        return True
    elif n_epi > 2000 and n_epi%5==0:
        return True
    return False

def evaluate_policy(env, agent,eval_seed=None):
    times = 10  # Perform three evaluations and calculate the average
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


def test_policy(env_name, agent, DISTURBANCE_DICT, writer):
    mass_list = DISTURBANCE_DICT[env_name]['mass_list']
    friction_list = DISTURBANCE_DICT[env_name]['friction_list']
    reward_list = []
    # for mass in mass_list:
    #     env = gym.make(env_name)
    #     cur_mass = env.model.body_mass
    #     new_mass = mass * cur_mass
    #     env.model.body_mass[:] = new_mass
    #     reward = evaluate_policy(env, agent)
    #     writer.add_scalar("reward/reward", reward, mass)
    #     reward_list.append(reward)
    #     print(reward)
    for friction in friction_list:
        env = gym.make(env_name)
        cur_friction = env.model.geom_friction
        new_friction = friction * cur_friction
        env.model.geom_friction[:] = new_friction
        reward = evaluate_policy(env, agent)
        writer.add_scalar("reward/reward", reward, friction)
        reward_list.append(reward)
        print(reward)
    np.save('./data_evaluate/SCSAC_{}_{}.npy'.format(env_name, 'friction_agent_2000'),
            np.array(reward_list))
    print(reward_list)


def main_test(args, agent_args):
    DISTURBANCE_DICT = {
        "Hopper-v2": {"mass_list": np.linspace(0.2, 1.6, 11), "friction_list": np.linspace(0.7, 1.3, 11)},
        "Walker2d-v2": {"mass_list": np.linspace(0.5, 1.5, 11), "friction_list": np.linspace(0.4, 1.8, 11)},
        "HalfCheetah-v2": {"mass_list": np.linspace(0.6, 1.3, 11), "friction_list": np.linspace(0.1, 2.7, 11)},
        "InvertedDoublePendulum-v2": {"mass_list": np.linspace(0.6, 2.6, 11),
                                      "friction_list": np.linspace(0.4, 2.4, 11)}}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter()
    env = gym.make(args.env_name)
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = SCSAC(writer, device, state_dim, action_dim, agent_args)
    agent.load_state_dict(torch.load("./model_weights/" + args.load, map_location=torch.device('cpu')))
    test_policy(args.env_name, agent, DISTURBANCE_DICT, writer)


if __name__ == '__main__':
    parser = ArgumentParser('parameters')

    parser.add_argument("--env_name", type=str, default='Hopper-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--exp_name", type=str, default='default_exp', help='实验名称，用于保存结果')
    parser.add_argument("--algo", type=str, default='sac', help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    # parser.add_argument("--load", type=str, default='agent_2000', help='load network name in ./model_weights')
    parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=100, help='save interval(default: 100)')
    parser.add_argument("--print_interval", type=int, default=20, help='print interval(default : 20)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.2, help='reward scaling(default : 0.1)')
    parser.add_argument("--seed", type=int, default=42, help="随机种子(默认42)")
    parser.add_argument("--steps", type=int, default=1500000, help='训练的总步数，替代epochs参数')
    parser.add_argument("--eval_interval", type=int, default=50000, help='评估和保存模型的间隔步数')

    args = parser.parse_args()
    parser = ConfigParser()
    parser.read('./config.ini')
    print(parser)
    agent_args = Dict(parser, args.algo)
    # main_test(args, agent_args)
    main(args, agent_args)
