from configparser import ConfigParser
from argparse import ArgumentParser
from log import setup_logger

import torch
import gym
import numpy as np
import os
from Pd_sac import PDSAC
from utils.utils import make_transition, Dict
from torch.utils.tensorboard import SummaryWriter

# os.environ['LD_LIBRARY_PATH'] = "/home/cmdout/.mujoco/mujoco210/bin:/usr/lib/nvidia"
os.environ
# os.makedirs('./model_weights', exist_ok=True)

opio = [1774, 3118, 3609, 4888, 5480, 5463, 5302, 5115, 4968, 4591, 4118]


def main(args, agent_args):
    # print(f"--{args.noise_multiplier}-----------{args.seed}-----------66-")
    # 随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not args.use_cuda:
        device = 'cpu'

    os.makedirs(f'./train_result/{args.exp_name}/{args.env_name}', exist_ok=True)
    log_dir_temp=f'./train_result/{args.exp_name}/{args.env_name}/tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}'
    log_dir=f'./train_result/{args.exp_name}/{args.env_name}/tensorboard_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}'
    counter=0
    while os.path.exists(log_dir):
        counter+=1
        log_dir=f"{log_dir_temp}({counter})"
        
    
    os.makedirs(name=log_dir,exist_ok=True)
    os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs',exist_ok=True)
    log_log_dir=""
    model_dir=""
    if counter==0:
        log_log_dir=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}.log'
        os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}',exist_ok=True)
        model_dir=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}'
    else:
        log_log_dir=f'./train_result/{args.exp_name}/{args.env_name}/logs_logs/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}({counter}).log'
        os.makedirs(name=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}({counter})',exist_ok=True)
        model_dir=f'./train_result/{args.exp_name}/{args.env_name}/model_weights/{args.env_name}_{args.algo}_seed{args.seed}_noise{args.noise_multiplier}({counter})'
    # 新建日志
    logger = setup_logger(log_dir=log_log_dir)
    
    
    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env_name)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    env_evaluate = gym.make(args.env_name)
    env_evaluate.reset(seed=args.seed)
    env_evaluate.action_space.seed(args.seed)

    action_dim = env.action_space.shape[0] # action=6
    state_dim = env.observation_space.shape[0] #state_dim=17
    agent = PDSAC(writer, device, state_dim, action_dim, args.noise_multiplier, agent_args)
    if args.load != 'no':
        agent.load_state_dict(torch.load("./model_weights/" + args.load))
    steps = 0
    evaluate_reward=0

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
            next_state, reward, terminated, truncated, info = env.step(action.reshape(-1))
            done = terminated or truncated
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
                    evaluate_reward = evaluate_policy(env_evaluate, agent)
                logger.info("avg score : {:.1f}, of steps :{}".format(evaluate_reward, steps))
                torch.save(agent.state_dict(), model_dir+'/agent_' + str(steps) + "_" + str(evaluate_reward))
        if args.tensorboard and steps > 0:
            writer.add_scalar("score/score", score, steps)
        

            
def is_evl_reward(steps, args):
    eval_interval=args.eval_interval
    if steps > 1000000:
        eval_interval=eval_interval/10
    elif steps > 500000:
        eval_interval=eval_interval/5
    if steps == 1000:
        print(f"steps: {steps}, eval_interval: {eval_interval}")
    if steps % eval_interval == 0 and steps != 0:
        return True
    # if score>5200:
    #     return True
    return False

# def is_save_model(n_epi,score,args):
#     if score>5200:
#         return True
#     if n_epi % args.save_interval == 0 and n_epi != 0 and n_epi > 500:
#         return True
#     elif n_epi > 1200 and n_epi%10==0:
#         return True
#     elif n_epi > 2000 and n_epi%5==0:
#         return True
#     return False

def evaluate_policy(env, agent):
    # times = 100  # Perform three evaluations and calculate the average
    times=10
    evaluate_reward = 0
    for _ in range(times):
        # state = env.reset()
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
    np.save('./data_evaluate/PDSAC_{}_{}.npy'.format(env_name, 'friction_agent_2500'),
            np.array(reward_list))


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
    agent = PDSAC(writer, device, state_dim, action_dim, agent_args)
    agent.load_state_dict(torch.load("./model_weights/noise0.2/" + args.load, map_location=torch.device('cpu')))
    test_policy(args.env_name, agent, DISTURBANCE_DICT, writer)


if __name__ == '__main__':
    parser = ArgumentParser('parameters')
    parser.add_argument("--env_name", type=str, default='Walker2d-v2', help="'Ant-v2','HalfCheetah-v2','Hopper-v2','Humanoid-v2','HumanoidStandup-v2',\
          'InvertedDoublePendulum-v2', 'InvertedPendulum-v2' (default : Hopper-v2)")
    parser.add_argument("--exp_name", type=str, default='test', help='实验名称，用于保存结果')
    parser.add_argument("--algo", type=str, default='sac', help='algorithm to adjust (default : ppo)')
    parser.add_argument('--train', type=bool, default=True, help="(default: True)")
    parser.add_argument('--render', type=bool, default=False, help="(default: False)")
    parser.add_argument('--steps', type=int, default=1500000, help='number of steps, (default: 1000)')
    parser.add_argument('--tensorboard', type=bool, default=True, help='use_tensorboard, (default: False)')
    parser.add_argument("--load", type=str, default='no', help='load network name in ./model_weights')
    parser.add_argument("--save_interval", type=int, default=10000, help='save interval(default: 100000)')
    parser.add_argument("--eval_interval", type=int, default=50000, help='print interval(default : 100000)')
    parser.add_argument("--use_cuda", type=bool, default=True, help='cuda usage(default : True)')
    parser.add_argument("--reward_scaling", type=float, default=0.2, help='reward scaling(default : 0.1)')
    parser.add_argument("--noise_multiplier", type=float, default=0.2, help="设置高斯噪声的标准差(默认0.2)")
    parser.add_argument("--seed", type=int, default=1, help="随机种子(默认42)")
    args = parser.parse_args()

    parser = ConfigParser()
    parser.read('./config.ini')
    # print(parser.sections())
    agent_args = Dict(parser, args.algo)
    main(args, agent_args)
