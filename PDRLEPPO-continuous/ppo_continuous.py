import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal
import numpy as np
import os

from opacus import PrivacyEngine
from rle_network import RLEManager


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    def __init__(self, args):
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # The mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    def __init__(self, args):
        super(Actor_Gaussian, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))  # We use 'nn.Parameter' to train log_std automatically
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = self.max_action * torch.tanh(self.mean_layer(s))  # [-1,1]->[-max_action,max_action]
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous(nn.Module):
    def __init__(self, args):
        super(PPO_continuous, self).__init__()
        self.policy_dist = args.policy_dist
        self.max_action = args.max_action
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_state_norm = args.use_state_norm
        self.use_reward_norm = args.use_reward_norm
        self.use_reward_scaling = args.use_reward_scaling
        self.use_orthogonal_init = args.use_orthogonal_init
        self.use_tanh = args.use_tanh
        self.use_rle = getattr(args, 'use_rle', False)

        self.device = args.device

        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
        self.critic = Critic(args)
        
        # RLE支持：双价值网络
        if self.use_rle:
            self.intrinsic_critic = Critic(args)
            self.rle_manager = RLEManager(args)
        
        # 将模型移动到指定设备
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        if self.use_rle:
            self.intrinsic_critic = self.intrinsic_critic.to(self.device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
            if self.use_rle:
                self.optimizer_intrinsic_critic = torch.optim.Adam(self.intrinsic_critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
            if self.use_rle:
                self.optimizer_intrinsic_critic = torch.optim.Adam(self.intrinsic_critic.parameters(), lr=self.lr_c)
        
        self.privacy_engine_actor = PrivacyEngine(
            self.actor,
            sample_rate=1.0,
            # alphas=[10, 100],
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=1,
        )

    def evaluate(self, state):  # When evaluating the policy, we only use the mean
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
        if self.policy_dist == "Beta":
            action = self.actor.mean(state).detach().cpu().numpy().flatten()
        else:
            action = self.actor(state).detach().cpu().numpy().flatten()
        return action

    def choose_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(self.device)
        if self.policy_dist == "Beta":
            with torch.no_grad():
                dist = self.actor.get_dist(state)
                action = dist.sample()  # Sample the action according to the probability distribution
                action_logprob = dist.log_prob(action)  # The log probability density of the action
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(state)
                action = dist.sample()  # Sample the action according to the probability distribution
                action = torch.clamp(action, -self.max_action, self.max_action)  # [-max,max]
                action_logprob = dist.log_prob(action)  # The log probability density of the action
        return action.cpu().numpy().flatten(), action_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        if self.use_rle:
            states, actions, action_logprobs, rewards, next_states, dead_or_win, dones, intrinsic_rewards, goals, worker_ids, rle_features = replay_buffer.numpy_to_tensor()
        else:
            states, actions, action_logprobs, rewards, next_states, dead_or_win, dones = replay_buffer.numpy_to_tensor()
        
        # 将数据移动到GPU
        states = states.to(self.device)
        actions = actions.to(self.device)
        action_logprobs = action_logprobs.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dead_or_win = dead_or_win.to(self.device)
        dones = dones.to(self.device)
        
        if self.use_rle:
            intrinsic_rewards = intrinsic_rewards.to(self.device)
            goals = goals.to(self.device)
            worker_ids = worker_ids.to(self.device)
            rle_features = rle_features.to(self.device)
            
            # 更新RLE特征归一化统计
            self.rle_manager.update_normalization(rle_features)
        
        """
            Calculate the advantage using GAE
            'dead_or_win=True' means the episode terminated due to reaching a terminal state (death or win)
            'dones=True' represents the terminal of an episode (either terminated or truncated). When calculating the adv, if dones=True, gae=0
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(states)
            vs_ = self.critic(next_states)
            deltas = rewards + self.gamma * (1.0 - dead_or_win) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(self.device)
            v_target = adv + vs
            
            # RLE: 计算内在奖励的优势函数
            if self.use_rle:
                int_adv = []
                int_gae = 0
                int_vs = self.intrinsic_critic(states)
                int_vs_ = self.intrinsic_critic(next_states)
                int_deltas = intrinsic_rewards + self.gamma * (1.0 - dead_or_win) * int_vs_ - int_vs
                for delta, d in zip(reversed(int_deltas.flatten().cpu().numpy()), reversed(dones.flatten().cpu().numpy())):
                    int_gae = delta + self.gamma * self.lamda * int_gae * (1.0 - d)
                    int_adv.insert(0, int_gae)
                int_adv = torch.tensor(int_adv, dtype=torch.float).view(-1, 1).to(self.device)
                int_v_target = int_adv + int_vs
                
                # 组合优势函数
                combined_adv = adv + getattr(self.rle_manager, 'int_coef', 0.01) * int_adv
            else:
                combined_adv = adv
            
            if self.use_adv_norm:  # Trick 1:advantage normalization
                combined_adv = ((combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(states[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                action_logprob_now = dist_now.log_prob(actions[index])
                # a/b=exp(log(a)-log(b))  In multi-dimensional continuous action space，we need to sum up the log_prob
                ratios = torch.exp(action_logprob_now.sum(1, keepdim=True) - action_logprobs[index].sum(1, keepdim=True))  # shape(mini_batch_size X 1)

                surr1 = ratios * combined_adv[index]  # Only calculate the gradient of 'action_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * combined_adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # Trick 5: policy entropy
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(states[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()
                
                # RLE: 更新内在价值网络
                if self.use_rle:
                    int_v_s = self.intrinsic_critic(states[index])
                    int_critic_loss = F.mse_loss(int_v_target[index], int_v_s)
                    self.optimizer_intrinsic_critic.zero_grad()
                    int_critic_loss.backward()
                    if self.use_grad_clip:
                        torch.nn.utils.clip_grad_norm_(self.intrinsic_critic.parameters(), 0.5)
                    self.optimizer_intrinsic_critic.step()
            
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # RLE: 软更新特征网络
        if self.use_rle:
            self.rle_manager.update_feature_network(self.actor)

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
        if self.use_rle:
            for p in self.optimizer_intrinsic_critic.param_groups:
                p['lr'] = lr_c_now
