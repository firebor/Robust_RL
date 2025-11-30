from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
# from opacus import PrivacyEngine


class NISAC(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, noise_multiplier, args):
        super(NISAC,self).__init__()

        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim,
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std).to(device)

        self.q_1 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
                          self.args.activation_function, self.args.last_activation).to(device)
        self.q_2 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
                          self.args.activation_function, self.args.last_activation).to(device)

        self.target_q_1 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
                                 self.args.activation_function, self.args.last_activation).to(device)
        self.target_q_2 = Critic(self.args.layer_num, state_dim + action_dim, 1, self.args.hidden_dim,
                                 self.args.activation_function, self.args.last_activation).to(device)

        self.soft_update(self.q_1, self.target_q_1, 1.)
        self.soft_update(self.q_2, self.target_q_2, 1.)

        self.alpha = nn.Parameter(torch.tensor(self.args.alpha_init))

        self.data = ReplayBuffer(action_prob_exist=False, max_size=int(self.args.memory_size), state_dim=state_dim,
                                 num_action=action_dim, device=device)
        self.target_entropy = - torch.tensor(action_dim)

        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=self.args.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.args.q_lr)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.args.alpha_lr)
        self.n_epi = 0

        self.noise_multiplier = noise_multiplier

        # Initialize learnable noise amplitudes as parameters
        self.log_actor_noise_amplitude = nn.Parameter(torch.log(torch.tensor(noise_multiplier)))
        self.log_critic_noise_amplitude = nn.Parameter(torch.log(torch.tensor(noise_multiplier)))

        # Target values for noise amplitudes (can be adjusted based on experiments)
        self.target_actor_noise = torch.tensor(1)  # Small target value to encourage noise reduction over time
        self.target_critic_noise = torch.tensor(1)

        # Create optimizers for noise amplitudes
        self.actor_noise_optimizer = optim.Adam([self.log_actor_noise_amplitude], lr=self.args.alpha_lr)
        self.critic_noise_optimizer = optim.Adam([self.log_critic_noise_amplitude], lr=self.args.alpha_lr)

        self.device = device
        self.writer = writer

    def put_data(self, transition):
        self.data.put_data(transition)

    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def get_action(self, state):
        mu, std = self.actor(state)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) + 1e-3)
        return a, a_log_prob.sum(-1, keepdim=True)

    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, dones):
        ###target
        with torch.no_grad():
            next_actions, next_action_log_prob = self.get_action(next_states)
            q_1 = self.target_q_1(next_states, next_actions)
            q_2 = self.target_q_2(next_states, next_actions)
            q = torch.min(q_1, q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)
            targets = rewards + self.args.gamma * v

        q = Q(states, actions)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()

        # Calculate the total gradient norm
        norms = [
            torch.linalg.vector_norm(parameters.grad)
            for parameters in Q.parameters()
            if parameters.grad is not None
        ]

        total_norm = torch.linalg.vector_norm(torch.stack(norms))

        # Add Gaussian noise to gradients manually using learnable amplitude
        # Get the actual amplitude by exponentiating the log parameter
        critic_noise_amplitude = torch.exp(self.log_critic_noise_amplitude)
        for param in Q.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad, device=self.device) * critic_noise_amplitude * total_norm
                param.grad = param.grad + noise

        grad_norm = torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=0.99)

        q_optimizer.step()
        return loss, grad_norm

    def actor_update(self, states):
        now_actions, now_action_log_prob = self.get_action(states)
        q_1 = self.q_1(states, now_actions)
        q_2 = self.q_2(states, now_actions)
        q = torch.min(q_1, q_2)

        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()

        norms = [
            torch.linalg.vector_norm(parameters.grad)
            for parameters in self.actor.parameters()
            if parameters.grad is not None
            ]

        total_norm = torch.linalg.vector_norm(torch.stack(norms))

        # Add Gaussian noise to gradients manually using learnable amplitude
        # Get the actual amplitude by exponentiating the log parameter
        actor_noise_amplitude = torch.exp(self.log_actor_noise_amplitude)
        for param in self.actor.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad, device=self.device) * actor_noise_amplitude * total_norm
                param.grad = param.grad + noise

        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.99)
        self.actor_optimizer.step()
        return loss, now_action_log_prob, grad_norm

    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return loss

    def actor_noise_update(self, actor_grad_norm):
        # Calculate the difference between current gradient norm and target
        # We want to reduce noise when gradients are large and increase when small
        norm_diff = actor_grad_norm - self.target_actor_noise

        # Loss is designed to adjust noise amplitude based on gradient norm
        # When norm_diff > 0 (large gradients), we want to reduce noise
        # When norm_diff < 0 (small gradients), we want to increase noise
        loss = (torch.exp(self.log_actor_noise_amplitude) * norm_diff.detach()).mean()

        self.actor_noise_optimizer.zero_grad()
        loss.backward()
        self.actor_noise_optimizer.step()
        return loss

    def critic_noise_update(self, critic_grad_norm):
        # Similar approach as actor noise update
        norm_diff = critic_grad_norm - self.target_critic_noise
        loss = (torch.exp(self.log_critic_noise_amplitude) * norm_diff.detach()).mean()

        self.critic_noise_optimizer.zero_grad()
        loss.backward()
        self.critic_noise_optimizer.step()
        return loss

    def train_net(self, batch_size, steps):
        data = self.data.sample(shuffle=True, batch_size=batch_size)
        states, actions, rewards, next_states, dones = data['state'], data['action'],data['reward'], data['next_state'],data['done']

        # Noise amplitudes are now learnable parameters, no need to calculate them here

        ###q update
        q_1_loss, q1_grad_norm = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, dones)
        q_2_loss, q2_grad_norm = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, dones)

        # Use the gradient norms from the update methods
        critic_grad_norm = (q1_grad_norm + q2_grad_norm) / 2  # Average of both critics

        ### actor update
        actor_loss, prob, actor_grad_norm = self.actor_update(states)

        ###alpha update
        alpha_loss = self.alpha_update(prob)

        ###noise amplitude updates
        actor_noise_loss = self.actor_noise_update(actor_grad_norm)
        critic_noise_loss = self.critic_noise_update(critic_grad_norm)

        self.soft_update(self.q_1, self.target_q_1, self.args.soft_update_rate)
        self.soft_update(self.q_2, self.target_q_2, self.args.soft_update_rate)

        # cumulative_reward=rewards.sum().item()

        # Increment episode counter
        self.n_epi += 1

        if self.writer != None and steps % self.args.log_interval == 0:
            # self.writer.add_scalar("rewards/cumulative_reward",cumulative_reward,n_epi)
            self.writer.add_scalar("loss/q_1", q_1_loss, steps)
            self.writer.add_scalar("loss/q_2", q_2_loss, steps)
            self.writer.add_scalar("loss/actor", actor_loss, steps)
            self.writer.add_scalar("loss/alpha", alpha_loss, steps)
            self.writer.add_scalar("loss/actor_noise", actor_noise_loss, steps)
            self.writer.add_scalar("loss/critic_noise", critic_noise_loss, steps)

            # Log gradient norms
            self.writer.add_scalar("grad_norm/actor", actor_grad_norm, steps)
            self.writer.add_scalar("grad_norm/critic_1", q1_grad_norm, steps)
            self.writer.add_scalar("grad_norm/critic_2", q2_grad_norm, steps)

            # Log the actual noise amplitudes (exponentiated from log parameters)
            actor_noise_amplitude = torch.exp(self.log_actor_noise_amplitude).item()
            critic_noise_amplitude = torch.exp(self.log_critic_noise_amplitude).item()
            self.writer.add_scalar("noise/actor_amplitude", actor_noise_amplitude, steps)
            self.writer.add_scalar("noise/critic_amplitude", critic_noise_amplitude, steps)

        # 释放缓存
        # 释放中间变量
        del states, actions, rewards, next_states, dones, q_1_loss, q_2_loss, actor_loss, prob, alpha_loss
        del actor_noise_loss, critic_noise_loss, critic_grad_norm, q1_grad_norm, q2_grad_norm, actor_grad_norm
