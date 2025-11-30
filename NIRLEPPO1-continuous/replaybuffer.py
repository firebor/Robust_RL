import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, args):
        self.states = np.zeros((args.batch_size, args.state_dim))
        self.actions = np.zeros((args.batch_size, args.action_dim))
        self.action_logprobs = np.zeros((args.batch_size, args.action_dim))
        self.rewards = np.zeros((args.batch_size, 1))
        self.next_states = np.zeros((args.batch_size, args.state_dim))
        self.dead_or_win = np.zeros((args.batch_size, 1))
        self.dones = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, state, action, action_logprob, reward, next_state, dead_or_win, done):
        self.states[self.count] = state
        self.actions[self.count] = action
        self.action_logprobs[self.count] = action_logprob
        self.rewards[self.count] = reward
        self.next_states[self.count] = next_state
        self.dead_or_win[self.count] = dead_or_win
        self.dones[self.count] = done
        self.count += 1

    def numpy_to_tensor(self):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        action_logprobs = torch.tensor(self.action_logprobs, dtype=torch.float)
        rewards = torch.tensor(self.rewards, dtype=torch.float)
        next_states = torch.tensor(self.next_states, dtype=torch.float)
        dead_or_win = torch.tensor(self.dead_or_win, dtype=torch.float)
        dones = torch.tensor(self.dones, dtype=torch.float)

        return states, actions, action_logprobs, rewards, next_states, dead_or_win, dones
