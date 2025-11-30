import numpy as np
import torch


class Dict(dict):
    def __init__(self, config, section_name, location=False):
        super(Dict, self).__init__()
        self.initialize(config, section_name, location)

    def initialize(self, config, section_name, location):
        for key, value in config.items(section_name):
            if location:
                self[key] = value
            else:
                self[key] = eval(value)

    def __getattr__(self, val):
        return self[val]


def make_transition(state, action, reward, next_state, done, log_prob=None):
    transition = {}
    transition['state'] = state
    transition['action'] = action
    transition['reward'] = reward
    transition['next_state'] = next_state
    transition['log_prob'] = log_prob
    transition['done'] = done
    return transition


def make_mini_batch(*value):
    mini_batch_size = value[0]
    full_batch_size = len(value[1])
    full_indices = np.arange(full_batch_size)
    np.random.shuffle(full_indices)
    for i in range(full_batch_size // mini_batch_size):
        indices = full_indices[mini_batch_size * i: mini_batch_size * (i + 1)]
        yield [x[indices] for x in value[1:]]


def convert_to_tensor(*value):
    # device = value[0]
    return [x for x in value[1:]]

    # return [torch.tensor(x).float().to(device) for x in value[1:]]


class ReplayBuffer():
    def __init__(self, action_prob_exist, max_size, state_dim, num_action, device):
        self.max_size = max_size
        self.data_idx = 0
        self.action_prob_exist = action_prob_exist
        self.data = {}
        self.device = device
        
        # 直接在 GPU 上初始化数据
        self.data['state'] = torch.zeros((self.max_size, state_dim), device=device)
        self.data['action'] = torch.zeros((self.max_size, num_action), device=device)
        self.data['reward'] = torch.zeros((self.max_size, 1), device=device)
        self.data['next_state'] = torch.zeros((self.max_size, state_dim), device=device)
        self.data['done'] = torch.zeros((self.max_size, 1), device=device)
        if self.action_prob_exist:
            self.data['log_prob'] = torch.zeros((self.max_size, 1), device=device)

    def put_data(self, transition):
        idx = self.data_idx % self.max_size
        # 直接将数据放入 GPU
        self.data['state'][idx] = torch.tensor(transition['state'], device=self.device)
        self.data['action'][idx] = torch.tensor(transition['action'], device=self.device)
        self.data['reward'][idx] = torch.tensor(transition['reward'], device=self.device)
        self.data['next_state'][idx] = torch.tensor(transition['next_state'], device=self.device)
        self.data['done'][idx] = torch.tensor(transition['done'], device=self.device)
        if self.action_prob_exist:
            self.data['log_prob'][idx] = torch.tensor(transition['log_prob'], device=self.device)
        
        self.data_idx += 1

    def sample(self, shuffle, batch_size=None):
        if shuffle:
            sample_num = min(self.max_size, self.data_idx)
            rand_idx = torch.randint(0, sample_num, (batch_size,), device=self.device)
            sampled_data = {}
            sampled_data['state'] = self.data['state'][rand_idx]
            sampled_data['action'] = self.data['action'][rand_idx]
            sampled_data['reward'] = self.data['reward'][rand_idx]
            sampled_data['next_state'] = self.data['next_state'][rand_idx]
            sampled_data['done'] = self.data['done'][rand_idx]
            if self.action_prob_exist:
                sampled_data['log_prob'] = self.data['log_prob'][rand_idx]
            return sampled_data
        else:
            return self.data

    def size(self):
        return min(self.max_size, self.data_idx)


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
