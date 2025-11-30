import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque


def orthogonal_init(layer, gain=1.0):
    """正交初始化"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class RunningMeanStd:
    """运行均值和标准差计算器"""
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


class RLEFeatureNetwork(nn.Module):
    """RLE特征网络，与策略网络共享相同的骨干架构"""
    
    def __init__(self, args):
        super(RLEFeatureNetwork, self).__init__()
        self.feature_size = args.rle_feature_size
        self.device = args.device
        
        # 与策略网络相同的骨干架构
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.feature_layer = nn.Linear(args.hidden_width, self.feature_size)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]
        
        if args.use_orthogonal_init:
            print("------RLE network use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.feature_layer, gain=0.01)
        
        # 特征归一化统计
        self.running_ms = RunningMeanStd(shape=(1, self.feature_size))
        self.feature_mean = torch.zeros(1, self.feature_size, device=self.device)
        self.feature_std = torch.ones(1, self.feature_size, device=self.device)
        
        self.to(self.device)
    
    def forward(self, state):
        """前向传播，输出原始特征"""
        x = self.activate_func(self.fc1(state))
        x = self.activate_func(self.fc2(x))
        features = self.feature_layer(x)
        return features
    
    def get_normalized_features(self, state):
        """获取归一化后的特征"""
        raw_features = self.forward(state)
        normalized_features = (raw_features - self.feature_mean) / (self.feature_std + 1e-5)
        return normalized_features, raw_features
    
    def update_normalization_stats(self, features_batch):
        """更新特征归一化统计"""
        if isinstance(features_batch, torch.Tensor):
            features_batch = features_batch.detach().cpu().numpy()
        
        self.running_ms.update(features_batch)
        self.feature_mean = torch.tensor(self.running_ms.mean, device=self.device, dtype=torch.float32)
        self.feature_std = torch.sqrt(torch.tensor(self.running_ms.var, device=self.device, dtype=torch.float32))


class RLEManager:
    """RLE管理器，负责目标采样、切换和奖励计算"""
    
    def __init__(self, args):
        self.feature_size = args.rle_feature_size
        self.switch_steps = args.rle_switch_steps
        self.tau = args.rle_tau
        self.int_coef = args.rle_int_coef
        self.device = args.device
        
        # 初始化特征网络
        self.feature_network = RLEFeatureNetwork(args)
        
        # 目标向量管理
        self.current_goals = None
        self.steps_since_resample = torch.zeros(args.num_workers, device=self.device)
        self.num_workers = args.num_workers
        
        # 初始化目标向量
        self.resample_goals()
    
    def resample_goals(self):
        """重新采样目标向量"""
        # 从标准正态分布采样
        goals = torch.randn(self.num_workers, self.feature_size, device=self.device)
        # 归一化目标向量
        goals = goals / torch.norm(goals, dim=1, keepdim=True)
        self.current_goals = goals
        self.steps_since_resample = torch.zeros(self.num_workers, device=self.device)
    
    def step(self, dones):
        """更新RLE状态"""
        # 增加步数计数
        self.steps_since_resample += 1
        
        # 检查哪些worker需要重新采样目标
        need_resample = (self.steps_since_resample >= self.switch_steps) | dones
        
        if need_resample.any():
            # 为需要重新采样的worker生成新目标
            new_goals = torch.randn(self.num_workers, self.feature_size, device=self.device)
            new_goals = new_goals / torch.norm(new_goals, dim=1, keepdim=True)
            
            # 只更新需要重新采样的worker的目标
            self.current_goals = torch.where(
                need_resample.unsqueeze(1), 
                new_goals, 
                self.current_goals
            )
            
            # 重置步数计数
            self.steps_since_resample = torch.where(
                need_resample,
                torch.zeros_like(self.steps_since_resample),
                self.steps_since_resample
            )
    
    def compute_intrinsic_reward(self, states, worker_ids=None):
        """计算内在奖励"""
        if worker_ids is None:
            worker_ids = torch.arange(len(states), device=self.device)
        
        # 获取归一化特征
        normalized_features, raw_features = self.feature_network.get_normalized_features(states)
        
        # 获取对应的目标向量
        goals = self.current_goals[worker_ids]
        
        # 计算内在奖励: F(s, z) = f(s) / ||f(s)|| · z
        feature_norms = torch.norm(normalized_features, dim=1, keepdim=True)
        intrinsic_rewards = torch.sum(normalized_features * goals, dim=1) / (feature_norms.squeeze() + 1e-8)
        
        return intrinsic_rewards * self.int_coef, raw_features.detach()
    
    def update_feature_network(self, policy_network):
        """软更新特征网络"""
        for target_param, policy_param in zip(
            self.feature_network.parameters(), 
            policy_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1 - self.tau) * target_param.data
            )
    
    def update_normalization(self, features_batch):
        """更新特征归一化统计"""
        self.feature_network.update_normalization_stats(features_batch) 