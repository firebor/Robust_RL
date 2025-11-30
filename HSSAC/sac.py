from networks.network import Actor, Critic
from utils.utils import ReplayBuffer, convert_to_tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.autograd import grad
from torch.optim.lr_scheduler import StepLR


class SAC(nn.Module):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(SAC,self).__init__()

        self.hessian_id=1
        self.args = args
        self.actor = Actor(self.args.layer_num, state_dim, action_dim, self.args.hidden_dim, \
                           self.args.activation_function, self.args.last_activation, self.args.trainable_std).to(device)

        self.q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation).to(device)
        self.q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function,self.args.last_activation).to(device)
        
        self.target_q_1 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation).to(device)
        self.target_q_2 = Critic(self.args.layer_num, state_dim+action_dim, 1, self.args.hidden_dim, self.args.activation_function, self.args.last_activation).to(device)
        
        self.soft_update(self.q_1, self.target_q_1, 1.)
        self.soft_update(self.q_2, self.target_q_2, 1.)
        
        self.alpha = nn.Parameter(torch.tensor(self.args.alpha_init))
        
        self.data = ReplayBuffer(action_prob_exist = False, max_size = int(self.args.memory_size), state_dim = state_dim, num_action = action_dim,device=device)
        self.target_entropy = - torch.tensor(action_dim)

        self.q_1_optimizer = optim.Adam(self.q_1.parameters(), lr=self.args.q_lr)
        self.q_2_optimizer = optim.Adam(self.q_2.parameters(), lr=self.args.q_lr)
        # self.q_1_optimizer = optim.SGD(self.q_1.parameters(), lr=self.args.q_lr)
        # self.q_2_optimizer = optim.SGD(self.q_2.parameters(), lr=self.args.q_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=self.args.alpha_lr)
        # self.actor_optimizer = optim.SGD(self.actor.parameters(), lr=self.args.actor_lr)
        # self.alpha_optimizer = optim.SGD([self.alpha], lr=self.args.alpha_lr)

        # self.q_1_scheduler = StepLR(self.q_1_optimizer, step_size=1000, gamma=0.95)
        # self.q_2_scheduler = StepLR(self.q_2_optimizer, step_size=1000, gamma=0.95)
        # self.actor_scheduler = StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        # self.alpha_scheduler = StepLR(self.alpha_optimizer, step_size=1000, gamma=0.95)
        
        self.device = device
        self.writer = writer
        
    def put_data(self,transition):
        self.data.put_data(transition)
        
    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)
    
    def get_action(self,state):
        mu,std = self.actor(state)
        dist = Normal(mu, std)
        u = dist.rsample()
        u_log_prob = dist.log_prob(u)
        a = torch.tanh(u)
        a_log_prob = u_log_prob - torch.log(1 - torch.square(a) +1e-3)
        return a, a_log_prob.sum(-1, keepdim=True)
    
    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, dones):
        ###target
        with torch.no_grad():
            next_actions, next_action_log_prob = self.get_action(next_states)
            q_1 = self.target_q_1(next_states, next_actions)
            q_2 = self.target_q_2(next_states, next_actions)
            q = torch.min(q_1,q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)

        hessian_trace = self.compute_hessian_trace(states, actions)
        # 从优化器中获取当前学习率
        current_lr = q_optimizer.param_groups[0]['lr']
        regularization = current_lr/2 * hessian_trace
        targets = rewards + self.args.gamma * v - regularization.detach()
        
        q = Q(states, actions)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        return loss
    
    def actor_update(self, states):
        now_actions, now_action_log_prob = self.get_action(states)
        q_1 = self.q_1(states, now_actions)
        q_2 = self.q_2(states, now_actions)
        q = torch.min(q_1, q_2)
        
        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss,now_action_log_prob
    
    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()    
        loss.backward()
        self.alpha_optimizer.step()
        return loss
    
    def train_net(self, batch_size, n_epi):
        data = self.data.sample(shuffle = True, batch_size = batch_size)
        states, actions, rewards, next_states, dones = convert_to_tensor(self.device, data['state'], data['action'], data['reward'], data['next_state'], data['done'])

        ###q update
        q_1_loss = self.q_update(self.q_1, self.q_1_optimizer, states, actions, rewards, next_states, dones)
        q_2_loss = self.q_update(self.q_2, self.q_2_optimizer, states, actions, rewards, next_states, dones)

        ### actor update
        actor_loss,prob = self.actor_update(states)
        
        ###alpha update
        alpha_loss = self.alpha_update(prob)
        
        self.soft_update(self.q_1, self.target_q_1, self.args.soft_update_rate)
        self.soft_update(self.q_2, self.target_q_2, self.args.soft_update_rate)
        # if self.writer != None:
        #     self.writer.add_scalar("loss/q_1", q_1_loss, n_epi)
        #     self.writer.add_scalar("loss/q_2", q_2_loss, n_epi)
        #     self.writer.add_scalar("loss/actor", actor_loss, n_epi)
        #     self.writer.add_scalar("loss/alpha", alpha_loss, n_epi)
            

# 新增鲁棒SAC算法，考虑参数扰动
class RobustSAC(SAC):
    def __init__(self, writer, device, state_dim, action_dim, args):
        super(RobustSAC, self).__init__(writer, device, state_dim, action_dim, args)
        # 噪声强度参数，控制鲁棒性的权重
        self.sigma = args.sigma if hasattr(args, 'sigma') else 0.03
        # Hessian迹的权重
        self.hessian_weight = args.hessian_weight if hasattr(args, 'hessian_weight') else 1
        
    # 计算Hessian矩阵的迹的近似值
    def compute_hessian_trace(self, states, actions=None):
        # 如果没有提供动作，则使用当前策略生成动作
        if actions is None:
            actions, _ = self.get_action(states)
        
        # 计算Q值
        q_1 = self.q_1(states, actions)
        q_2 = self.q_2(states, actions)
        q = torch.min(q_1, q_2)
        q_mean = q.mean()
        # 计算一阶梯度
        critic_params = list(self.q_1.parameters())

        first_grads = grad(q_mean, critic_params, create_graph=True, retain_graph=True)
        # for grad1 in first_grads:
        #     if not grad1.requires_grad:
        #         grad1.requires_grad_(True)
        
        
        # # 使用Hutchinson估计器计算Hessian迹
        # trace_estimate = 0.0
        
        # # 为每个参数单独计算Hessian-向量乘积
        # for i, g in enumerate(first_grads):
        #     if g is not None:
        #         # 生成随机向量，与梯度形状相同
        #         v = torch.randn_like(g,requires_grad=True)
        #         # 计算Hessian-向量乘积，允许未使用的梯度
        #         hvp = grad(g, critic_params, grad_outputs=v, retain_graph=True, allow_unused=True)
        #         # 只使用对应的参数的hvp
        #         if hvp[i] is not None:
        #             trace_estimate += torch.sum(hvp[i] * v)
        # return trace_estimate

        trace_estimates = 0.0
        num_sample = 10
        for _ in range(num_sample):
            # 为每个参数生成一个服从 Rademacher 分布的随机张量
            v = [torch.randint_like(p, low=0, high=2, dtype=p.dtype, device=p.device) * 2 - 1 for p in critic_params]
            
            # 计算 loss 关于 parameters 的一阶梯度（建立计算图以便后续计算二阶梯度）
            grad_params = torch.autograd.grad(q_mean, critic_params, create_graph=True)
            for grad1 in grad_params:
                if not grad1.requires_grad:
                    grad1.requires_grad_(True)           
            # 计算内积 grad_params 和 v 的和
            grad_dot_v = sum((g * v_elem).sum() for g, v_elem in zip(grad_params, v))
            
            # 再对 grad_dot_v 关于 parameters 进行梯度计算，得到 Hessian 向量积 H * v
            hvp = torch.autograd.grad(grad_dot_v, critic_params, retain_graph=True,allow_unused=True)
            
            # 累加 v^T (H v)
            trace_estimates += sum((v_elem * hv).sum() if hv is not None else 0 for v_elem, hv in zip(v, hvp))
        
        # 对多次采样取平均
        
        trace_estimate = trace_estimates / num_sample
        if self.writer is not None:
            self.writer.add_scalar("robust/hessian_trace", trace_estimate.item(), self.hessian_id)
            self.hessian_id+=1
        return trace_estimate
    
    def q_update(self, Q, q_optimizer, states, actions, rewards, next_states, dones):
        ###target
        with torch.no_grad():
            next_actions, next_action_log_prob = self.get_action(next_states)
            q_1 = self.target_q_1(next_states, next_actions)
            q_2 = self.target_q_2(next_states, next_actions)
            q = torch.min(q_1,q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)

        hessian_trace = self.compute_hessian_trace(states, actions)
        # 从优化器中获取当前学习率
        current_lr = q_optimizer.param_groups[0]['lr']
        regularization = current_lr/2 * hessian_trace
        targets = rewards + self.args.gamma * v - regularization.detach()
        
        q = Q(states, actions)
        loss = F.smooth_l1_loss(q, targets)
        q_optimizer.zero_grad()
        loss.backward()
        q_optimizer.step()
        return loss
    
    # 重写actor_update方法，添加Hessian迹正则化
    # def actor_update1(self, states):
    #     now_actions, now_action_log_prob = self.get_action(states)
    #     q_1 = self.q_1(states, now_actions)
    #     q_2 = self.q_2(states, now_actions)
    #     q = torch.min(q_1, q_2)
        
    #     # 计算Hessian迹
    #     hessian_trace = self.compute_hessian_trace(states, now_actions)
        
    #     # 原始SAC损失
    #     original_loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        
    #     # 添加Hessian迹正则化项
    #     # 由于在极大值处Hessian为负定，我们希望最小化其绝对值，即最大化其值
    #     current_lr = self.actor_optimizer.param_groups[0]['lr']
    #     regularization = current_lr/2 * hessian_trace
        
    #     # 总损失
    #     loss = original_loss - regularization
    #     # loss = original_loss+100
        
    #     self.actor_optimizer.zero_grad()
    #     loss.backward()
    #     self.actor_optimizer.step()
        
    #     # 记录Hessian迹
    #     if self.writer is not None:
    #         self.writer.add_scalar("robust/hessian_trace", hessian_trace.item(), 0)
    #         self.writer.add_scalar("robust/regularization", regularization.item(), 0)
        
    #     return loss, now_action_log_prob
            
    # def compute_hessian_trace(self, states, actions=None):
    #     # 确保输入需要梯度
    #     if not states.requires_grad:
    #         states.requires_grad_(True)
        
    #     if actions is None:
    #         with torch.set_grad_enabled(True):  # 确保梯度计算开启
    #             actions, _ = self.get_action(states)
        
    #     # 确保动作需要梯度
    #     if not actions.requires_grad:
    #         actions.requires_grad_(True)
        
    #     # 计算Q值
    #     q_1 = self.q_1(states, actions)
    #     q_2 = self.q_2(states, actions)
    #     q = torch.min(q_1, q_2)
        
    #     # 获取actor参数
    #     actor_params = list(self.actor.parameters())
        
    #     # 检查是否所有参数都需要梯度
    #     for param in actor_params:
    #         if not param.requires_grad:
    #             param.requires_grad_(True)
        
    #     # 计算一阶梯度
    #     first_grads = grad(q.mean(), actor_params, create_graph=True, retain_graph=True)
        
    #     # 计算Hessian迹
    #     trace_estimate = torch.tensor(0.0, device=states.device, requires_grad=True)
        
    #     for i, g in enumerate(first_grads):
    #         if g is not None:
    #             # 创建随机向量
    #             v = torch.randn_like(g)
    #             # 计算Hessian-vector乘积
    #             hvp = grad(g, actor_params, grad_outputs=v, retain_graph=True)
    #             # 确保hvp不为None
    #             if hvp[i] is not None:
    #                 # 使用加法运算符以保持梯度链
    #                 trace_estimate = trace_estimate + torch.sum(hvp[i] * v)
        
    #     return trace_estimate

    # def actor_update(self, states):
    #     # 确保states需要梯度
    #     if not states.requires_grad:
    #         states.requires_grad_(True)
        
    #     now_actions, now_action_log_prob = self.get_action(states)
    #     q_1 = self.q_1(states, now_actions)
    #     q_2 = self.q_2(states, now_actions)
    #     q = torch.min(q_1, q_2)
        
    #     # 原始损失
    #     original_loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        
    #     # 计算Hessian迹
    #     hessian_trace = self.compute_hessian_trace(states, now_actions)
        
    #     # 计算正则化项
    #     regularization = (self.sigma**2 / 2) * hessian_trace
        
    #     # 确保regularization有梯度
    #     print(f"regularization requires_grad: {regularization.requires_grad}")
        
    #     # 如果只想使用正则化项
    #     loss=original_loss - self.hessian_weight * regularization
    #     # loss=regularization
        
    #     # 确保loss有梯度
    #     print(f"loss requires_grad: {loss.requires_grad}")
        
    #     # 清零梯度
    #     self.actor_optimizer.zero_grad()
        
    #     # 反向传播
    #     loss.backward()
        
    #     # 打印梯度信息
    #     for name, param in self.actor.named_parameters():
    #         if param.grad is not None:
    #             print(f"Param {name} has grad norm: {param.grad.norm().item()}")
    #         else:
    #             print(f"Param {name} has NO gradient")
        
    #     # 更新参数
    #     self.actor_optimizer.step()
        
    #     return loss, now_action_log_prob
