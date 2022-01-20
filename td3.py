import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils import ReplayPool
from networks import Policy, DoubleQFunc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:

    def __init__(self, seed, state_dim, action_dim, action_lim=1, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256, hidden_size=256, update_interval=2, buffer_size=1e6, target_noise=0.2, target_noise_clip=0.5, explore_noise=0.1):
        self.gamma = gamma #遗忘率
        self.tau = tau #target网络的更新率
        self.batchsize = batchsize
        self.update_interval = update_interval #多久更新一次actor以及target
        self.action_lim = action_lim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        torch.manual_seed(seed)

        # aka critic
        self.q_funcs = DoubleQFunc(state_dim, action_dim, hidden_size=hidden_size).to(device) #q_funcs
        self.q_funcs = torch.nn.DataParallel(self.q_funcs)
        self.target_q_funcs = copy.deepcopy(self.q_funcs) #target_q_funcs网络
        self.target_q_funcs.eval() #固定target_q_funcs网络
        for p in self.target_q_funcs.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device) #policy
        self.policy = torch.nn.DataParallel(self.policy)
        self.target_policy = copy.deepcopy(self.policy) #target_policy
        self.target_q_funcs.eval()
        for p in self.target_policy.parameters(): #固定target_policy
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_funcs.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.replay_pool = ReplayPool(action_dim=action_dim, state_dim=state_dim, capacity=int(buffer_size))

        self._update_counter = 0

    def reallocate_replay_pool(self, new_size: int):
        assert new_size != self.replay_pool.capacity, "错误，超过了replay_buffer的上限"
        new_replay_pool = ReplayPool(capacity=new_size)
        new_replay_pool.initialise(self.replay_pool)
        self.replay_pool = new_replay_pool

    def get_action(self, state, state_filter=None, deterministic=False):
        if state_filter:
            state = state_filter(state)
        state = torch.Tensor(state).view(1,-1).to(device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze().cpu().numpy())
    
    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_funcs.parameters(), self.q_funcs.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_funcs(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch = self.target_policy(nextstate_batch)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_t1, q_t2 = self.target_q_funcs(nextstate_batch, nextaction_batch)
            # take min to mitigate positive bias in q-function training
            q_target = torch.min(q_t1, q_t2)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target
        q_1, q_2 = self.q_funcs(state_batch, action_batch)
        loss_1 = F.mse_loss(q_1, value_target)
        loss_2 = F.mse_loss(q_2, value_target)
        return loss_1, loss_2

    def update_policy(self, state_batch):
        action_batch = self.policy(state_batch)
        q_b1, q_b2 = self.q_funcs(state_batch, action_batch)
        qval_batch = torch.min(q_b1, q_b2)
        policy_loss = (-qval_batch).mean()
        return policy_loss

    def optimize(self, n_updates, state_filter=None):
        q1_loss, q2_loss, pi_loss = 0, 0, None
        for i in range(n_updates):
            samples = self.replay_pool.sample(self.batchsize)
            if state_filter:
                state_batch = torch.FloatTensor(state_filter(samples.state)).to(device)
                nextstate_batch = torch.FloatTensor(state_filter(samples.nextstate)).to(device)
            else:
                state_batch = torch.FloatTensor(samples.state).to(device)
                nextstate_batch = torch.FloatTensor(samples.nextstate).to(device)
            action_batch = torch.FloatTensor(samples.action).to(device)
            reward_batch = torch.FloatTensor(samples.reward).to(device).unsqueeze(1)
            done_batch = torch.FloatTensor(samples.real_done).to(device).unsqueeze(1)
            
            # update q-funcs
            q1_loss_step, q2_loss_step = self.update_q_funcs(state_batch, action_batch, reward_batch, nextstate_batch, done_batch)
            q_loss_step = q1_loss_step + q2_loss_step
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()
            
            self._update_counter += 1

            q1_loss += q1_loss_step.detach().item()
            q2_loss += q2_loss_step.detach().item()

            if self._update_counter % self.update_interval == 0:
                if not pi_loss:
                    pi_loss = 0
                # update policy
                for p in self.q_funcs.parameters():
                    p.requires_grad = False
                pi_loss_step = self.update_policy(state_batch)
                self.policy_optimizer.zero_grad()
                pi_loss_step.backward()
                self.policy_optimizer.step()
                for p in self.q_funcs.parameters():
                    p.requires_grad = True
                # update target policy and q-functions using Polyak averaging
                self.update_target()
                pi_loss += pi_loss_step.detach().item()

        return q1_loss, q2_loss, pi_loss
