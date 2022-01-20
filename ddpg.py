import copy
import numpy as np
import torch
import torch.nn.functional as F

from utils import ReplayPool
from networks import Policy, QFunc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ddpg:

    def __init__(self, seed, state_dim, action_dim, action_lim=1, lr=3e-4, gamma=0.99, tau=5e-3, batchsize=256,
                 hidden_size=256, update_interval=1, buffer_size=1e6, target_noise=0.2, target_noise_clip=0.5,
                 explore_noise=0.1):
        self.gamma = gamma  # 遗忘率
        self.tau = tau  # target网络的更新率
        self.batchsize = batchsize
        self.update_interval = update_interval  # 多久更新一次actor以及target
        self.action_lim = action_lim

        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.explore_noise = explore_noise

        torch.manual_seed(seed)

        # aka critic
        self.q_func = QFunc(state_dim, action_dim, hidden_size=hidden_size).to(device)  # q_funcs
        # self.q_func = torch.nn.DataParallel(self.q_func)
        self.target_q_func = copy.deepcopy(self.q_func)
        self.target_q_func.eval()
        for p in self.target_q_func.parameters():
            p.requires_grad = False

        # aka actor
        self.policy = Policy(state_dim, action_dim, hidden_size=hidden_size).to(device)  # policy
        # self.policy = torch.nn.DataParallel(self.policy)
        self.target_policy = copy.deepcopy(self.policy)
        self.target_policy.eval()
        for p in self.target_policy.parameters():
            p.requires_grad = False

        self.q_optimizer = torch.optim.Adam(self.q_func.parameters(), lr=lr)
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
        state = torch.Tensor(state).view(1, -1).to(device)
        with torch.no_grad():
            action = self.policy(state)
        if not deterministic:
            action += self.explore_noise * torch.randn_like(action)
        action.clamp_(-self.action_lim, self.action_lim)
        return np.atleast_1d(action.squeeze().cpu().numpy())

    def update_target(self):
        """moving average update of target networks"""
        with torch.no_grad():
            for target_q_param, q_param in zip(self.target_q_func.parameters(), self.q_func.parameters()):
                target_q_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_q_param.data)
            for target_pi_param, pi_param in zip(self.target_policy.parameters(), self.policy.parameters()):
                target_pi_param.data.copy_(self.tau * pi_param.data + (1.0 - self.tau) * target_pi_param.data)

    def update_q_func(self, state_batch, action_batch, reward_batch, nextstate_batch, done_batch):
        with torch.no_grad():
            nextaction_batch = self.target_policy(nextstate_batch)
            target_noise = self.target_noise * torch.randn_like(nextaction_batch)
            target_noise.clamp_(-self.target_noise_clip, self.target_noise_clip)
            nextaction_batch += target_noise
            nextaction_batch.clamp_(-self.action_lim, self.action_lim)
            q_target = self.target_q_func(nextstate_batch, nextaction_batch)
            value_target = reward_batch + (1.0 - done_batch) * self.gamma * q_target

        q = self.q_func(state_batch, action_batch)
        loss = F.mse_loss(q, value_target)
        return loss

    def update_policy(self, state_batch):
        action_batch = self.policy(state_batch)
        q = self.q_func(state_batch, action_batch)
        policy_loss = (-q).mean()
        return policy_loss

    def optimize(self, n_updates, state_filter=None):
        q_loss, pi_loss = 0, None
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
            q_loss_step = self.update_q_func(state_batch, action_batch, reward_batch,
                                                                 nextstate_batch, done_batch)
            self.q_optimizer.zero_grad()
            q_loss_step.backward()
            self.q_optimizer.step()

            self._update_counter += 1

            q_loss += q_loss_step.detach().item()

            if not pi_loss:
                pi_loss = 0
            # update policy

            pi_loss_step = self.update_policy(state_batch)
            self.policy_optimizer.zero_grad()
            pi_loss_step.backward()
            self.policy_optimizer.step()

            # update target policy and q-functions using Polyak averaging
            self.update_target()
            pi_loss += pi_loss_step.detach().item()

        return q_loss, pi_loss
