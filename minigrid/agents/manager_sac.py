import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    def __init__(self, buffer_size, input_dim, action_dim):
        self.size = buffer_size
        self.ptr = 0
        self.count = 0
        self.state = np.zeros((buffer_size, *input_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, *input_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.count, size=batch_size)
        return (
            torch.tensor(self.state[idxs]),
            torch.tensor(self.action[idxs]),
            torch.tensor(self.reward[idxs]),
            torch.tensor(self.next_state[idxs]),
            torch.tensor(self.done[idxs])
        )


class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(input_dim[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim[0] + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.apply(init_weights)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    def __init__(self, obs_dim, action_dim, max_action, device='cpu'):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

        self.actor = GaussianPolicy(obs_dim, action_dim, max_action).to(device)
        self.q1 = QNetwork(obs_dim, action_dim).to(device)
        self.q2 = QNetwork(obs_dim, action_dim).to(device)
        self.q1_target = QNetwork(obs_dim, action_dim).to(device)
        self.q2_target = QNetwork(obs_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        self.replay_buffer = ReplayBuffer(1_000_000, obs_dim, action_dim)

    def select_action(self, state, evaluate=False):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        if evaluate:
            action = torch.tanh(mean)
        else:
            normal = Normal(mean, std)
            action = torch.tanh(normal.sample())
        return (action * self.actor.max_action).cpu().detach().numpy()[0]

    def optimize(self, batch_size):
        s, a, r, s2, d = self.replay_buffer.sample(batch_size)
        s, a, r, s2, d = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device)

        with torch.no_grad():
            next_action, log_prob = self.actor.sample(s2)
            target_q1 = self.q1_target(s2, next_action)
            target_q2 = self.q2_target(s2, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * log_prob
            target = r + (1 - d) * self.gamma * target_q

        current_q1 = self.q1(s, a)
        current_q2 = self.q2(s, a)

        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        actions, log_pi = self.actor.sample(s)
        q1_pi = self.q1(s, actions)
        q2_pi = self.q2(s, actions)
        min_q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, net, target):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
