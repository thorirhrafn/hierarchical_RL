import numpy as np
import torch as T
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
        self.state = np.zeros((buffer_size, input_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((buffer_size, input_dim), dtype=np.float32)
        self.reward = np.zeros((buffer_size, 1), dtype=np.float32)
        self.done = np.zeros((buffer_size, 1), dtype=np.float32)

    def add(self, state, action, reward, new_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = new_state
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.count, size=batch_size)
        return (
            T.tensor(self.state[idxs]),
            T.tensor(self.action[idxs]),
            T.tensor(self.reward[idxs]),
            T.tensor(self.next_state[idxs]),
            T.tensor(self.done[idxs])
        )


class GaussianNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, max_action, hidden_units=256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_units, action_dim)
        self.log_std = nn.Linear(hidden_units, action_dim)
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        log_std = T.clamp(self.log_std(x), -20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = T.tanh(x_t)
        action = y_t * self.max_action
        log_prob = normal.log_prob(x_t) - T.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_units=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )
        self.apply(init_weights)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        return self.net(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, lr=3e-4, device='cpu'):
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2

        self.actor = GaussianNetwork(state_dim, action_dim, max_action).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(10000, state_dim, action_dim)

    def select_action(self, state, evaluate=False):
        state = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.device)
        mean, std = self.actor(state)
        if evaluate:
            action = T.tanh(mean)
        else:
            normal = Normal(mean, std)
            action = T.tanh(normal.sample())
        return (action * self.actor.max_action).cpu().detach().numpy()[0]

    def optimize(self, batch_size):
        if self.replay_buffer.count < batch_size:
            return
        # Sample a batch from the replay buffer
        state, action, reward, new_state, done = self.replay_buffer.sample(batch_size)
        state, action, reward, new_state, done = state.to(self.device), action.to(self.device), reward.to(self.device), new_state.to(self.device), done.to(self.device)

        # Compute target Q values for critics
        with T.no_grad():
            next_action, log_prob = self.actor.sample(new_state)
            target_q1 = self.q1_target(new_state, next_action)
            target_q2 = self.q2_target(new_state, next_action)
            target_q = T.min(target_q1, target_q2) - self.alpha * log_prob
            target = reward + (1 - done) * self.gamma * target_q

        # Update Q networks for critics
        current_q1 = self.q1(state, action)
        current_q2 = self.q2(state, action)

        q1_loss = F.mse_loss(current_q1, target)
        q2_loss = F.mse_loss(current_q2, target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update actor network
        actions, log_pi = self.actor.sample(state)
        q1_pi = self.q1(state, actions)
        q2_pi = self.q2(state, actions)
        min_q_pi = T.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_pi - min_q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Soft update target networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, net, target):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
