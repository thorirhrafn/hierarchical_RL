import torch as T
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# ===============================
#   Replay Buffer
# ===============================
Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward','goal'))

class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Store transition in buffer memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Return random samples from the buffer for training
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# ===============================
#   Convolutional Q-Network
# ===============================
class GoalDQN(nn.Module):
    def __init__(self, num_actions, image_shape=(3, 56, 56), hidden_units=128):
        super(GoalDQN, self).__init__()

        in_channels, H, W = image_shape
        self.input_channels = in_channels * 2  # concatenate state and goal

        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with T.no_grad():
            sample_input = T.zeros(1, self.input_channels, H, W)
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

    def forward(self, state, goal):
        # Concatenate along channel dimension
        x = T.cat([state, goal], dim=1)
        x = self.cnn(x)
        x = self.fc(x)
        return x

# ===============================
#   DQN Agent
# ===============================
class DQNAgent:
    def __init__(
        self, input_shape, num_actions, device,
        gamma=0.99, lr=1e-4, batch_size=64,
        buffer_size=100_000, tau=0.005,
        epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=1e-5
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_units = 128

        self.policy_net =  GoalDQN(hidden_units=self.hidden_units, num_actions=num_actions, image_shape=input_shape).to(device)
        self.target_net =  GoalDQN(hidden_units=self.hidden_units, num_actions=num_actions, image_shape=input_shape).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, goal, train=True):
        if train and random.random() < self.epsilon:
            return T.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=T.long)
        with T.no_grad():
            return self.policy_net(state, goal).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch data to tensors
        state_batch  = T.cat([s.to(self.device) for s in batch.state])
        next_states  = [s.to(self.device) for s in batch.next_state if s is not None]
        non_final_mask = T.tensor([s is not None for s in batch.next_state], device=self.device, dtype=T.bool)
        non_final_next_states = T.cat(next_states) if next_states else None

        action_batch = T.cat([a.to(self.device).view(1, 1) for a in batch.action])  # shape: [B, 1]
        reward_batch = T.tensor(batch.reward, dtype=T.float32, device=self.device).unsqueeze(1)  # shape: [B, 1]
        goal_batch = T.stack([
            g.view(g.shape[-3:]).to(self.device) for g in batch.goal
        ])

        # Compute Q(s, a)
        # action_batch = action_batch.unsqueeze(1)
        state_action_values = self.policy_net(state_batch, goal_batch).gather(1, action_batch)

        # Compute V(s) for all next states.
        next_state_values = T.zeros(self.batch_size, device=self.device)
        with T.no_grad():
            if non_final_next_states is not None:
                non_final_goal_batch = goal_batch[non_final_mask]
                next_q_values = self.target_net(non_final_next_states, non_final_goal_batch).max(1)[0]
                next_state_values[non_final_mask] = next_q_values
        # Compute expected Q values
        expected_q_values = reward_batch.squeeze() + self.gamma * next_state_values  # shape: [B]
        expected_q_values = expected_q_values.unsqueeze(1)  # shape: [B, 1]

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping for stability
        T.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def get_mse_reward(self, vae, obs, goal):
        _, z_obs, obs_mu, obs_logvar = vae(obs)
        _, z_goal, obs_goal, obs_logvar = vae(goal)
        mse_per_sample = ((obs_mu - obs_goal) ** 2).mean(dim=1)  # shape: (batch_size,)
        reward = -mse_per_sample.mean()
        return reward    

