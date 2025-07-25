import torch as T
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

# ===============================
#   Replay Buffer
# ===============================
Transition = namedtuple('Transition',  ('state', 'action', 'next_state', 'reward'))

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
class DQN(nn.Module):
    def __init__(self, num_actions, image_shape=(3, 56, 56), hidden_units=128):
        super(DQN, self).__init__()

        in_channels, _, _ = image_shape

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with T.no_grad():
            sample_input = T.zeros(1, *image_shape)
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

    def forward(self, x):
        x = self.cnn[:-1](x)  # all layers except flatten
        x = self.cnn[-1](x)   # flatten manually
        x = self.fc(x) # get action from the fully connected layers
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

        self.policy_net =  DQN(hidden_units=self.hidden_units, num_actions=num_actions, image_shape=input_shape).to(device)
        self.target_net =  DQN(hidden_units=self.hidden_units, num_actions=num_actions, image_shape=input_shape).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, train=True):
        if train and random.random() < self.epsilon:
            return T.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=T.long)
        with T.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        if len(self.buffer) < self.batch_size:
            return

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=T.bool)
        non_final_next_states = T.cat([s.to(self.device) for s in batch.next_state if s is not None])
        state_batch  = T.cat([s.to(self.device) for s in batch.state])
        action_batch = T.cat([a.to(self.device) for a in batch.action])
        reward_batch = T.cat([r.to(self.device) for r in batch.reward])

        # Compute Q(s, a)
        # action_batch = action_batch.unsqueeze(1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s) for all next states.
        next_state_values = T.zeros(self.batch_size, device=self.device)
        with T.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

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

