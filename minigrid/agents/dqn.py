import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

# ===============================
class ReplayBuffer:
    def __init__(self, buffer_size, input_dim, action_dim):
        self.size = buffer_size
        self.ptr = 0
        self.count = 0
        self.state = np.zeros((buffer_size, input_dim), dtype=np.float32)
        self.action = np.zeros((buffer_size), dtype=np.int64)
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
            T.tensor(self.state[idxs], dtype=T.float32),
            T.tensor(self.action[idxs], dtype=T.long),
            T.tensor(self.reward[idxs], dtype=T.float32),
            T.tensor(self.next_state[idxs], dtype=T.float32),
            T.tensor(self.done[idxs], dtype=T.float32)
        )

# ===============================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

# ===============================
class DQN(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_units=256):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x) # process input and get action from the fully connected layers
        return x

# ===============================
class DQNAgent:
    def __init__(
        self, input_shape, num_actions, device,
        gamma=0.99, lr=3e-4, batch_size=128,
        buffer_size=100_000, tau=0.005,
        epsilon_start=0.95, epsilon_end=0.05, epsilon_decay=1e-5
    ):
        self.device = device
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.hidden_units = 256

        self.policy_net =  DQN(input_dim=self.input_shape, num_actions=num_actions, hidden_units=self.hidden_units).to(device)
        self.target_net =  DQN(input_dim=self.input_shape, num_actions=num_actions, hidden_units=self.hidden_units).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size, self.input_shape, self.num_actions)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        # self.steps_done = 0

    def select_action(self, state, train=True):
        if train and random.random() < self.epsilon:
            action = T.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=T.long)
            return action
        with T.no_grad():
            state = T.tensor(state, dtype=T.float32).unsqueeze(0).to(self.device)   # unsqueeze to add batch dimension: shape (1, input_dim)
            action = self.policy_net(state).max(1)[1].view(1, 1) # return the action with the highest Q-value: # shape (1,1), dtype long
            return action

    def optimize(self):
        if self.replay_buffer.count < self.batch_size:
            return

        # sample a batch from the replay buffer and move to device
        state, action, reward, new_state, done = self.replay_buffer.sample(self.batch_size)
        state, action, reward, new_state, done = state.to(self.device), action.to(self.device), reward.to(self.device), new_state.to(self.device), done.to(self.device)

        # Compute Q-values for the current states and actions
        action = action.view(-1, 1)  # reshape action to match the output of the network
        # print("Action shape:", action.shape)
        # print(f'action: {action}')
        # print("State shape:", state.shape)
        # print(f'state: {state}')
        q_values = self.policy_net(state)
        # print("Q-values shape:", q_values.shape)
        # Gather the Q-values for the actions taken
        q_values = q_values.gather(1, action)
        # print("Gathered Q-values shape:", q_values.shape)
        # print(f'q_values: {q_values}')
        # print("Reward shape:", reward.shape)
        # print(f'reward: {reward}')
        # print("Done shape:", done.shape)
        # print(f'done: {done}')

        # Compute target Q-values
        with T.no_grad():
            next_q_values = self.target_net(new_state)
            next_q_values = next_q_values.max(1)[0].unsqueeze(1)

        target_q_values = reward + self.gamma * next_q_values * (1.0 - done)
        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping for stability
        T.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
          target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

