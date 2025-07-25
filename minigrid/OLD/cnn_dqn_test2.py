import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym
from torchvision import transforms
from PIL import Image
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

# ===============================
#   Convolutional Q-Network
# ===============================
class ConvQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w = input_shape
    
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute output dimension after conv layers
        with T.no_grad():
            dummy_input = T.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy_input).view(1, -1).shape[1]

        self.head = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = x / 255.0  # Normalize pixel values [0,255] → [0,1]
        x = self.cnn(x)
        return self.head(x.view(x.size(0), -1))


# ===============================
#   Replay Buffer
# ===============================
class ReplayBuffer:
    def __init__(self, capacity, input_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            T.tensor(self.states[idxs], dtype=T.float32),
            T.tensor(self.actions[idxs], dtype=T.int64),
            T.tensor(self.rewards[idxs], dtype=T.float32),
            T.tensor(self.next_states[idxs], dtype=T.float32),
            T.tensor(self.dones[idxs], dtype=T.bool)
        )


# ===============================
#   DQN Agent
# ===============================
class DQNAgent:
    def __init__(
        self, input_shape, num_actions, device,
        gamma=0.99, lr=1e-4, batch_size=64,
        buffer_size=100_000, tau=0.005,
        epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1e-5
    ):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy_net = ConvQNetwork(input_shape, num_actions).to(device)
        self.target_net = ConvQNetwork(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, input_shape)

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

    def select_action(self, state, train=True):
        if train and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with T.no_grad():
            state_tensor = T.tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def learn(self):
        if self.buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device) / 255.0
        next_states = next_states.to(self.device) / 255.0
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with T.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)


# ===============================
#   Training Loop
# ===============================
def main():
    device = T.device('cuda' if T.cuda.is_available() else 'cpu')
    env = gym.make('MiniGrid-Empty-6x6-v0')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    input_shape = obs.shape
    print(f'Input shape: {input_shape}')
    print(f'Number of actions: {env.action_space.n}')
    num_actions = env.action_space.n

    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        device=device,
        batch_size=64,
        buffer_size=10000,
        epsilon_decay=1e-4,
        epsilon_end=0.1,
        lr=1e-4,
        tau=0.005
    )

    num_episodes = 500
    max_steps = 200
    rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(obs, train=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.learn()

            obs = next_obs
            total_reward += reward
            if done:
                break

        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-50:])
        print(f"Ep {episode} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    env.close()


if __name__ == "__main__":
    main()
