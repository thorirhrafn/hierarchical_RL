import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, lr, input_shape, num_actions, hidden_units=128, device='cpu'):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(*input_shape, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.to(device)

    def forward(self, obs):
        x = self.fc(obs)
        return x
    

class Agent():
    def __init__(self, gamma, epsilon, lr, input_shape, num_actions, batch_size, 
                 device='cpu',
                 max_buffer_size=10000, 
                 eps_end=0.05, 
                 eps_decay=1e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.action_space = [i for i in range(num_actions)]
        self.buffer_size = max_buffer_size
        self.device = device
        self.index_counter = 0

        self.policy_net = DQN(lr, input_shape=input_shape, num_actions=num_actions, device=device, hidden_units=128)
        self.target_net = DQN(lr, input_shape=input_shape, num_actions=num_actions, device=device, hidden_units=128)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.state_memory = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.next_state_memory = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        self.done_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.index_counter % self.buffer_size
        # Store the transition in the replay buffer
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.done_memory[index] = done
        # Increment the memory counter
        self.index_counter += 1

    def select_action(self, obs, train=True):
        #epsilon-greedy action selection
        # If training, select a random action with probability epsilon
        # Otherwise, select the action with the highest Q-value
        if train:
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space)
            else:
                obs_tensor = T.tensor([obs]).to(self.device)
                q_values = self.policy_net(obs_tensor)
            return T.argmax(q_values).item()    
        else:
            obs_tensor = T.tensor([obs]).to(self.device)
            with T.no_grad():
                q_values = self.policy_net(obs_tensor)
            return T.argmax(q_values).item()
        
    def learn(self):
        if self.index_counter < self.batch_size:
            return
        
        # Sample a batch of transitions from the replay buffer
        # Ensure we do not sample more than available transitions
        max_buffer_size = min(self.index_counter, self.buffer_size)
        batch = np.random.choice(max_buffer_size, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # Convert the batch to tensors 
        state_batch = T.tensor(self.state_memory[batch]).to(self.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.device)
        next_state_batch = T.tensor(self.next_state_memory[batch]).to(self.device)
        done_batch = T.tensor(self.done_memory[batch]).to(self.device)
        # action gets passed as a scaler to the environment
        action_batch = self.action_memory[batch]

        q_values = self.policy_net(state_batch)[batch_index, action_batch]
        # Get the next Q values from the target network
        # We use the target network to stabilize training
        with T.no_grad():
            # Get the Q values for the next states from the target network
            # We use T.max to get the maximum Q value for each action in the next state
            # This is used for the Bellman equation
            # We set the Q values to 0 for terminal states (done_batch) 
            next_q_values = self.target_net(next_state_batch)
            next_q_values[done_batch] = 0.0
        # temporal difference learning
        target_q_values = reward_batch + self.gamma * T.max(next_q_values, dim=1)[0]
        loss = self.policy_net.loss_fn(target_q_values, q_values).to(self.device)

        # Zero the gradients, backpropagate, and update the weights
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

        # Decay epsilon
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.eps_end else self.eps_end  

def main():
    # device will be 'cuda' if a GPU is available
    gpu_index = 7
    DEVICE = T.device(f'cuda:{gpu_index}' if T.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    if T.cuda.is_available():
        print(T.cuda.mem_get_info(gpu_index))
        print('( global free memory , total GPU memory )')

    # Create the LunarLander environment    
    env = gym.make('LunarLander-v3')
    input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=1e-3,
        input_shape=input_shape,
        num_actions=num_actions,
        batch_size=64,
        device=DEVICE,
        max_buffer_size=10000,
        eps_end=0.05,
        eps_decay=1e-4
    )

    n_episodes = 500
    timesteps = 200

    episode_rewards = []

    print('Starting to train the agent!')
    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(timesteps):
            action = agent.select_action(obs, train=True)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store the transition in the replay buffer
            agent.store_transition(obs, action, reward, next_obs, done)

            # Optimize the policy network
            agent.learn()

            # Move to next state
            obs = next_obs
            total_reward += reward

            # Update the target network every 10 episodes
            if episode % 10 == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
            if done:
                break

        episode_rewards.append(total_reward)
        print(f'Episode {episode}, Reward: {total_reward}, Average Reward: {np.mean(episode_rewards[-50:])}, Epsilon: {agent.epsilon:.4f}')
    
    return
    
if __name__ == "__main__":
    main()