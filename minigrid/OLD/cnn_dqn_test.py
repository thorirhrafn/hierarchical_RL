import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

import gymnasium as gym

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions, hidden_units=128, device='cpu'):
        super(DQN, self).__init__()

        c, h, w = input_shape
        
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 8, kernel_size=3, stride=1, padding=1),   # get the number of channels from input_shape
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(5376, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions)
        )

        self.to(device)

    def forward(self, obs):
        x = self.cnn(obs)
        x = self.fc(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, input_shape):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, *input_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.float32)
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
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_states[idxs],
            self.dones[idxs]
        )    

def pre_process(img, image_size=(56, 56)):    # 56x56 pixels is the MiniGrid observation size
    img_frame = Image.fromarray(img)
    transform_img = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    result = transform_img(img_frame)
    return result   

def pre_process_batch(img_batch, image_size=(56, 56)):    # 56x56 pixels is the MiniGrid observation size
    transform_img = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    result = [transform_img(Image.fromarray(img)) for img in img_batch]
    return T.stack(result)  # shape: [B, C, H, W]

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
        # self.buffer_size = max_buffer_size
        self.device = device
        self.index_counter = 0
        print(f'Input shape in Agent constructor: {input_shape}')
        self.policy_net = DQN(input_shape=input_shape, num_actions=num_actions, device=device, hidden_units=128)
        self.target_net = DQN(input_shape=input_shape, num_actions=num_actions, device=device, hidden_units=128)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Set the target network to evaluation mode

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(max_buffer_size, input_shape)

        # self.state_memory = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        # self.next_state_memory = np.zeros((self.buffer_size, *input_shape), dtype=np.float32)
        # self.action_memory = np.zeros(self.buffer_size, dtype=np.int32)
        # self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        # self.done_memory = np.zeros(self.buffer_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)
        # index = self.index_counter % self.buffer_size
        # Store the transition in the replay buffer
        # self.state_memory[index] = state
        # self.action_memory[index] = action
        # self.reward_memory[index] = reward
        # self.next_state_memory[index] = next_state
        # self.done_memory[index] = done
        # Increment the memory counter
        # self.index_counter += 1

    def select_action(self, obs, train=True):
        #epsilon-greedy action selection
        # If training, select a random action with probability epsilon
        # Otherwise, select the action with the highest Q-value
        if train:
            if np.random.random() < self.epsilon:
                return np.random.choice(self.action_space)
            else:
                obs_tensor = pre_process(obs).to(self.devce) # T.tensor([obs]).to(self.device)
                q_values = self.policy_net(obs_tensor)
            return T.argmax(q_values).item()    
        else:
            obs_tensor = pre_process(obs).to(self.devce) # T.tensor([obs]).to(self.device)
            with T.no_grad():
                q_values = self.policy_net(obs_tensor)
            return T.argmax(q_values).item()
        
    def learn(self):
        if self.buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = pre_process_batch(states).to(self.device)
        next_states = pre_process_batch(next_states).to(self.device)
        actions = actions
        rewards = T.tensor([rewards], device=self.device)
        dones = dones
        # print(f'states: {states}')
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        # print(f'q_values shape: {q_values.shape}')
        # print(f'Q-values: {q_values}')
        with T.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            next_q[dones] = 0.0
            target_q = rewards + self.gamma * next_q

        loss = nn.MSELoss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # soft update target network
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        '''
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
        '''

def main():
    # device will be 'cuda' if a GPU is available
    gpu_index = 7
    DEVICE = T.device(f'cuda:{gpu_index}' if T.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    if T.cuda.is_available():
        print(T.cuda.mem_get_info(gpu_index))
        print('( global free memory , total GPU memory )')

    # Create the LunarLander environment    
    env = gym.make("MiniGrid-Empty-6x6-v0")
    # get the pixel observations
    env = RGBImgPartialObsWrapper(env)
    # # remove the mission field
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    obs = pre_process(obs)
    input_shape = obs.shape
    print(f'Input shape: {input_shape}')
    # input_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = Agent(
        gamma=0.99,
        epsilon=1.0,
        lr=1e-4,
        input_shape=input_shape,
        num_actions=num_actions,
        batch_size=8,
        device=DEVICE,
        max_buffer_size=10000,
        eps_end=0.05,
        eps_decay=1e-5
    )

    n_episodes = 500
    timesteps = 50

    episode_rewards = []

    print('Starting to train the agent!')
    for episode in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(timesteps):
            # obs = pre_process(obs).to(DEVICE)
            # Select an action using the agent's policy
            action = agent.select_action(obs, train=True)
            # Take a step in the environment
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if terminated:
                next_obs = None
            else:
                next_obs = next_obs

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