import numpy as np
import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, num_actions):
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((buffer_size, *input_shape))
        self.action_buffer = np.zeros((buffer_size, num_actions))
        self.next_state_buffer = np.zeros((buffer_size, *input_shape))
        self.reward_buffer = np.zeros((buffer_size))
        self.done_buffer = np.zeros((buffer_size), dtype=np.bool)

    def add(self, state, action, next_state, reward, done):
        index = self.buffer_counter % self.buffer_size
        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.next_state_buffer[index] = next_state
        self.reward_buffer[index] = reward
        self.done_buffer[index] = done
        self.buffer_counter += 1

    def sample(self, batch_size):
        max_size = min(self.buffer_counter, self.buffer_size)
        batch_idx = np.random.choice(max_size, batch_size, replace=False)
        states = self.state_buffer[batch_idx]
        actions = self.action_buffer[batch_idx]
        next_states = self.next_state_buffer[batch_idx]
        rewards = self.reward_buffer[batch_idx]
        dones = self.done_buffer[batch_idx]

        return states, actions, next_states, rewards, dones
    

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden_units=256):
        super(CriticNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim[0]+num_actions, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        
    def forward(self, state, actions):
        q_value = self.fc(T.cat([state, actions], dim=1))
        return q_value
    
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_units=256):
        super(ValueNetwork, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim[0], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)
        )
        
    def forward(self, state):
        state_value = self.fc(state)
        return state_value    
    
class ActorNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, max_action, hidden_units=256, device='cpu'):
        super(ActorNetwork, self).__init__()
        self.reparam_noise = 1e-6
        self.max_action = max_action
        self.device = device
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim[0], hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_units, num_actions)
        self.sigma = nn.Linear(hidden_units, num_actions)
        
    def forward(self, state):
        x = self.fc(state)
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma  
    
    def sample_action(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        normal = T.distributions.Normal(mu, sigma)
        # Reparameterization trick
        # If reparametrize is True, we use rsample() to allow gradients to flow
        if reparametrize:
            actions = normal.rsample()
        else:                   
            actions = normal.sample()
        
        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        log_prob = normal.log_prob(actions) - T.log(1 - action.pow(2) + self.reparam_noise)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob
    

class SACAgent():
    def __init__(self, input_dim, num_actions, device,
                 gamma=0.99, lr=3e-4, batch_size=256,
                 buffer_size=100000, tau=0.005,
                 hidden_units=256, max_action=1.0, reward_scale=2.0):
        self.device = device
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.reward_scale = reward_scale

        self.actor = ActorNetwork(input_dim, num_actions, max_action, hidden_units, device=device).to(device)
        self.critic_1 = CriticNetwork(input_dim, num_actions, hidden_units).to(device)
        self.critic_2 = CriticNetwork(input_dim, num_actions, hidden_units).to(device)

        self.value_network = ValueNetwork(input_dim, hidden_units).to(device)
        self.target_value_network = ValueNetwork(input_dim, hidden_units).to(device)
        self.target_value_network.load_state_dict(self.value_network.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=lr)

        self.buffer = ReplayBuffer(buffer_size, input_dim, num_actions)

    def select_action(self, state):
        state_tensor = T.tensor([state], dtype=T.float32).to(self.device)
        action, _ = self.actor.sample_action(state_tensor, reparametrize=False)
        return action.cpu().detach().numpy()[0]
    
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, next_state, reward, done)

    def update_network_parameters(self):    
        target_value_params = self.target_value_network.named_parameters()
        value_params = self.value_network.named_parameters()
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = self.tau * value_state_dict[name].clone() + (1 - self.tau) * target_value_state_dict[name].clone()

        self.target_value_network.load_state_dict(value_state_dict)


    def optimize(self):
        if self.buffer.buffer_counter < self.batch_size:
            return

        states, actions, next_states, rewards, dones = self.buffer.sample(self.batch_size)
        # convert numpy arrays to tensors
        states = T.tensor(states, dtype=T.float32).to(self.device)
        actions = T.tensor(actions, dtype=T.float32).to(self.device)
        next_states = T.tensor(next_states, dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device)
        dones = T.tensor(dones).to(self.device)
        print(f'dones shape: {dones.shape}')
        print(f'dones: {dones}')
        values = self.value_network(states).view(-1)
        new_values = self.target_value_network(next_states).view(-1)
        new_values[dones] = 0.0  # Set value to 0 for terminal states

        # Update Value Network
        actions, log_probs = self.actor.sample_action(states, reparametrize=False)
        log_probs = log_probs.view(-1)
        critic_values = T.min(
            self.critic_1(states, actions),
            self.critic_2(states, actions)
        )
        critic_values = critic_values.view(-1)

        self.value_optimizer.zero_grad()
        target_values = critic_values - log_probs
        value_loss = 0.5 * F.mse_loss(values, target_values)
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        # Update Actor Network
        actions, log_probs = self.actor.sample_action(states, reparametrize=True)
        log_probs = log_probs.view(-1)
        critic_values = T.min(
            self.critic_1(states, actions),
            self.critic_2(states, actions)
        )
        critic_values = critic_values.view(-1)


        actor_loss = log_probs - critic_values
        actor_loss = T.mean(actor_loss)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        # Update Critic Networks
        q_hat = self.reward_scale * rewards + self.gamma * new_values
        q1_value = self.critic_1(states, actions).view(-1)
        q2_value = self.critic_2(states, actions).view(-1)

        critic_1_loss = 0.5 * F.mse_loss(q1_value, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_value, q_hat)

        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Update Target Value Network
        self.update_network_parameters()     
