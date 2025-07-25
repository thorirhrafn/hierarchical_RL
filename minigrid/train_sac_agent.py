import torch as T
import gymnasium as gym
import numpy as np
from agents.sac import SACAgent

def main():
    # device will be 'cuda' if a GPU is available
    gpu_index = 6
    DEVICE = T.device(f'cuda:{gpu_index}' if T.cuda.is_available() else 'cpu')
    print(f'Device: {DEVICE}')
    if T.cuda.is_available():
        print(T.cuda.mem_get_info(gpu_index))
        print('( global free memory , total GPU memory )')

    # Create the environment
    env = gym.make('Ant-v5')
    input_dim = env.observation_space.shape
    num_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]  # Assuming continuous action space
    print(f'Input shape: {input_dim}')
    print(f'Number of actions: {num_actions}')
    print(f'Max action: {max_action}')

    agent = SACAgent(
        state_dim=input_dim,
        action_dim=num_actions,
        max_action=max_action,
        lr=3e-4,
        device=DEVICE
    )

    # agent = SACAgent(
    #     input_dim=input_dim,
    #     num_actions=num_actions,
    #     device=DEVICE,
    #     batch_size=32,
    #     buffer_size=10000,
    #     gamma=0.99,
    #     tau=0.005,
    #     lr=3e-4,
    #     hidden_units=256,
    #     max_action=max_action 
    # )

    num_episodes = 500
    max_steps = 200
    rewards = []

    print('Starting to train the agent!')
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(obs)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store the transition in the replay buffer
            agent.replay_buffer.add(obs, action, reward, new_obs, done)

            # Optimize the policy network
            agent.optimize(batch_size=64)

            # Move to next state
            obs = new_obs

            if done:
                break

        rewards.append(total_reward)
        print(f'Episode {episode}, Total Reward: {total_reward}, Average Reward: {np.mean(rewards[-20:])}')

    print('Training completed!')
    env.close()

if __name__ == "__main__":
    main()