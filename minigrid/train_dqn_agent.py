import gymnasium as gym
import numpy as np
import torch as T
from agents.dqn import DQNAgent

# ===============================
#   Training Loop
# ===============================
def main():
    # device will be 'cuda' if a GPU is available
    gpu_index = 6
    device = T.device(f'cuda:{gpu_index}' if T.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if T.cuda.is_available():
        print(T.cuda.mem_get_info(gpu_index))
        print('( global free memory , total GPU memory )')

    env = gym.make('LunarLander-v3')
    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f'Input shape: {input_dim}')
    print(f'Number of actions: {num_actions}')

    agent = DQNAgent(
        input_shape=input_dim,
        num_actions=num_actions,
        device=device,
        batch_size=16,
        buffer_size=10000,
        epsilon_decay=3e-5,
        epsilon_start=0.95,
        epsilon_end=0.05,
        lr=1e-4,
        tau=0.005
    )

    num_episodes = 500
    max_steps = 1000
    rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(obs, train=True)
            # print(f"Action selected: {action.item()}")
            action = action.item()
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store the transition in the replay buffer
            agent.replay_buffer.add(obs, action, reward, new_obs, done)
            agent.optimize()

            obs = new_obs

            if done:
                break

        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-20:])
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    env.close()
    print('Training finished!')

if __name__ == "__main__":
    main()    