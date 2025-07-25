import gymnasium as gym
import torch as T
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper

from minigrid.agents.cnn_dqn import DQNAgent
from agents.utils import pre_process  

# ===============================
#   Training Loop
# ===============================
def main():
    # device will be 'cuda' if a GPU is available
    gpu_index = 7
    device = T.device(f'cuda:{gpu_index}' if T.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if T.cuda.is_available():
        print(T.cuda.mem_get_info(gpu_index))
        print('( global free memory , total GPU memory )')

    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    obs = pre_process(obs, device=device)
    input_shape = obs.shape[1:]
    print(f'Input shape: {input_shape}')
    print(f'Number of actions: {env.action_space.n}')
    num_actions = env.action_space.n

    agent = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        device=device,
        batch_size=64,
        buffer_size=10000,
        epsilon_decay=5e-5,
        epsilon_start=0.95,
        epsilon_end=0.05,
        lr=1e-4,
        tau=0.005
    )

    num_episodes = 750
    max_steps = 50
    rewards = []

    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs = pre_process(obs, device=device)
        total_reward = 0

        for t in range(max_steps):
            action = agent.select_action(obs, train=True)
            new_obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = T.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_obs = None
            else:
                next_obs = pre_process(new_obs, device=device)

            # Store the transition in the replay buffer
            agent.buffer.push(obs, action, next_obs, reward)
            agent.optimize()

            obs = next_obs

            if done:
                break

        rewards.append(total_reward)
        avg_reward = np.mean(rewards[-20:])
        print(f"Episode {episode} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    env.close()
    print('Training finished!')

if __name__ == "__main__":
    main()    