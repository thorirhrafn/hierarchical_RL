import os
import gymnasium as gym
import torch as T
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from PIL import Image

from huggingface_hub import hf_hub_download

from minigrid.agents.goal_cnn_dqn import DQNAgent
from minigrid.agents.sac import SACAgent
from agents.utils import pre_process, vae_transform, VAE

from environments.four_rooms import FourRoomMazeEnv

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


    # load a pre-trained VAE model
    print("Loading VAE model...")
    vae_model = VAE(latent_dim=8)
    vae_model.load_state_dict(T.load(".VAE/model_folder/vae_model_test.pt", weights_only=True))
    vae_model.to(device)
    vae_model.eval()  

    # Create the FourRoomMaze environment
    # This environment is a custom maze environment for testing the goal-conditioned DQN agen
    env = FourRoomMazeEnv()
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    # obs = pre_process(obs, device=device)
    input_shape = obs.shape[1:]
    print(f'Input shape: {input_shape}')
    print(f'Number of actions: {env.action_space.n}')
    num_actions = env.action_space.n

    worker = DQNAgent(
        input_shape=input_shape,
        num_actions=num_actions,
        device=device,
        batch_size=32,
        buffer_size=10000,
        epsilon_decay=1e-4,
        epsilon_start=0.95,
        epsilon_end=0.05,
        lr=1e-4,
        tau=0.005
    )

    manager = SACAgent(
        state_dim=input_shape,
        action_dim=num_actions,
        max_action=1.0,
        lr=3e-4,
        device=device
    )

    num_episodes = 500
    max_steps = 100
    manager_steps = 10
    rewards = []
    worker_rewards = []

    print("Starting to train the agent!")
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs_mu, _ = vae_model.encode(obs.unsqueeze(0).to(device))
        total_reward = 0
        total_worker_reward = 0

        # Select a goal from the VAE latent space
        if episode % manager_steps == 0:
            goal_action = manager.select_action(obs_mu)
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store the transition in the replay buffer
            agent.replay_buffer.add(obs, action, reward, new_obs, done)

        for step in range(max_steps):
            # Select a goal from the VAE latent space
            if step % manager_steps == 0:
                goal_action = manager.select_action(obs_mu)


            action = worker.select_action(obs, goal_action, train=True)
            new_obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            worker_reward = worker.get_mse_reward(vae=vae_model, obs=pre_process(new_obs, device=device), goal=goal)
            current_worker_reward = worker_reward.cpu().item()
            total_worker_reward += current_worker_reward
   
            done = terminated or truncated

            if terminated:
                next_obs = None

            # Store the transition in the replay buffer
            worker.buffer.push(obs, action, next_obs, worker_reward, goal)
            worker.optimize()

            obs = next_obs

            if done:
                break

        rewards.append(total_reward)
        worker_rewards.append(total_worker_reward)
        avg_reward = np.mean(rewards[-20:])
        avg_worker_reward = np.mean(worker_rewards[-20:])
        print(f"Episode {episode} | Subgoal Found: {subgoal_found} | Worker Reward: {total_worker_reward:.2f} | Avg: {avg_worker_reward:.2f} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    env.close()
    print('Training finished!')

if __name__ == "__main__":
    main()    