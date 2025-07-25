import os
import gymnasium as gym
import torch as T
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from PIL import Image

from huggingface_hub import hf_hub_download

from agents.dqn import DQNAgent
from agents.sac import SACAgent
from VAE.vae_network import VAE
from agents.utils import pre_process, vae_transform
from environments.four_rooms import FourRoomMazeEnv

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

    # load a pre-trained VAE model
    print("Loading VAE model...")
    vae_model = VAE(latent_dim=8)
    vae_model.load_state_dict(T.load("./VAE/model_folder/vae_model.pt", weights_only=True))
    vae_model.to(device)
    vae_model.eval()

    # Create the FourRoomMaze environment
    # This environment is a custom maze environment for testing the goal-conditioned DQN agen
    env = FourRoomMazeEnv()
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    obs_tensor = pre_process(obs, device=device)
    z_obs, _ = vae_model.encode(obs_tensor)
    z_obs = z_obs.detach().cpu().numpy()  # convert to numpy array
    num_actions = env.action_space.n
    obs_shape = z_obs.shape[1]  # shape after encoding
    obs_goal_shape =obs_shape * 2  # concatenate observation and goal, so double the latent dimension

    print(f'Observartion shape: {obs_shape}')
    print(f'Number of actions: {num_actions}')
    

    worker = DQNAgent(
        input_shape=obs_goal_shape,
        num_actions=num_actions,
        device=device,
        batch_size=128,
        buffer_size=10000,
        epsilon_decay=5e-5,
        epsilon_start=0.95,
        epsilon_end=0.05,
        lr=3e-4,
        tau=0.005
    )

    manager = SACAgent(
        state_dim=obs_shape,
        action_dim=obs_shape,
        max_action=1.0,
        lr=3e-4,
        device=device
    )

    num_episodes = 2500
    max_steps = 80
    manager_steps = 8
    rewards = []
    worker_rewards = []

    # grid information to track subgoal progress
    grid_size = env.unwrapped.height, env.unwrapped.width
    visit_counts = np.zeros(grid_size, dtype=int)

    print("Starting to train the agent!")
    for episode in range(1, num_episodes + 1):
        # reset the environment and get the initial observation
        obs, _ = env.reset()
        obs_tensor = pre_process(obs, device=device)
        z_obs, _ = vae_model.encode(obs_tensor)
        z_obs = z_obs.detach().cpu().numpy()  # convert to numpy array
        mgr_obs = z_obs  # manager observation is the VAE encoded observation

        # initialze counters
        total_reward = 0
        total_worker_reward = 0
        subgoal_found = 0
        subgoal_generated = 0

        # Get initial goal from the manager policy
        z_goal = manager.select_action(z_obs)
        # print(f'Initial goal vector: {z_goal}')

        for step in range(max_steps):
            # Track agent position for subgoal progress
            agent_pos = env.unwrapped.agent_pos
            x, y = agent_pos
            visit_counts[y, x] += 1

            obs_goal = np.concatenate([z_obs, z_goal], axis=0).reshape(-1)  # concatenate observation and goal arrays
            action = worker.select_action(obs_goal, train=True)
            action = action.item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # get the MSE reward between the new observation and the goal
            next_obs_tensor = pre_process(next_obs, device=device)
            next_z_obs, _ = vae_model.encode(next_obs_tensor)
            next_z_obs = next_z_obs.detach().cpu().numpy()  # convert to numpy array
            # print(f'Next observation shape: {next_z_obs.shape}')
            # print(f'Goal shape: {z_goal.shape}')
            z_mse = np.mean((next_z_obs - z_goal) ** 2, axis=1)
            worker_reward = -z_mse.mean()            
            
            total_worker_reward += worker_reward
            if worker_reward == -0.0:
                subgoal_found += 1

            done = terminated or truncated

            # Manager selects a goal from the VAE latent space every ten steps
            if step % manager_steps == 0:
                new_goal = manager.select_action(next_z_obs)
                # Store the transition in the manger replay buffer and optimize
                manager.replay_buffer.add(mgr_obs, z_goal, total_reward, next_z_obs, done)
                manager.optimize(batch_size=32)
                # Update the manager observation and goal
                mgr_obs = next_z_obs
                z_goal = new_goal
                subgoal_generated += 1

            z_obs = next_z_obs
            next_obs_goal = np.concatenate([next_z_obs, z_goal], axis=0).reshape(-1)  # concatenate next observation and goal arrays

            # Store the transition in the replay buffer
            worker.replay_buffer.add(obs_goal, action, worker_reward, next_obs_goal, done)
            worker.optimize()

            if done:
                break

        rewards.append(total_reward)
        worker_rewards.append(total_worker_reward)
        avg_reward = np.mean(rewards[-20:])
        avg_worker_reward = np.mean(worker_rewards[-20:])
        print(f"Episode {episode} | Subgoals Generated: {subgoal_generated} |Subgoals Found: {subgoal_found}")
        print(f"Worker Reward: {total_worker_reward:.2f} | Avg: {avg_worker_reward:.2f} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {worker.epsilon:.3f}")

    env.close()
    print('Training finished!')

    print("Visit counts:")
    print(visit_counts)

if __name__ == "__main__":
    main()    