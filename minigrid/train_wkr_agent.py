import os
import gymnasium as gym
import torch as T
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from PIL import Image

from huggingface_hub import hf_hub_download

from agents.dqn import DQNAgent
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

    # load pre-defined subgoal images
    # These images should be in the same format as the VAE inpu
    goal_obs = []
    img_dir = "./subgoals/"

    img_path = os.path.join(img_dir, "subgoal1.png")
    img = Image.open(img_path).convert("RGB")   # ensure 3 channels
    img_tensor = vae_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # move to the same device as the model
    goal_obs.append(img_tensor)

    img_path = os.path.join(img_dir, "subgoal2.png")
    img = Image.open(img_path).convert("RGB")   # ensure 3 channels
    img_tensor = vae_transform(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # move to the same device as the model
    goal_obs.append(img_tensor) 

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
    input_shape =obs_shape * 2  # concatenate observation and goal, so double the latent dimension

    print(f'Input shape: {input_shape}')
    print(f'Number of actions: {num_actions}')
    print(f'Number of goals: {len(goal_obs)}')
    

    agent = DQNAgent(
        input_shape=input_shape,
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

    num_episodes = 500
    max_steps = 100
    rewards = []
    worker_rewards = []

    # grid information to track subgoal progress
    grid_size = env.unwrapped.height, env.unwrapped.width
    visit_counts1 = np.zeros(grid_size, dtype=int)
    visit_counts2 = np.zeros(grid_size, dtype=int)

    print("Starting to train the agent!")
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs_tensor = pre_process(obs, device=device)
        z_obs, _ = vae_model.encode(obs_tensor)
        z_obs = z_obs.detach().cpu().numpy()  # convert to numpy array
        total_reward = 0
        total_worker_reward = 0
        goal_idx = 0
        subgoal_found = 0
        goal_tensor = goal_obs[goal_idx]
        # print(f'Goal tensor shape: {goal_tensor.shape}')
        z_goal, _ = vae_model.encode(goal_tensor)
        z_goal = z_goal.detach().cpu().numpy()  # convert to numpy array
        # print(f'goal vector: {z_goal}')

        for t in range(max_steps):
            agent_pos = env.unwrapped.agent_pos
            x, y = agent_pos
            if goal_idx == 1:
               visit_counts2[y, x] += 1
            else:
                visit_counts1[y, x] += 1

            obs_goal = np.concatenate([z_obs, z_goal], axis=0).reshape(-1)  # concatenate observation and goal arrays
            action = agent.select_action(obs_goal, train=True)
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
            z_obs = next_z_obs
            
            total_worker_reward += worker_reward
            if worker_reward == -0.0:
              if goal_idx < 1:
                # steps = 0
                subgoal_found += 1
                goal_idx += 1
                goal_tensor = goal_obs[goal_idx]
                z_goal, _ = vae_model.encode(goal_tensor)
                z_goal = z_goal.detach().cpu().numpy()  # convert to numpy array
                # print(f'new goal vector: {z_goal}')

            done = terminated or truncated
            next_obs_goal = np.concatenate([next_z_obs, z_goal], axis=0).reshape(-1)  # concatenate next observation and goal arrays

            # Store the transition in the replay buffer
            agent.replay_buffer.add(obs_goal, action, worker_reward, next_obs_goal, done)
            agent.optimize()

            if done:
                break

        rewards.append(total_reward)
        worker_rewards.append(total_worker_reward)
        avg_reward = np.mean(rewards[-20:])
        avg_worker_reward = np.mean(worker_rewards[-20:])
        print(f"Episode {episode} | Subgoal Found: {subgoal_found} | Worker Reward: {total_worker_reward:.2f} | Avg: {avg_worker_reward:.2f} | Reward: {total_reward:.2f} | Avg: {avg_reward:.2f} | Eps: {agent.epsilon:.3f}")

    env.close()
    print('Training finished!')

    print("Visit counts for subgoal 1:")
    print(visit_counts1)
    print("Visit counts for subgoal 2:")
    print(visit_counts2)

if __name__ == "__main__":
    main()    