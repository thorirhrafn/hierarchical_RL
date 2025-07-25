import os
import gymnasium as gym
import torch as T
import numpy as np
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from PIL import Image

from huggingface_hub import hf_hub_download

from minigrid.agents.goal_cnn_dqn import DQNAgent
from agents.utils import pre_process, vae_transform, VAE
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
    goal_obs.append(img_tensor)

    img_path = os.path.join(img_dir, "subgoal2.png")
    img = Image.open(img_path).convert("RGB")   # ensure 3 channels
    img_tensor = vae_transform(img)
    goal_obs.append(img_tensor) 

    # load a pre-trained VAE model
    print("Loading VAE model...")
    file_path = hf_hub_download(repo_id="thorirhrafn/minigrid_vae", filename="vae_model.pt")
    vae_model = VAE(latent_dim=64).to(device)
    vae_model.load_state_dict(T.load(file_path))
    vae_model.eval()   

    # Create the FourRoomMaze environment
    # This environment is a custom maze environment for testing the goal-conditioned DQN agen
    env = FourRoomMazeEnv()
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    obs, _ = env.reset()
    obs = pre_process(obs, device=device)
    input_shape = obs.shape[1:]
    print(f'Input shape: {input_shape}')
    print(f'Number of actions: {env.action_space.n}')
    print(f'Number of goals: {len(goal_obs)}')
    num_actions = env.action_space.n

    agent = DQNAgent(
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

    num_episodes = 500
    max_steps = 100
    rewards = []
    worker_rewards = []

    print("Starting to train the agent!")
    for episode in range(1, num_episodes + 1):
        obs, _ = env.reset()
        obs = pre_process(obs, device=device)
        total_reward = 0
        total_worker_reward = 0
        goal_idx = 0
        subgoal_found = 0
        goal = goal_obs[goal_idx].unsqueeze(0).to(device)

        for t in range(max_steps):
            action = agent.select_action(obs, goal, train=True)
            new_obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            worker_reward = agent.get_mse_reward(vae=vae_model, obs=pre_process(new_obs, device=device), goal=goal)
            current_worker_reward = worker_reward.cpu().item()
            total_worker_reward += current_worker_reward
            if current_worker_reward == -0.0:
              if goal_idx < 1:
                # steps = 0
                subgoal_found += 1
                goal_idx += 1
                goal = goal_obs[goal_idx].unsqueeze(0).to(device)

            done = terminated or truncated

            if terminated:
                next_obs = None
            else:
                next_obs = pre_process(new_obs, device=device)

            # Store the transition in the replay buffer
            agent.buffer.push(obs, action, next_obs, worker_reward, goal)
            agent.optimize()

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