import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

from PIL import Image

import huggingface_hub
from huggingface_hub import create_repo
from huggingface_hub import HfApi, upload_folder

from vae_network import VAE
from tqdm import tqdm

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

    # Load dataset
    full_dataset = load_dataset('thorirhrafn/minigrid_obs_data', split='train')
    split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']

    # Convert to torch.Tensor during batch loading
    train_dataset = train_dataset.with_transform(
        lambda example: {
            "image_tensor": T.tensor(example["image_tensor"], dtype=T.float32)
        }
    )
    test_dataset = test_dataset.with_transform(
        lambda example: {
            "image_tensor": T.tensor(example["image_tensor"], dtype=T.float32)
        }
    )

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f'train dataset size: {len(train_dataset)}')
    print(f'test dataset size: {len(test_dataset)}')

    # Initialize model and optimizer
    model = VAE(latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)

    # Loss function
    def vae_loss(recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld = -0.5 * T.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld

    # Training loop
    print("Starting training...")
    epochs = 10
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            x = batch["image_tensor"].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    print("Training complete!")
    T.save(model.state_dict(), "./model_folder/vae_model.pt")

    return

if __name__ == "__main__":
    main()