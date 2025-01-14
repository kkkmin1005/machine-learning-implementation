import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.VAE import VAE
from config import config 


# 1. 테스트 데이터 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # (1, 28, 28) -> (784,)
])

test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(input_dim=config['input_dim'], 
          hidden_dims=config['hidden_dims'], 
          latent_dim=config['latent_dim'])

vae.load_state_dict(torch.load("vae_mnist.pth"))
vae = vae.to(device)

vae.eval()

with torch.no_grad():
    for batch, _ in test_loader:
        batch = batch.to(device)
        x_reconstructed, _, _ = vae(batch) 
        break

original = batch.view(-1, 28, 28).cpu() 
reconstructed = x_reconstructed.view(-1, 28, 28).cpu() 

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8): 
    # 원본 이미지
    axes[0, i].imshow(original[i], cmap="gray")
    axes[0, i].axis("off")
    axes[0, i].set_title("Original")

    # 복원 이미지
    axes[1, i].imshow(reconstructed[i], cmap="gray")
    axes[1, i].axis("off")
    axes[1, i].set_title("Reconstructed")

plt.tight_layout()
plt.show()
