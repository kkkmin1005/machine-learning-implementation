import torch
from models.VAE import VAE
from utils import loss_function
from config import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

input_dim = config['input_dim']
hidden_dims = config['hidden_dims']
latent_dim = config['latent_dim']
learning_rate = config['learning_rate']
epochs = config['epochs']
batch_size = config['batch_size']

# MNIST 데이터 로드 및 전처리
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Lambda(lambda x: x.view(-1))  # 이미지를 (1, 28, 28)에서 (784,)로 변환
])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vae = VAE(input_dim, hidden_dims, latent_dim).to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr = learning_rate)

for epoch in range(epochs):
    vae.train()
    total_loss = 0

    for batch, _ in train_loader:
        batch = batch.to(device)

        x_reconstructed, mu, log_var = vae(batch)

        loss = loss_function(batch, x_reconstructed, mu, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

torch.save(vae.state_dict(), "vae_mnist.pth")