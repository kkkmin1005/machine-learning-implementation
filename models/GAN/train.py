import torch
import torch.optim as optim
from torch.nn import BCELoss
from models.GAN import GAN
from utils import get_dataloader
from config import Config

def train():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gan = GAN(cfg.latent_dim, cfg.img_channels, cfg.feature_maps)
    generator, discriminator = gan.get_models()
    generator.to(device)
    discriminator.to(device)

    dataloader = get_dataloader(cfg)

    criterion = BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    for epoch in range(cfg.epochs):
        for i, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, 1, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1, 1, 1).to(device)

            # Discriminator
            outputs = discriminator(real_images)
            loss_real = criterion(outputs, real_labels)

            noise = torch.randn(batch_size, cfg.latent_dim, 1, 1).to(device)
            fake_images = generator(noise)
            outputs = discriminator(fake_images.detach())
            loss_fake = criterion(outputs, fake_labels)

            loss_d = loss_real + loss_fake
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            # Generator
            outputs = discriminator(fake_images)
            loss_g = criterion(outputs, real_labels)
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{cfg.epochs}], Step [{i}/{len(dataloader)}], "
                        f"Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")

    torch.save(generator.state_dict(), "GAN_CIFAR10.pth")

if __name__ == "__main__":
    train()