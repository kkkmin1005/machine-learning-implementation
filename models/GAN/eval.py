import torch
from models.GAN import GAN
from config import Config
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def evaluate():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GAN 모델 초기화 및 Generator 불러오기
    gan = GAN(cfg.latent_dim, cfg.img_channels, cfg.feature_maps)
    generator, _ = gan.get_models()
    generator.to(device)
    generator.load_state_dict(torch.load("GAN_CIFAR10.pth"))  # 사전 학습된 가중치 로드
    generator.eval()

    # 노이즈 생성 및 가짜 이미지 생성
    noise = torch.randn(64, cfg.latent_dim, 1, 1).to(device)
    fake_images = generator(noise)

    # 가짜 이미지를 시각화
    fake_images = (fake_images * 0.5 + 0.5).clamp(0, 1)  # [-1, 1] → [0, 1]로 변환
    grid = vutils.make_grid(fake_images, nrow=8)  # 이미지 그리드 생성

    # Matplotlib으로 출력
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # (C, H, W) → (H, W, C)로 변환
    plt.show()

if __name__ == "__main__":
    evaluate()
