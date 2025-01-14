import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, img_channels=3, featrue_maps=64):
        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(input_dim, featrue_maps*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(featrue_maps*8),
            nn.ReLU(),

            nn.ConvTranspose2d(featrue_maps*8, featrue_maps*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featrue_maps*4),
            nn.ReLU(),

            nn.ConvTranspose2d(featrue_maps*4, featrue_maps*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featrue_maps*2),
            nn.ReLU(),

            nn.ConvTranspose2d(featrue_maps*2, featrue_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(featrue_maps),
            nn.ReLU(),

            nn.ConvTranspose2d(featrue_maps, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)
