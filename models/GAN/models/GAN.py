import torch
from modules.Generator import *
from modules.Discriminator import *

class GAN:
    def __init__(self, input_dim, img_channels=3, feature_maps=64):
        self.generator = Generator(input_dim, img_channels, feature_maps)
        self.discriminator = Discriminator(img_channels, feature_maps)

    def get_models(self):
        return self.generator, self.discriminator