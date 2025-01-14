class Config:
    def __init__(self):
        self.img_size = 32
        self.img_channels = 3
        self.feature_maps = 64
        self.latent_dim = 100
        self.batch_size = 128
        self.lr = 0.0002
        self.epochs = 1
        self.data_path = "./data"
