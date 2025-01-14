import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()

        layers = []
        dim = latent_dim

        for hidden_dim in hidden_dims[::-1]:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_dims[0], output_dim)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        z = self.decoder(z)
        x_hat = self.output_layer(z)
        x_hat = self.output_activation(x_hat)

        return x_hat


