import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super(Encoder, self).__init__()

        layers = []
        dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        x = self.encoder(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var

