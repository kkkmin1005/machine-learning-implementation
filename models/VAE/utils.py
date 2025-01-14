import torch
import torch.nn.functional as F

def loss_function(x, x_hat, mu, log_var):
    reconstruct_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return reconstruct_loss + kl_loss