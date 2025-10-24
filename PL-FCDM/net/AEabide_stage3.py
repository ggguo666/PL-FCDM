import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim, device):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.upper_triangular_size = input_shape[0] * (input_shape[0] -1) // 2  # 上三角部分元素数量
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(self.upper_triangular_size, 2000),
            nn.Tanh(),
            nn.Linear(2000, 1000),
            nn.Tanh(),
            nn.Linear(1000, 500),
            nn.Tanh(),
            nn.Linear(500, latent_dim * latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * latent_dim, 500),
            nn.Tanh(),
            nn.Linear(500, 1000),
            nn.Tanh(),
            nn.Linear(1000, 2000),
            nn.Tanh(),
            nn.Linear(2000, self.upper_triangular_size)
        )


    def forward(self, x):
        # 将输入 x 移到对应的设备上，例如 GPU
        x = x.to(self.device)
        encoded_matrix_flat = self.encoder(x)
        # 将编码后的矩阵 reshape 为 latent_dim x latent_dim 的形状
        encoded_matrix = encoded_matrix_flat.view(encoded_matrix_flat.size(0), 1, self.latent_dim, self.latent_dim)
        decoded_flat = self.decoder(encoded_matrix.view(encoded_matrix.size(0), -1))
        return decoded_flat, encoded_matrix