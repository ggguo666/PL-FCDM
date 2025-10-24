import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, input_shape, latent_dim, device):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.upper_triangular_size = input_shape[0] * (input_shape[0] -1) // 2  # 上三角部分元素数量
        self.latent_dim_size = latent_dim * (latent_dim - 1) // 2  # 上三角部分元素数量
        self.latent_dim = latent_dim
        self.device = device

        self.encoder = nn.Sequential(
            nn.Linear(self.upper_triangular_size, 2048),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(2048, 1024),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(1024, 512),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(512, self.latent_dim_size),  # Output a flattened latent_dim x latent_dim matrix
            # nn.Linear(512, latent_dim * latent_dim),
            # nn.Linear(1024, self.latent_dim),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),
        )

        self.decoder = nn.Sequential(
            # nn.Linear(latent_dim * latent_dim, 512),
            nn.Linear(self.latent_dim_size, 512),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(1024, 2048),
            # nn.Dropout(0.2),
            # nn.LeakyReLU(0.2),

            nn.Linear(2048, self.upper_triangular_size),

        )
    ############输出两个都是对称矩阵

    def forward(self, x):
        # 将输入 x 移到对应的设备上，例如 GPU
        x = x.to(self.device)

        encoded_matrix_flat = self.encoder(x)

        # 将 decoded_flat reshape 成与输入上三角大小匹配的张量
        encoded_matrix = torch.zeros(encoded_matrix_flat.size(0), self.latent_dim, self.latent_dim, dtype=torch.float32,
                              device=self.device)

        # 填充上三角部分
        triu_indices = torch.triu_indices(self.latent_dim, self.latent_dim, offset=1, device=self.device)

        encoded_matrix[:, triu_indices[0], triu_indices[1]] = encoded_matrix_flat

        # 在进行转置和相加操作之前先克隆 decoded 张量
        encoded_matrix = encoded_matrix.clone()

        # 填充下三角部分（通过转置和相加）
        encoded_matrix1 = encoded_matrix.permute(0, 2, 1).clone()  # 克隆 decoded1 张量
        encoded_matrix += encoded_matrix1

        # 将对角线元素设置为 1
        diag_indices = torch.arange(self.latent_dim, device=self.device)
        encoded_matrix[:, diag_indices, diag_indices] = 1.0
        # Reshape to (batch_size, 1, 116, 116)
        encoded_matrix = encoded_matrix.view(-1, 1, self.latent_dim, self.latent_dim)




        decoded_flat = self.decoder(encoded_matrix_flat)

        # 将 decoded_flat reshape 成与输入上三角大小匹配的张量
        decoded = torch.zeros(decoded_flat.size(0), self.input_shape[0], self.input_shape[1], dtype=torch.float32,
                              device=self.device)
        # 填充上三角部分
        triu_indices = torch.triu_indices(self.input_shape[0], self.input_shape[1], offset=1, device=self.device)
        decoded[:, triu_indices[0], triu_indices[1]] = decoded_flat
        # 在进行转置和相加操作之前先克隆 decoded 张量
        decoded = decoded.clone()
        # 填充下三角部分（通过转置和相加）
        decoded1 = decoded.permute(0, 2, 1).clone()  # 克隆 decoded1 张量
        decoded += decoded1
        # 将对角线元素设置为 1
        diag_indices = torch.arange(self.input_shape[0], device=self.device)
        decoded[:, diag_indices, diag_indices] = 1.0
        # Reshape to (batch_size, 1, 116, 116)
        decoded = decoded.view(-1, 1, self.input_shape[0], self.input_shape[1])
        # 返回时记得将 encoded_matrix 也移到相同的设备上

        return decoded, encoded_matrix



    # ############最后输出decoder是对称矩阵
    # def forward(self, x):
    #     # 将输入 x 移到对应的设备上，例如 GPU
    #     x = x.to(self.device)
    #
    #     encoded_matrix_flat = self.encoder(x)
    #
    #     # 将编码后的矩阵 reshape 为 latent_dim x latent_dim 的形状
    #     encoded_matrix = encoded_matrix_flat.view(encoded_matrix_flat.size(0), 1, self.latent_dim, self.latent_dim)
    #
    #
    #     decoded_flat = self.decoder(encoded_matrix.view(encoded_matrix.size(0), -1))
    #
    #
    #     # 将 decoded_flat reshape 成与输入上三角大小匹配的张量
    #     decoded = torch.zeros(decoded_flat.size(0), self.input_shape[0], self.input_shape[1], dtype=torch.float32,
    #                           device=self.device)
    #
    #     # 填充上三角部分
    #     triu_indices = torch.triu_indices(self.input_shape[0], self.input_shape[1], offset=1, device=self.device)
    #     decoded[:, triu_indices[0], triu_indices[1]] = decoded_flat
    #
    #     # 在进行转置和相加操作之前先克隆 decoded 张量
    #     decoded = decoded.clone()
    #
    #     # 填充下三角部分（通过转置和相加）
    #     decoded1 = decoded.permute(0, 2, 1).clone()  # 克隆 decoded1 张量
    #     decoded += decoded1
    #
    #     # 将对角线元素设置为 1
    #     diag_indices = torch.arange(self.input_shape[0], device=self.device)
    #     decoded[:, diag_indices, diag_indices] = 1.0
    #     # Reshape to (batch_size, 1, 116, 116)
    #     decoded = decoded.view(-1, 1, self.input_shape[0], self.input_shape[1])
    #     # 返回时记得将 encoded_matrix 也移到相同的设备上
    #     return decoded, encoded_matrix
        # return decoded_flat, encoded_matrix

    # #输出decoder上三角
    # def forward(self, x):
    #     # 将输入 x 移到对应的设备上，例如 GPU
    #     x = x.to(self.device)
    #
    #     encoded_matrix_flat = self.encoder(x)
    #
    #     # 将编码后的矩阵 reshape 为 latent_dim x latent_dim 的形状
    #     encoded_matrix = encoded_matrix_flat.view(encoded_matrix_flat.size(0), 1, self.latent_dim, self.latent_dim)
    #     decoded_flat = self.decoder(encoded_matrix.view(encoded_matrix.size(0), -1))
    #
    #     return decoded_flat, encoded_matrix






