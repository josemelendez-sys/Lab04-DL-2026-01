"""Autoencoder convolucional pequeño para variaciones de rostros."""

from __future__ import annotations

import torch
from torch import nn


class ConvolutionalAutoencoder(nn.Module):
    """Encoder-decoder con espacio latente lineal."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.flatten = nn.Flatten()
        self.to_latent = nn.Linear(64 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.to_latent(self.flatten(self.encoder_conv(images)))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(latent.size(0), 64, 8, 8)
        return self.decoder(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    def reconstruct_with_noise(self, images: torch.Tensor, noise_std: float = 0.0) -> torch.Tensor:
        latent = self.encode(images)
        if noise_std > 0:
            latent = latent + torch.randn_like(latent) * noise_std
        return self.decode(latent)
