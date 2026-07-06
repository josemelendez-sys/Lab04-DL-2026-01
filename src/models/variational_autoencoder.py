"""Autoencoder variacional convolucional."""

from __future__ import annotations

import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
    """VAE didáctico con reparametrización explícita."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
        )
        self.mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 64 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 32 -> 64
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), # 64 -> 128
            nn.Tanh(),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(images)
        return self.mu(features), self.logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.from_latent(latent).view(latent.size(0), 64, 8, 8)
        return self.decoder(features)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar
