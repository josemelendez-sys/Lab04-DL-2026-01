"""Entrenamiento básico de DCGAN."""

from __future__ import annotations

import torch
from torch import nn


def train_gan(generator, discriminator, loader, config: dict, device, output_dir):
    """Entrena un DCGAN pequeño para fines didácticos."""

    generator.to(device)
    discriminator.to(device)
    criterion = nn.BCEWithLogitsLoss()
    latent_dim = int(config["gan"]["latent_dim"])
    opt_g = torch.optim.Adam(
        generator.parameters(),
        lr=float(config["gan"]["learning_rate_generator"]),
        betas=(float(config["gan"].get("beta1", 0.5)), 0.999),
    )
    opt_d = torch.optim.Adam(
        discriminator.parameters(),
        lr=float(config["gan"]["learning_rate_discriminator"]),
        betas=(float(config["gan"].get("beta1", 0.5)), 0.999),
    )

    total_epochs = int(config["gan"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        total_d_loss = 0.0
        total_g_loss = 0.0
        real_correct = 0
        fake_correct = 0
        sample_count = 0
        for batch in loader:
            real = batch["image"].to(device)
            batch_size = real.size(0)
            real_targets = torch.ones(batch_size, device=device)
            fake_targets = torch.zeros(batch_size, device=device)

            opt_d.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = generator(noise).detach()
            real_logits = discriminator(real)
            fake_logits = discriminator(fake)
            d_loss = criterion(real_logits, real_targets) + criterion(fake_logits, fake_targets)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake = generator(noise)
            g_loss = criterion(discriminator(fake), real_targets)
            g_loss.backward()
            opt_g.step()

            sample_count += batch_size
            total_d_loss += d_loss.item() * batch_size
            total_g_loss += g_loss.item() * batch_size
            real_correct += (real_logits >= 0).sum().item()
            fake_correct += (fake_logits < 0).sum().item()

        if sample_count == 0:
            raise RuntimeError("El DataLoader no contiene muestras.")
        print(
            f"[GAN] Epoca {epoch}/{total_epochs} - "
            f"d_loss={total_d_loss / sample_count:.6f} - "
            f"g_loss={total_g_loss / sample_count:.6f} - "
            f"d_real_acc={real_correct / sample_count:.6f} - "
            f"d_fake_acc={fake_correct / sample_count:.6f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": generator.state_dict(), "config": config}, output_dir / "generator.pt")
    torch.save(
        {"model_state_dict": discriminator.state_dict(), "config": config},
        output_dir / "discriminator.pt",
    )
