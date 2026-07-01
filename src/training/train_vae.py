"""Entrenamiento de VAE."""

from __future__ import annotations

import torch

from src.training.losses import vae_loss


def train_vae(model, loader, config: dict, device, output_path):
    """Entrena un VAE con reconstrucción y KL."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["vae"]["learning_rate"]))
    beta = float(config["vae"].get("beta", 1.0))
    total_epochs = int(config["vae"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        reconstruction_loss = 0.0
        kl_loss = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(images)
            loss, reconstruction_term, kl_term = vae_loss(reconstruction, images, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            sample_count += batch_size
            total_loss += loss.item() * batch_size
            reconstruction_loss += reconstruction_term.item() * batch_size
            kl_loss += kl_term.item() * batch_size

        if sample_count == 0:
            raise RuntimeError("El DataLoader no contiene muestras.")
        print(
            f"[VAE] Epoca {epoch}/{total_epochs} - "
            f"loss={total_loss / sample_count:.6f} - "
            f"reconstruction_mse={reconstruction_loss / sample_count:.6f} - "
            f"kl_loss={kl_loss / sample_count:.6f}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)
