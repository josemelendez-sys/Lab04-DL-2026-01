"""Entrenamiento de autoencoder convolucional."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def train_cae(model, loader, config: dict, device, output_path):
    """Entrena un CAE con pérdida MSE."""

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["cae"]["learning_rate"]))
    total_epochs = int(config["cae"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for batch in loader:
            images = batch["image"].to(device)
            optimizer.zero_grad()
            reconstruction = model(images)
            loss = F.mse_loss(reconstruction, images)
            loss.backward()
            optimizer.step()
            batch_size = images.size(0)
            sample_count += batch_size
            total_loss += loss.item() * batch_size

        if sample_count == 0:
            raise RuntimeError("El DataLoader no contiene muestras.")
        avg_loss = total_loss / sample_count
        print(
            f"[CAE] Epoca {epoch}/{total_epochs} - "
            f"loss={avg_loss:.6f} - reconstruction_mse={avg_loss:.6f}",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": config}, output_path)
