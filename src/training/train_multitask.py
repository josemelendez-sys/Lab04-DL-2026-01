"""Bucle de entrenamiento del modelo multitarea."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch

from src.training.losses import MultiTaskLoss
from src.utils.file_utils import ensure_dir


def train_multitask_model(model, train_loader, val_loader, config: dict, device, output_dir):
    """Entrena, valida y guarda el mejor modelo por pérdida de validación."""

    output_dir = ensure_dir(output_dir)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = MultiTaskLoss(
        lambda_gender=float(config["losses"]["lambda_gender"]),
        lambda_age=float(config["losses"]["lambda_age"]),
        age_mode=config["age"]["prediction_mode"],
    )
    best_val_loss = float("inf")
    history: list[dict[str, float]] = []
    patience = int(config["training"].get("patience", 8))
    stale_epochs = 0

    total_epochs = int(config["training"]["epochs"])
    for epoch in range(1, total_epochs + 1):
        print(f"[Multitask] Epoca {epoch}/{total_epochs} - entrenamiento...", flush=True)
        train_losses = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            phase="train",
            epoch=epoch,
            total_epochs=total_epochs,
        )
        print(_format_epoch_log("[Multitask][train]", epoch, total_epochs, train_losses), flush=True)

        print(f"[Multitask] Epoca {epoch}/{total_epochs} - validacion...", flush=True)
        val_losses = _run_epoch(
            model,
            val_loader,
            criterion,
            None,
            device,
            phase="validation",
            epoch=epoch,
            total_epochs=total_epochs,
        )
        print(_format_epoch_log("[Multitask][validation]", epoch, total_epochs, val_losses), flush=True)

        row = {"epoch": epoch, **_prefix("train", train_losses), **_prefix("val", val_losses)}
        history.append(row)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            stale_epochs = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "config": config},
                output_dir / "best_model.pt",
            )
            print(
                f"[Multitask] Mejor modelo actualizado en epoca {epoch} "
                f"(val_loss={best_val_loss:.6f}).",
                flush=True,
            )
        else:
            stale_epochs += 1
            if config["training"].get("early_stopping", True) and stale_epochs >= patience:
                print(
                    f"[Multitask] Early stopping en epoca {epoch} "
                    f"sin mejora por {stale_epochs} epocas.",
                    flush=True,
                )
                break

    _write_history(output_dir / "train_log.csv", history)
    return history


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    *,
    phase: str,
    epoch: int,
    total_epochs: int,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total_loss": 0.0, "gender_loss": 0.0, "age_loss": 0.0}
    sample_count = 0
    gender_correct = 0
    age_correct = 0
    age_abs_error = 0.0
    age_squared_error = 0.0
    batch_total = len(loader) if hasattr(loader, "__len__") else 0
    progress_interval = max(1, batch_total // 5) if batch_total else 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["image"].to(device)
            gender_targets = batch["gender"].to(device)
            age_targets = batch["age"].to(device)
            if training:
                optimizer.zero_grad()
            gender_logits, age_output = model(images)
            losses = criterion(gender_logits, age_output, gender_targets, age_targets)
            if training:
                losses.total.backward()
                optimizer.step()
            batch_size = images.size(0)
            sample_count += batch_size
            totals["total_loss"] += losses.total.item() * batch_size
            totals["gender_loss"] += losses.gender.item() * batch_size
            totals["age_loss"] += losses.age.item() * batch_size
            gender_predictions = gender_logits.argmax(dim=1)
            gender_correct += (gender_predictions == gender_targets).sum().item()

            if criterion.age_mode == "classification":
                age_predictions = age_output.argmax(dim=1)
                age_correct += (age_predictions == age_targets.long()).sum().item()
            else:
                age_predictions = age_output.detach().view(-1)
                age_errors = age_predictions - age_targets.float().view(-1)
                age_abs_error += age_errors.abs().sum().item()
                age_squared_error += age_errors.pow(2).sum().item()

            if not training and batch_total and (
                batch_index == batch_total or batch_index % progress_interval == 0
            ):
                print(
                    f"[Multitask][{phase}] Epoca {epoch}/{total_epochs} "
                    f"lote {batch_index}/{batch_total}",
                    flush=True,
                )

    if sample_count == 0:
        raise RuntimeError("El DataLoader no contiene muestras.")

    results = {key: value / sample_count for key, value in totals.items()}
    results["gender_accuracy"] = gender_correct / sample_count
    if criterion.age_mode == "classification":
        results["age_accuracy"] = age_correct / sample_count
    else:
        results["age_mae"] = age_abs_error / sample_count
        results["age_rmse"] = math.sqrt(age_squared_error / sample_count)
    return results


def _prefix(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _format_epoch_log(prefix: str, epoch: int, total_epochs: int, values: dict[str, float]) -> str:
    metrics = " - ".join(f"{key}={value:.6f}" for key, value in values.items())
    return f"{prefix} Epoca {epoch}/{total_epochs} - {metrics}"


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
