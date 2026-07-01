"""Genera imágenes sintéticas usando VAE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.generated_assets import resolve_generation_seed, should_generate_assets
from src.augmentations.vae_augmentation import generate_vae_images
from src.config import load_config
from src.data.age_groups import build_age_groups, group_midpoint
from src.utils.file_utils import write_csv
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera imágenes con VAE.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--gender", type=int, default=0)
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Reutiliza imágenes VAE existentes y falla si no están completas.",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenera imágenes VAE aunque ya existan.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semilla de generación VAE.")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        generate = should_generate_assets(
            config,
            "vae",
            use_existing=args.use_existing,
            force_generate=args.force_generate,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not generate:
        return 0

    seed = resolve_generation_seed(config, "vae", args.seed)
    set_seed(seed)
    print(f"Generando imágenes VAE con semilla {seed}.")

    import torch

    from src.models.model_factory import build_vae

    groups = build_age_groups(config)
    rows: list[dict] = []
    for group in groups:
        model = build_vae(config)
        checkpoint = Path(config["paths"]["models_dir"]) / "vae" / group.name / "best_model.pt"
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
        rows.extend(
            generate_vae_images(
                model,
                config,
                group.name,
                age=group_midpoint(group.name, groups),
                gender=args.gender,
            )
        )

    if rows:
        write_csv(Path(config["paths"]["metadata_dir"]) / "vae_metadata.csv", rows, list(rows[0]))
    print(f"Imágenes VAE registradas: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
