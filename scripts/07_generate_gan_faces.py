"""Genera imágenes sintéticas usando GAN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentations.generated_assets import resolve_generation_seed, should_generate_assets
from src.augmentations.gan_augmentation import generate_gan_images
from src.config import load_config
from src.data.age_groups import build_age_groups, group_midpoint
from src.utils.file_utils import write_csv
from src.utils.seed import set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Genera imágenes con GAN.")
    parser.add_argument("--config", default="config/path.yaml")
    parser.add_argument("--gender", type=int, default=0)
    parser.add_argument(
        "--use-existing",
        action="store_true",
        help="Reutiliza imágenes GAN existentes y falla si no están completas.",
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Regenera imágenes GAN aunque ya existan.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Semilla de generación GAN.")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        generate = should_generate_assets(
            config,
            "gan",
            use_existing=args.use_existing,
            force_generate=args.force_generate,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if not generate:
        return 0

    seed = resolve_generation_seed(config, "gan", args.seed)
    set_seed(seed)
    print(f"Generando imágenes GAN con semilla {seed}.")

    import torch

    from src.models.model_factory import build_gan

    groups = build_age_groups(config)
    rows: list[dict] = []
    for group in groups:
        generator, _ = build_gan(config)
        checkpoint = Path(config["paths"]["models_dir"]) / "gan" / group.name / "generator.pt"
        generator.load_state_dict(torch.load(checkpoint, map_location="cpu")["model_state_dict"])
        rows.extend(
            generate_gan_images(
                generator,
                config,
                group.name,
                age=group_midpoint(group.name, groups),
                gender=args.gender,
            )
        )

    if rows:
        write_csv(Path(config["paths"]["metadata_dir"]) / "gan_metadata.csv", rows, list(rows[0]))
    print(f"Imágenes GAN registradas: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
