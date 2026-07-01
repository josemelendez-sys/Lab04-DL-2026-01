"""Utilidades para reutilizar o regenerar imágenes sintéticas."""

from __future__ import annotations

from pathlib import Path

from src.utils.file_utils import read_csv


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def resolve_generation_seed(config: dict, source: str, cli_seed: int | None = None) -> int:
    """Resuelve la semilla de generación para una fuente sintética."""

    if cli_seed is not None:
        return cli_seed

    generation = config.get("generation", {})
    source_seed = generation.get(f"{source}_seed")
    if source_seed is not None:
        return int(source_seed)

    shared_seed = generation.get("seed")
    if shared_seed is not None:
        return int(shared_seed)

    return int(config["project"]["seed"])


def generated_metadata_path(config: dict, source: str) -> Path:
    """Devuelve la ruta del CSV de metadatos sintéticos."""

    return Path(config["paths"]["metadata_dir"]) / f"{source}_metadata.csv"


def generated_images_dir(config: dict, source: str) -> Path:
    """Devuelve la carpeta raíz de imágenes sintéticas."""

    return Path(config["paths"]["generated_dir"]) / source


def validate_generated_assets(config: dict, source: str) -> list[dict[str, str]]:
    """Valida que existan imágenes y metadatos sintéticos coherentes."""

    images_dir = generated_images_dir(config, source)
    metadata_path = generated_metadata_path(config, source)

    if not images_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de imágenes generadas para {source.upper()}: {images_dir}"
        )

    image_files = [
        path
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        raise FileNotFoundError(
            f"No hay imágenes generadas para {source.upper()} en {images_dir}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe metadata sintética para {source.upper()}: {metadata_path}"
        )

    rows = read_csv(metadata_path)
    if not rows:
        raise RuntimeError(f"La metadata sintética está vacía: {metadata_path}")

    missing_paths = [
        row.get("image_path", "")
        for row in rows
        if not row.get("image_path") or not Path(row["image_path"]).exists()
    ]
    if missing_paths:
        preview = ", ".join(missing_paths[:3])
        raise FileNotFoundError(
            f"La metadata de {source.upper()} referencia imágenes inexistentes: {preview}"
        )

    return rows


def should_generate_assets(
    config: dict,
    source: str,
    *,
    use_existing: bool,
    force_generate: bool,
) -> bool:
    """Decide si se deben generar imágenes o reutilizar las existentes."""

    if use_existing and force_generate:
        raise ValueError("Use solo una opción: --use-existing o --force-generate.")

    if force_generate:
        return True

    try:
        rows = validate_generated_assets(config, source)
    except (FileNotFoundError, RuntimeError):
        if use_existing:
            raise
        return True

    print(f"Usando imágenes {source.upper()} existentes: {len(rows)} registros.")
    return False
