from __future__ import annotations

from pathlib import Path

import pytest

from src.augmentations.generated_assets import (
    resolve_generation_seed,
    should_generate_assets,
    validate_generated_assets,
)
from src.utils.file_utils import write_csv


def make_config(tmp_path: Path) -> dict:
    return {
        "project": {"seed": 11},
        "generation": {"seed": 22, "vae_seed": 33},
        "paths": {
            "generated_dir": str(tmp_path / "generated"),
            "metadata_dir": str(tmp_path / "metadata"),
        },
    }


def test_resolve_generation_seed_priority(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert resolve_generation_seed(config, "vae", 44) == 44
    assert resolve_generation_seed(config, "vae") == 33
    assert resolve_generation_seed(config, "gan") == 22

    config.pop("generation")
    assert resolve_generation_seed(config, "gan") == 11


def test_validate_generated_assets_requires_images_and_metadata(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        validate_generated_assets(config, "vae")

    image_path = tmp_path / "generated" / "vae" / "child" / "vae_000000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")

    with pytest.raises(FileNotFoundError):
        validate_generated_assets(config, "vae")


def test_should_generate_assets_reuses_valid_existing_data(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    image_path = tmp_path / "generated" / "vae" / "child" / "vae_000000.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image")
    write_csv(
        tmp_path / "metadata" / "vae_metadata.csv",
        [{"image_path": str(image_path), "source": "vae"}],
        ["image_path", "source"],
    )

    assert not should_generate_assets(
        config,
        "vae",
        use_existing=False,
        force_generate=False,
    )
    assert not should_generate_assets(
        config,
        "vae",
        use_existing=True,
        force_generate=False,
    )
    assert should_generate_assets(
        config,
        "vae",
        use_existing=False,
        force_generate=True,
    )


def test_use_existing_fails_when_assets_are_missing(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(FileNotFoundError):
        should_generate_assets(
            config,
            "gan",
            use_existing=True,
            force_generate=False,
        )
