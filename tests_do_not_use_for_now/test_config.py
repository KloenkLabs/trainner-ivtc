import pytest
import numpy as np

from trainner_ivtc.config import load_config
from trainner_ivtc.image_io import save_luma_image


def write_config(tmp_path, body: str):
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_config_resolves_native_dimensions(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    save_luma_image(frames / "b.png", np.full((32, 48), 0, dtype=np.uint8))
    config_path = write_config(tmp_path, f"data:\n    source_dirs:\n        - {frames}\n    height: 0\n    width: 0\n")
    config = load_config(config_path)
    assert config["data"]["height"] == 32
    assert config["data"]["width"] == 48


def test_config_rejects_one_zero_dimension(tmp_path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    config_path = write_config(tmp_path, f"data:\n    source_dirs:\n        - {frames}\n    height: 0\n    width: 48\n")
    with pytest.raises(ValueError, match="both be 0"):
        load_config(config_path)


def test_config_rejects_removed_sample_keys(tmp_path) -> None:
    config_path = write_config(tmp_path, "data:\n    train_samples: 10\n")
    with pytest.raises(ValueError, match="train_samples_pct"):
        load_config(config_path)


def test_config_rejects_invalid_percentages(tmp_path) -> None:
    config_path = write_config(tmp_path, "data:\n    train_samples_pct: 80\n    val_samples_pct: 10\n")
    with pytest.raises(ValueError, match="sum to 100"):
        load_config(config_path)


def test_config_rejects_invalid_dataset_repeats(tmp_path) -> None:
    config_path = write_config(tmp_path, "data:\n    dataset_repeats: 0\n")
    with pytest.raises(ValueError, match="dataset_repeats"):
        load_config(config_path)


def test_config_rejects_crop_size_not_divisible_by_modulo(tmp_path) -> None:
    config_path = write_config(tmp_path, "data:\n    height: 32\n    width: 48\n    crop_height: 18\n    crop_width: 32\n    crop_modulo: 4\n")
    with pytest.raises(ValueError, match="crop_modulo"):
        load_config(config_path)


def test_config_rejects_crop_larger_than_source(tmp_path) -> None:
    config_path = write_config(tmp_path, "data:\n    height: 32\n    width: 48\n    crop_height: 40\n    crop_width: 32\n    crop_modulo: 4\n")
    with pytest.raises(ValueError, match="cannot exceed"):
        load_config(config_path)
