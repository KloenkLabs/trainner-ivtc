from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from trainner_ivtc.image_io import iter_image_paths


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1234,
    "paths": {
        "dataset_dir": "datasets/synthetic_global_luma_v1",
        "output_dir": "experiments/global_luma_v1",
    },
    "data": {
        "dataset_mode": "online",
        "source_dirs": [],
        "field_order": "tff",
        "window_frames": 11,
        "height": 192,
        "width": 256,
        "crop_height": 0,
        "crop_width": 0,
        "crop_modulo": 2,
        "num_workers": "auto",
        "train_samples_pct": 90,
        "val_samples_pct": 10,
        "dataset_repeats": 1,
        "source_cache_size": 256,
        "resample_train_each_epoch": True,
        "class_distribution": {
            "film_phase_0": 0.13,
            "film_phase_1": 0.13,
            "film_phase_2": 0.13,
            "film_phase_3": 0.13,
            "film_phase_4": 0.13,
            "video": 0.15,
            "blend": 0.08,
            "scene_cut": 0.06,
            "unknown": 0.06,
        },
        "noise_std": 2.0,
    },
    "model": {
        "base_channels": 32,
        "channel_mult": [1, 2, 4, 4],
        "dropout": 0.1,
    },
    "training": {
        "batch_size": 64,
        "epochs": 8,
        "num_workers": 4,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "amp": True,
        "device": "cuda",
        "print_freq": 25,
    },
    "inference": {
        "batch_size": 16,
        "device": "cuda",
        "field_order": "tff",
        "window_frames": 11,
    },
}


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    return height, width


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def reject_removed_data_keys(loaded: dict[str, Any], config_path: Path) -> None:
    data = loaded.get("data", {})
    if not isinstance(data, dict):
        return
    removed = [key for key in ("train_samples", "val_samples") if key in data]
    if removed:
        raise ValueError(f"{config_path}: data.{removed[0]} was replaced by data.{removed[0]}_pct. Use integer percentages, usually train_samples_pct: 90 and val_samples_pct: 10.")


def resolve_native_dimensions(config: dict[str, Any], config_path: Path) -> None:
    data = config["data"]
    height = int(data["height"])
    width = int(data["width"])
    if (height == 0) != (width == 0):
        raise ValueError(f"{config_path}: data.height and data.width must either both be 0 or both be explicit positive dimensions")
    if height != 0:
        if height < 2 or width < 2:
            raise ValueError(f"{config_path}: data.height and data.width must be positive even dimensions, or both 0 for native source dimensions")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"{config_path}: data.height and data.width must be even")
        return
    source_dirs = [str(path) for path in data.get("source_dirs", [])]
    if not source_dirs:
        raise ValueError(f"{config_path}: data.height and data.width are 0, but data.source_dirs is empty")
    resolved_size: tuple[int, int] | None = None
    for source_dir in source_dirs:
        paths = iter_image_paths(source_dir)
        current_size = image_size(paths[0])
        if resolved_size is None:
            resolved_size = current_size
        elif current_size != resolved_size:
            raise ValueError(f"{config_path}: all source dirs must have matching native dimensions when height/width are 0, got {resolved_size} and {current_size}")
    assert resolved_size is not None
    data["height"] = resolved_size[0]
    data["width"] = resolved_size[1]


def validate_data_config(config: dict[str, Any], config_path: Path) -> None:
    data = config["data"]
    dataset_mode = str(data.get("dataset_mode", "online"))
    if dataset_mode not in {"online", "manifest"}:
        raise ValueError(f"{config_path}: data.dataset_mode must be 'online' or 'manifest', got {dataset_mode!r}")
    data["dataset_mode"] = dataset_mode
    train_pct = int(data.get("train_samples_pct", 90))
    val_pct = int(data.get("val_samples_pct", 10))
    if train_pct < 0 or val_pct < 0 or train_pct + val_pct != 100:
        raise ValueError(f"{config_path}: data.train_samples_pct and data.val_samples_pct must be non-negative integers that sum to 100")
    if train_pct == 0 or val_pct == 0:
        raise ValueError(f"{config_path}: data.train_samples_pct and data.val_samples_pct must both be greater than 0 for train/validation workflows")
    data["train_samples_pct"] = train_pct
    data["val_samples_pct"] = val_pct
    data["dataset_repeats"] = int(data.get("dataset_repeats", 1))
    if data["dataset_repeats"] < 1:
        raise ValueError(f"{config_path}: data.dataset_repeats must be >= 1")
    data["source_cache_size"] = int(data.get("source_cache_size", 256))
    if data["source_cache_size"] < 0:
        raise ValueError(f"{config_path}: data.source_cache_size must be >= 0")
    resolve_native_dimensions(config, config_path)
    crop_height = int(data.get("crop_height", 0))
    crop_width = int(data.get("crop_width", 0))
    crop_modulo = int(data.get("crop_modulo", 2))
    if crop_modulo < 1:
        raise ValueError(f"{config_path}: data.crop_modulo must be >= 1")
    if (crop_height == 0) != (crop_width == 0):
        raise ValueError(f"{config_path}: data.crop_height and data.crop_width must either both be 0 or both be explicit positive dimensions")
    if crop_height < 0 or crop_width < 0:
        raise ValueError(f"{config_path}: data.crop_height and data.crop_width must be >= 0")
    if crop_height > 0:
        if crop_height > int(data["height"]) or crop_width > int(data["width"]):
            raise ValueError(f"{config_path}: crop size {crop_height}x{crop_width} cannot exceed source size {data['height']}x{data['width']}")
        if crop_height % 2 != 0 or crop_width % 2 != 0:
            raise ValueError(f"{config_path}: data.crop_height and data.crop_width must be even")
        if crop_height % crop_modulo != 0 or crop_width % crop_modulo != 0:
            raise ValueError(f"{config_path}: crop bounds must be divisible by data.crop_modulo, so crop_height and crop_width must also be divisible by it")
    data["crop_height"] = crop_height
    data["crop_width"] = crop_width
    data["crop_modulo"] = crop_modulo


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    reject_removed_data_keys(loaded, config_path)
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    validate_data_config(config, config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
