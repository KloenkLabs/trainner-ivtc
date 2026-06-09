from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1234,
    "paths": {
        "dataset_dir": "datasets/synthetic_global_luma_v1",
        "output_dir": "experiments/global_luma_v1",
    },
    "data": {
        "source_dirs": [],
        "field_order": "tff",
        "window_frames": 11,
        "height": 192,
        "width": 256,
        "num_workers": "auto",
        "train_samples": 2000,
        "val_samples": 400,
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


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config = deep_update(DEFAULT_CONFIG, loaded)
    config["config_path"] = str(config_path)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
