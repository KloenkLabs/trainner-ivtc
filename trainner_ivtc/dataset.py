from __future__ import annotations

import json
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trainner_ivtc.fields import FieldOrder, frames_to_model_tensor, input_feature_config, validate_field_order
from trainner_ivtc.image_io import load_luma_image
from trainner_ivtc.grid import dense_grid_shape
from trainner_ivtc.data.synthetic import CropBox, RandomCropSpec, SourceFramePool, count_sequence_frames, generate_mixed_cadence_sample, generate_sample_frames, generate_scene_change_telecine_frames, sample_class_index, split_source_sequences


def random_crop_frames(frames: list[np.ndarray], crop_height: int, crop_width: int, crop_modulo: int, rng: np.random.Generator) -> list[np.ndarray]:
    if crop_height <= 0 and crop_width <= 0:
        return frames
    height, width = frames[0].shape
    max_top = height - crop_height
    max_left = width - crop_width
    top_choices = max_top // crop_modulo + 1
    left_choices = max_left // crop_modulo + 1
    top = int(rng.integers(0, top_choices)) * crop_modulo
    left = int(rng.integers(0, left_choices)) * crop_modulo
    bottom = top + crop_height
    right = left + crop_width
    return [frame[top:bottom, left:right] for frame in frames]


def random_crop_box(height: int, width: int, crop_height: int, crop_width: int, crop_modulo: int, rng: np.random.Generator) -> CropBox | None:
    if crop_height <= 0 and crop_width <= 0:
        return None
    max_top = height - crop_height
    max_left = width - crop_width
    top_choices = max_top // crop_modulo + 1
    left_choices = max_left // crop_modulo + 1
    top = int(rng.integers(0, top_choices)) * crop_modulo
    left = int(rng.integers(0, left_choices)) * crop_modulo
    return top, left, crop_height, crop_width


class CadenceFrameDataset(Dataset):
    def __init__(self, manifest_path: str | Path, input_features: dict[str, bool] | None = None) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.input_features = input_features or {}
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")

    def set_epoch(self, epoch: int) -> None:
        _ = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = int(record["label"])
        if "frames" in record:
            frame_paths = [self.root / frame_path for frame_path in record["frames"]]
            frames = [load_luma_image(frame_path) for frame_path in frame_paths]
            field_order = validate_field_order(record.get("field_order", "tff"))
            fields = frames_to_model_tensor(frames, field_order, bool(self.input_features.get("scene_diff", False))).astype(np.float32) / 255.0
            sample_path = self.root / record.get("sample_dir", Path(record["frames"][0]).parent)
        else:
            if bool(self.input_features.get("scene_diff", False)):
                raise ValueError("Scene-diff input cannot be enabled for legacy NPZ samples without frame paths")
            sample_path = self.root / record["sample"]
            with np.load(sample_path) as data:
                fields = data["fields"].astype(np.float32) / 255.0
                label = int(data["label"])
        return {
            "fields": torch.from_numpy(fields),
            "label": torch.tensor(label, dtype=torch.long),
            "frame_index": int(record.get("frame_index", index)),
            "sample_path": str(sample_path),
        }


CadenceNpzDataset = CadenceFrameDataset


class OnlineSyntheticCadenceDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.config = config
        self.split = split
        self.shared_epoch = Value("i", 0)
        data = config["data"]
        train_sequences, val_sequences = split_source_sequences([str(path) for path in data.get("source_dirs", [])], int(data["train_samples_pct"]))
        sequences = train_sequences if split == "train" else val_sequences
        self.base_length = count_sequence_frames(sequences)
        self.dataset_repeats = int(data.get("dataset_repeats", 1))
        self.length = self.base_length * self.dataset_repeats
        if self.base_length <= 0:
            raise ValueError(f"Online synthetic {split} split has no source frames")
        self.height = int(data["height"])
        self.width = int(data["width"])
        self.crop_height = int(data.get("crop_height", 0))
        self.crop_width = int(data.get("crop_width", 0))
        self.crop_modulo = int(data.get("crop_modulo", 2))
        self.sample_height = self.crop_height if self.crop_height > 0 else self.height
        self.sample_width = self.crop_width if self.crop_width > 0 else self.width
        self.window_frames = int(data["window_frames"])
        self.field_order: FieldOrder = validate_field_order(str(data["field_order"]).lower())
        self.augmentations = data.get("augmentations", {})
        self.mixed_cadence = data.get("mixed_cadence", {})
        self.mixed_cadence_chance = float(self.mixed_cadence.get("chance", 0.0))
        self.mixed_boundary_cells = int(self.mixed_cadence.get("boundary_cells", 1))
        self.scene_change = data.get("scene_change", {})
        self.scene_change_chance = float(self.scene_change.get("chance", 0.0))
        self.input_features = input_feature_config(data)
        self.augmentations_enabled = split == "train"
        self.class_distribution = data["class_distribution"]
        self.resample_train_each_epoch = bool(data.get("resample_train_each_epoch", True))
        self.base_seed = int(config.get("seed", 1234)) + (0 if split == "train" else 100000000)
        cache_mode = str(data.get("source_cache_mode", "lru"))
        self.source_pool = SourceFramePool(None, self.height, self.width, sequences, int(data.get("source_cache_size", 256)), cache_mode)
        self.grid_shape = dense_grid_shape(self.sample_height // 2, self.sample_width, tuple(config.get("model", {}).get("channel_mult", [1, 2, 4, 4])))

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        with self.shared_epoch.get_lock():
            self.shared_epoch.value = int(epoch)

    def sample_seed(self, index: int) -> int:
        with self.shared_epoch.get_lock():
            current_epoch = int(self.shared_epoch.value)
        epoch = current_epoch if self.split == "train" and self.resample_train_each_epoch else 0
        return int((self.base_seed + index * 1009 + epoch * 1000003) % (2**31 - 1))

    def __getitem__(self, index: int) -> dict[str, Any]:
        seed = self.sample_seed(index)
        rng = np.random.default_rng(seed)
        if self.crop_height > 0:
            crop_seed = int((seed + 1597463007) % (2**31 - 1))
            source_crop = RandomCropSpec(self.crop_height, self.crop_width, self.crop_modulo, crop_seed)
            sample_height = self.crop_height
            sample_width = self.crop_width
        else:
            source_crop = None
            sample_height = self.height
            sample_width = self.width
        if self.mixed_cadence_chance > 0.0 and rng.random() < self.mixed_cadence_chance:
            mixed = generate_mixed_cadence_sample(rng, sample_height, sample_width, self.field_order, self.source_pool, self.class_distribution, self.grid_shape, self.mixed_boundary_cells, self.augmentations, self.augmentations_enabled, self.window_frames, source_crop)
            frames = mixed.frames
            label = int(mixed.labels[0])
            label_map = mixed.label_map
        else:
            label = sample_class_index(rng, self.class_distribution)
            if label in range(5) and self.scene_change_chance > 0.0 and rng.random() < self.scene_change_chance:
                frames = generate_scene_change_telecine_frames(rng, sample_height, sample_width, label, self.field_order, self.source_pool, self.class_distribution, self.augmentations, self.augmentations_enabled, self.window_frames, source_crop)
            else:
                frames = generate_sample_frames(rng, sample_height, sample_width, self.field_order, label, self.source_pool, self.augmentations, self.augmentations_enabled, self.window_frames, source_crop)
            label_map = np.full(self.grid_shape, label, dtype=np.int64)
        fields = frames_to_model_tensor(frames, self.field_order, bool(self.input_features.get("scene_diff", False)))
        return {
            "fields": torch.from_numpy(fields),
            "label": torch.tensor(label, dtype=torch.long),
            "label_map": torch.from_numpy(label_map),
            "frame_index": int(index),
            "sample_path": f"online:{self.split}:{index}",
        }


def manifest_path(dataset_dir: str | Path, split: str) -> Path:
    return Path(dataset_dir) / f"{split}_manifest.jsonl"
