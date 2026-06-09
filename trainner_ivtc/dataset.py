from __future__ import annotations

import json
from multiprocessing import Value
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trainner_ivtc.fields import frames_to_field_tensor, validate_field_order
from trainner_ivtc.image_io import load_luma_image
from trainner_ivtc.data.synthetic import SourceFramePool, count_sequence_frames, generate_sample_frames, sample_class_index, split_source_sequences


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


class CadenceFrameDataset(Dataset):
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f if line.strip()]
        if not self.records:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        label = int(record["label"])
        if "frames" in record:
            frame_paths = [self.root / frame_path for frame_path in record["frames"]]
            frames = [load_luma_image(frame_path) for frame_path in frame_paths]
            field_order = validate_field_order(record.get("field_order", "tff"))
            fields = frames_to_field_tensor(frames, field_order).astype(np.float32) / 255.0
            sample_path = self.root / record.get("sample_dir", Path(record["frames"][0]).parent)
        else:
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
        self.window_frames = int(data["window_frames"])
        self.field_order = validate_field_order(str(data["field_order"]).lower())
        self.noise_std = float(data.get("noise_std", 0.0))
        self.class_distribution = data["class_distribution"]
        self.resample_train_each_epoch = bool(data.get("resample_train_each_epoch", True))
        self.base_seed = int(config.get("seed", 1234)) + (0 if split == "train" else 100000000)
        self.source_pool = SourceFramePool(None, self.height, self.width, sequences, int(data.get("source_cache_size", 256)))

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
        crop_rng = np.random.default_rng((seed + 1597463007) % (2**31 - 1))
        label = sample_class_index(rng, self.class_distribution)
        frames = generate_sample_frames(rng, self.height, self.width, self.field_order, label, self.source_pool, self.noise_std, self.window_frames)
        frames = random_crop_frames(frames, self.crop_height, self.crop_width, self.crop_modulo, crop_rng)
        fields = frames_to_field_tensor(frames, self.field_order).astype(np.float32) / 255.0
        return {
            "fields": torch.from_numpy(fields),
            "label": torch.tensor(label, dtype=torch.long),
            "frame_index": int(index),
            "sample_path": f"online:{self.split}:{index}",
        }


def manifest_path(dataset_dir: str | Path, split: str) -> Path:
    return Path(dataset_dir) / f"{split}_manifest.jsonl"
