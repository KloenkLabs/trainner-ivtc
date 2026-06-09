from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from trainner_ivtc.fields import frames_to_field_tensor, validate_field_order
from trainner_ivtc.image_io import load_luma_image


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


def manifest_path(dataset_dir: str | Path, split: str) -> Path:
    return Path(dataset_dir) / f"{split}_manifest.jsonl"
