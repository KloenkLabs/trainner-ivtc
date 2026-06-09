import json

import numpy as np
import torch

from trainner_ivtc.dataset import CadenceFrameDataset, OnlineSyntheticCadenceDataset, random_crop_frames
from trainner_ivtc.image_io import save_luma_image


def test_cadence_frame_dataset_reads_png_frames(tmp_path) -> None:
    sample_dir = tmp_path / "train" / "000000"
    sample_dir.mkdir(parents=True)
    frames = []
    for i in range(11):
        rel = f"train/000000/frame_{i:02d}.png"
        save_luma_image(tmp_path / rel, np.full((32, 48), i, dtype=np.uint8))
        frames.append(rel)
    manifest = tmp_path / "train_manifest.jsonl"
    manifest.write_text(json.dumps({"sample_dir": "train/000000", "frames": frames, "field_order": "tff", "label": 2, "frame_index": 0}) + "\n", encoding="utf-8")
    dataset = CadenceFrameDataset(manifest)
    sample = dataset[0]
    assert sample["fields"].shape == (22, 16, 48)
    assert int(sample["label"]) == 2


def online_config(tmp_path):
    frames_dir = tmp_path / "source"
    frames_dir.mkdir()
    for i in range(10):
        save_luma_image(frames_dir / f"{i:04d}.png", np.full((32, 48), i, dtype=np.uint8))
    return {
        "seed": 1234,
        "data": {
            "source_dirs": [str(frames_dir)],
            "field_order": "tff",
            "window_frames": 11,
            "height": 32,
            "width": 48,
            "crop_height": 0,
            "crop_width": 0,
            "crop_modulo": 2,
            "train_samples_pct": 90,
            "val_samples_pct": 10,
            "source_cache_size": 2,
            "resample_train_each_epoch": True,
            "noise_std": 0.0,
            "class_distribution": {"film_phase_0": 1.0},
        },
        "model": {
            "channel_mult": [1, 2, 4, 4],
        },
    }


def test_online_synthetic_dataset_shape_and_split_lengths(tmp_path) -> None:
    config = online_config(tmp_path)
    train_dataset = OnlineSyntheticCadenceDataset(config, "train")
    val_dataset = OnlineSyntheticCadenceDataset(config, "val")
    sample = train_dataset[0]
    assert len(train_dataset) == 9
    assert len(val_dataset) == 1
    assert sample["fields"].shape == (22, 16, 48)
    assert sample["label_map"].shape == (2, 6)
    assert int(sample["label"]) == 0


def test_online_synthetic_dataset_repeats_split_length(tmp_path) -> None:
    config = online_config(tmp_path)
    config["data"]["dataset_repeats"] = 3
    train_dataset = OnlineSyntheticCadenceDataset(config, "train")
    val_dataset = OnlineSyntheticCadenceDataset(config, "val")
    assert len(train_dataset) == 27
    assert len(val_dataset) == 3
    assert not torch.equal(train_dataset[0]["fields"], train_dataset[9]["fields"])


def test_online_synthetic_dataset_resamples_train_but_not_val(tmp_path) -> None:
    config = online_config(tmp_path)
    config["data"]["noise_std"] = 3.0
    train_dataset = OnlineSyntheticCadenceDataset(config, "train")
    val_dataset = OnlineSyntheticCadenceDataset(config, "val")
    train_epoch_0 = train_dataset[0]["fields"]
    train_dataset.set_epoch(1)
    train_epoch_1 = train_dataset[0]["fields"]
    val_epoch_0 = val_dataset[0]["fields"]
    val_dataset.set_epoch(1)
    val_epoch_1 = val_dataset[0]["fields"]
    assert not torch.equal(train_epoch_0, train_epoch_1)
    assert torch.equal(val_epoch_0, val_epoch_1)


def test_random_crop_frames_uses_same_crop_for_window() -> None:
    base = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48).astype(np.float32)
    frames = [(base + i * 10000).astype(np.float32) for i in range(5)]
    cropped = random_crop_frames(frames, 16, 24, 4, np.random.default_rng(1))
    assert len(cropped) == 5
    assert cropped[0].shape == (16, 24)
    for i, frame in enumerate(cropped):
        assert np.array_equal(frame - i * 10000, cropped[0])


def test_online_synthetic_dataset_applies_crop_shape(tmp_path) -> None:
    config = online_config(tmp_path)
    config["data"]["crop_height"] = 16
    config["data"]["crop_width"] = 32
    config["data"]["crop_modulo"] = 4
    dataset = OnlineSyntheticCadenceDataset(config, "train")
    sample = dataset[0]
    assert sample["fields"].shape == (22, 8, 32)
    assert sample["label_map"].shape == (1, 4)
