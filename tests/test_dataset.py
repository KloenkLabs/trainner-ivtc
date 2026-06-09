import json

import numpy as np

from trainner_ivtc.dataset import CadenceFrameDataset
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
