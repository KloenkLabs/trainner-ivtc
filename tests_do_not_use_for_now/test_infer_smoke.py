import json

import numpy as np
import torch
from PIL import Image

from trainner_ivtc.image_io import save_luma_image
from trainner_ivtc.infer import run_inference
from trainner_ivtc.labels import CLASS_NAMES
from trainner_ivtc.model import GlobalCadenceClassifier


def test_inference_smoke_jsonl(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in range(13):
        frame = np.full((32, 48), i * 10, dtype=np.uint8)
        save_luma_image(frames_dir / f"{i:04d}.png", frame)
    model = GlobalCadenceClassifier(in_channels=22, base_channels=4, channel_mult=(1, 2), dropout=0.0)
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state": model.state_dict(), "model": {"base_channels": 4, "channel_mult": [1, 2], "dropout": 0.0}, "window_frames": 11, "field_order": "tff", "class_names": CLASS_NAMES, "config": {"inference": {"batch_size": 4, "device": "cpu"}}}, checkpoint)
    output = tmp_path / "predictions.jsonl"
    grid_dir = tmp_path / "grid"
    run_inference(checkpoint, frames_dir, output, device_override="cpu", output_mode="both", grid_output_dir=grid_dir)
    lines = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 13
    assert {"idx", "class_id", "class_name", "conf", "film_conf", "video_conf", "probs"} <= set(lines[0])
    assert "recommended_action" not in lines[0]
    maps = sorted((grid_dir / "maps").glob("*.png"))
    metadata = json.loads((grid_dir / "grid_meta.json").read_text(encoding="utf-8"))
    assert len(maps) == 13
    assert metadata["grid_size"] == {"height": 8, "width": 24}
    assert Image.open(maps[0]).size == (24, 8)
