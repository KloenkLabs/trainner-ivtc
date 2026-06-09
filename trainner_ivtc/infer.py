from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from trainner_ivtc.fields import FieldOrder, clamped_window_indices, frames_to_field_tensor, validate_field_order
from trainner_ivtc.image_io import iter_image_paths, load_luma_image
from trainner_ivtc.labels import prediction_to_json
from trainner_ivtc.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global luma cadence inference on an image sequence.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--input", required=True, help="Directory containing extracted interlaced frames.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--field-order", choices=["tff", "bff"], default=None, help="Override checkpoint/config field order.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--device", default=None, help="Override inference device.")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    resolved = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved == "cuda" and not torch.cuda.is_available():
        resolved = "cpu"
    return torch.device(resolved)


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    window_frames = int(checkpoint.get("window_frames", 11))
    model = build_model(model_config, in_channels=window_frames * 2).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def make_window_tensor(load_frame: Callable[[int], np.ndarray], total_frames: int, center: int, window_frames: int, field_order: FieldOrder) -> np.ndarray:
    indices = clamped_window_indices(center, total_frames, window_frames)
    frames = [load_frame(index) for index in indices]
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("All inference frames must have the same dimensions")
    return frames_to_field_tensor(frames, field_order).astype(np.float32) / 255.0


def run_inference(checkpoint_path: str | Path, input_dir: str | Path, output_path: str | Path, field_order_override: str | None = None, batch_size_override: int | None = None, device_override: str | None = None) -> None:
    device = resolve_device(device_override)
    model, checkpoint = load_checkpoint_model(checkpoint_path, device)
    config = checkpoint.get("config", {})
    inference_config = config.get("inference", {})
    window_frames = int(checkpoint.get("window_frames", inference_config.get("window_frames", 11)))
    field_order = validate_field_order((field_order_override or checkpoint.get("field_order") or inference_config.get("field_order", "tff")).lower())
    batch_size = int(batch_size_override or inference_config.get("batch_size", 16))
    paths = iter_image_paths(input_dir)

    @lru_cache(maxsize=max(256, window_frames * batch_size * 2))
    def load_cached(index: int) -> np.ndarray:
        return load_luma_image(paths[index])
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f, torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            frame_indices = list(range(start, min(start + batch_size, len(paths))))
            batch = np.stack([make_window_tensor(load_cached, len(paths), frame_index, window_frames, field_order) for frame_index in frame_indices], axis=0)
            tensor = torch.from_numpy(batch).to(device, non_blocking=True)
            logits = model(tensor)
            probabilities = torch.softmax(logits.float(), dim=1).cpu()
            for frame_index, probs in zip(frame_indices, probabilities, strict=False):
                f.write(json.dumps(prediction_to_json(frame_index, probs), separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    run_inference(args.checkpoint, args.input, args.output, args.field_order, args.batch_size, args.device)


if __name__ == "__main__":
    main()
