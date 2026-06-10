from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from trainner_ivtc.fields import FieldOrder, clamped_window_indices, frames_to_model_tensor, input_channel_count, input_feature_config, validate_field_order
from trainner_ivtc.grid import cutoff_prevalence_probabilities, encode_hsv_probability_map, encode_probability_map, global_logits_from_dense, grid_map_metadata
from trainner_ivtc.image_io import iter_image_paths, load_luma_image
from trainner_ivtc.labels import CLASS_IDS, CLASS_NAMES, prediction_to_json
from trainner_ivtc.model import build_model, upgrade_legacy_global_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dense luma cadence inference on an image sequence or video.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--input", required=True, help="Directory containing extracted interlaced frames, or an MP4/MKV video.")
    parser.add_argument("--output", required=True, help="Output JSONL path, or the base path used to derive the default grid output directory.")
    parser.add_argument("--field-order", choices=["tff", "bff"], default=None, help="Override checkpoint/config field order.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override inference data loader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override inference data loader prefetch factor.")
    parser.add_argument("--device", default=None, help="Override inference device.")
    parser.add_argument("--output-mode", choices=["jsonl", "grid", "both"], default="jsonl", help="Choose JSONL global predictions, dense grid maps, or both.")
    parser.add_argument("--grid-output-dir", default=None, help="Directory for dense grid maps and grid_meta.json. Defaults to output stem plus _grid.")
    parser.add_argument("--grid-confidence-cutoff", type=float, default=0.0, help="Set grid-map confidence below this 0..1 cutoff to 0. Values at or above the cutoff remain unchanged.")
    parser.add_argument("--json-confidence-cutoff", type=float, default=None, help="Derive JSON global class from the most prevalent per-cell class at or above this 0..1 confidence cutoff.")
    parser.add_argument("--hsv", action="store_true", help="Encode grid-map classes as hue with H=class_id*30, max saturation, and confidence as value.")
    parser.add_argument("--scene-diff", action=argparse.BooleanOptionalAction, default=None, help="Override checkpoint/config scene-diff input feature. Use only with checkpoints trained for the selected channel count.")
    return parser.parse_args()


def resolve_device(name: str | None) -> torch.device:
    resolved = name or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved == "cuda" and not torch.cuda.is_available():
        resolved = "cpu"
    return torch.device(resolved)


def resolve_checkpoint_input_features(checkpoint: dict[str, Any], scene_diff_override: bool | None = None) -> dict[str, bool]:
    if scene_diff_override is not None:
        return {"scene_diff": bool(scene_diff_override)}
    features = checkpoint.get("input_features")
    if features is None:
        features = checkpoint.get("config", {}).get("data", {}).get("input_features", {})
    return input_feature_config({"input_features": features})


def load_checkpoint_model(checkpoint_path: str | Path, device: torch.device, scene_diff_override: bool | None = None) -> tuple[torch.nn.Module, dict[str, Any], dict[str, bool]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model", {})
    window_frames = int(checkpoint.get("window_frames", 11))
    input_features = resolve_checkpoint_input_features(checkpoint, scene_diff_override)
    model = build_model(model_config, in_channels=input_channel_count(window_frames, input_features)).to(device)
    model.load_state_dict(upgrade_legacy_global_state_dict(checkpoint["model_state"]))
    model.eval()
    return model, checkpoint, input_features


def make_window_tensor(load_frame: Callable[[int], np.ndarray], total_frames: int, center: int, window_frames: int, field_order: FieldOrder, input_features: dict[str, bool] | None = None) -> np.ndarray:
    indices = clamped_window_indices(center, total_frames, window_frames)
    frames = [load_frame(index) for index in indices]
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("All inference frames must have the same dimensions")
    return frames_to_model_tensor(frames, field_order, bool((input_features or {}).get("scene_diff", False))).astype(np.float32) / 255.0


class InferenceFrameDataset(Dataset):
    def __init__(self, paths: list[Path], window_frames: int, field_order: FieldOrder, cache_size: int, input_features: dict[str, bool] | None = None) -> None:
        self.paths = paths
        self.window_frames = window_frames
        self.field_order: FieldOrder = field_order
        self.cache_size = max(0, cache_size)
        self.input_features = input_features or {}
        self.frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def load_cached(self, index: int) -> np.ndarray:
        if self.cache_size == 0:
            return load_luma_image(self.paths[index])
        cached = self.frame_cache.get(index)
        if cached is not None:
            self.frame_cache.move_to_end(index)
            return cached
        frame = load_luma_image(self.paths[index])
        self.frame_cache[index] = frame
        if len(self.frame_cache) > self.cache_size:
            self.frame_cache.popitem(last=False)
        return frame

    def __getitem__(self, index: int) -> dict[str, Any]:
        fields = make_window_tensor(self.load_cached, len(self.paths), index, self.window_frames, self.field_order, self.input_features)
        return {
            "fields": torch.from_numpy(fields),
            "frame_index": int(index),
        }


def make_inference_loader(dataset: Dataset, batch_size: int, num_workers: int, prefetch_factor: int, device: torch.device) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def is_numbered_png_frame(path: Path) -> bool:
    return path.is_file() and path.suffix == ".png" and len(path.stem) == 8 and path.stem.isdigit() and path.stat().st_size > 0


def valid_numbered_png_frames(folder: Path) -> list[Path]:
    paths = [path for path in folder.iterdir() if is_numbered_png_frame(path)]
    paths.sort(key=lambda path: path.name)
    return paths


def video_frames_dir(video_path: Path) -> Path:
    return video_path.with_name(f"{video_path.stem}_frames")


def resolve_input_paths(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_dir():
        return iter_image_paths(path)
    if path.suffix.lower() not in {".mp4", ".mkv"}:
        raise ValueError(f"Inference input must be an image directory or an MP4/MKV video: {path}")
    output_dir = video_frames_dir(path)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Video frame output path already exists and is not a directory: {output_dir}")
    if output_dir.exists() and any(child.is_file() for child in output_dir.iterdir()):
        print(f"WARNING: skipping frame extraction because output directory already exists and contains files: {output_dir}", file=sys.stderr, flush=True)
        paths = valid_numbered_png_frames(output_dir)
        if len(paths) < 2:
            raise SystemExit(f"No usable extracted frames found in {output_dir}; expected at least two non-empty files named like 00000001.png")
        return paths
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "%08d.png"
    subprocess.run(["ffmpeg", "-loglevel", "warning", "-stats", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", str(output_pattern)], check=True)
    paths = valid_numbered_png_frames(output_dir)
    if len(paths) < 2:
        raise SystemExit(f"No usable extracted frames found in {output_dir}; expected at least two non-empty files named like 00000001.png")
    return paths


def default_grid_output_dir(output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix:
        return path.with_name(f"{path.stem}_grid")
    return path


def run_inference(
    checkpoint_path: str | Path,
    input_dir: str | Path,
    output_path: str | Path,
    field_order_override: str | None = None,
    batch_size_override: int | None = None,
    device_override: str | None = None,
    num_workers_override: int | None = None,
    prefetch_factor_override: int | None = None,
    output_mode: str = "jsonl",
    grid_output_dir: str | Path | None = None,
    grid_confidence_cutoff: float = 0.0,
    json_confidence_cutoff: float | None = None,
    hsv: bool = False,
    scene_diff_override: bool | None = None,
) -> None:
    start_time = time.perf_counter()
    device = resolve_device(device_override)
    model, checkpoint, input_features = load_checkpoint_model(checkpoint_path, device, scene_diff_override)
    config = checkpoint.get("config", {})
    inference_config = config.get("inference", {})
    window_frames = int(checkpoint.get("window_frames", inference_config.get("window_frames", 11)))
    field_order_raw = str(field_order_override or checkpoint.get("field_order") or inference_config.get("field_order", "tff")).lower()
    field_order: FieldOrder = validate_field_order(field_order_raw)
    batch_size = int(batch_size_override if batch_size_override is not None else inference_config.get("batch_size", 16))
    num_workers = int(num_workers_override if num_workers_override is not None else inference_config.get("num_workers", 8))
    prefetch_factor = int(prefetch_factor_override if prefetch_factor_override is not None else inference_config.get("prefetch_factor", 2))
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")
    if prefetch_factor < 1:
        raise ValueError(f"prefetch_factor must be >= 1, got {prefetch_factor}")
    if output_mode not in {"jsonl", "grid", "both"}:
        raise ValueError(f"output_mode must be 'jsonl', 'grid', or 'both', got {output_mode!r}")
    if grid_confidence_cutoff < 0.0 or grid_confidence_cutoff > 1.0:
        raise ValueError(f"grid_confidence_cutoff must be between 0.0 and 1.0, got {grid_confidence_cutoff}")
    if json_confidence_cutoff is not None and (json_confidence_cutoff < 0.0 or json_confidence_cutoff > 1.0):
        raise ValueError(f"json_confidence_cutoff must be between 0.0 and 1.0, got {json_confidence_cutoff}")
    write_jsonl = output_mode in {"jsonl", "both"}
    write_grid = output_mode in {"grid", "both"}
    paths = resolve_input_paths(input_dir)
    source_frame_shape = load_luma_image(paths[0]).shape
    total_batches = (len(paths) + batch_size - 1) // batch_size
    progress_freq = max(1, total_batches // 20)
    cache_size = max(256, window_frames * batch_size * 2)
    dataset = InferenceFrameDataset(paths, window_frames, field_order, cache_size, input_features)
    loader = make_inference_loader(dataset, batch_size, num_workers, prefetch_factor, device)
    out_path = Path(output_path)
    grid_dir = Path(grid_output_dir) if grid_output_dir is not None else default_grid_output_dir(out_path)
    maps_dir = grid_dir / "maps"
    if write_jsonl:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if write_grid:
        maps_dir.mkdir(parents=True, exist_ok=True)
    print(f"Starting inference: frames={len(paths)} batches={total_batches} batch_size={batch_size} num_workers={num_workers} prefetch_factor={prefetch_factor} device={device} field_order={field_order} window_frames={window_frames} input_features={json.dumps(input_features, separators=(',', ':'))} output_mode={output_mode} grid_confidence_cutoff={grid_confidence_cutoff} json_confidence_cutoff={json_confidence_cutoff} hsv={hsv}", flush=True)
    jsonl_file = out_path.open("w", encoding="utf-8") if write_jsonl else None
    first_grid_shape: tuple[int, int] | None = None
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, start=1):
                tensor = batch["fields"].to(device, non_blocking=True)
                dense_logits = model(tensor)
                global_probabilities = torch.softmax(global_logits_from_dense(dense_logits).float(), dim=1).cpu()
                dense_probabilities = torch.softmax(dense_logits.float(), dim=1).cpu() if write_grid or json_confidence_cutoff is not None else None
                if first_grid_shape is None:
                    first_grid_shape = int(dense_logits.shape[-2]), int(dense_logits.shape[-1])
                frame_indices = [int(frame_index) for frame_index in batch["frame_index"].tolist()]
                for offset, (frame_index, probs) in enumerate(zip(frame_indices, global_probabilities, strict=False)):
                    if jsonl_file is not None:
                        if json_confidence_cutoff is not None:
                            assert dense_probabilities is not None
                            probs = cutoff_prevalence_probabilities(dense_probabilities[offset], json_confidence_cutoff)
                        jsonl_file.write(json.dumps(prediction_to_json(frame_index, probs), separators=(",", ":")) + "\n")
                    if write_grid and dense_probabilities is not None:
                        encoded = encode_hsv_probability_map(dense_probabilities[offset], grid_confidence_cutoff) if hsv else encode_probability_map(dense_probabilities[offset], grid_confidence_cutoff)
                        Image.fromarray(encoded, mode="RGB").save(maps_dir / f"{frame_index:08d}.png")
                if batch_index == 1 or batch_index == total_batches or batch_index % progress_freq == 0:
                    processed = min(batch_index * batch_size, len(paths))
                    elapsed = time.perf_counter() - start_time
                    fps = processed / elapsed if elapsed > 0 else 0.0
                    print(f"inference batch={batch_index}/{total_batches} frames={processed}/{len(paths)} fps={fps:.2f}", flush=True)
    finally:
        if jsonl_file is not None:
            jsonl_file.close()
    if write_grid:
        assert first_grid_shape is not None
        model_config = checkpoint.get("model", {})
        metadata = grid_map_metadata(source_frame_shape, first_grid_shape, tuple(model_config.get("channel_mult", [1, 2, 4, 4])), field_order, str(checkpoint_path), CLASS_NAMES, CLASS_IDS, hsv)
        metadata["frames"] = len(paths)
        metadata["maps_dir"] = "maps"
        metadata["confidence_cutoff"] = float(grid_confidence_cutoff)
        metadata["confidence_cutoff_encoded"] = int(round(grid_confidence_cutoff * 255.0))
        metadata["json_confidence_cutoff"] = float(json_confidence_cutoff) if json_confidence_cutoff is not None else None
        metadata["input_features"] = input_features
        (grid_dir / "grid_meta.json").write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    elapsed = time.perf_counter() - start_time
    fps = len(paths) / elapsed if elapsed > 0 else 0.0
    outputs = []
    if write_jsonl:
        outputs.append(str(out_path))
    if write_grid:
        outputs.append(str(grid_dir))
    print(f"Inference complete: frames={len(paths)} elapsed={elapsed:.2f}s fps={fps:.2f} output={', '.join(outputs)}", flush=True)


def main() -> None:
    args = parse_args()
    run_inference(args.checkpoint, args.input, args.output, args.field_order, args.batch_size, args.device, args.num_workers, args.prefetch_factor, args.output_mode, args.grid_output_dir, args.grid_confidence_cutoff, args.json_confidence_cutoff, args.hsv, args.scene_diff)


if __name__ == "__main__":
    main()
