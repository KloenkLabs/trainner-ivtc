from __future__ import annotations

import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from trainner_ivtc.fields import FieldOrder, frames_to_field_tensor, telecine_pair_for_frame, validate_field_order, validate_window_frames, weave_field_pair
from trainner_ivtc.image_io import iter_image_paths, load_luma_image, save_luma_image
from trainner_ivtc.labels import CLASS_NAMES, CLASS_TO_INDEX, class_id


@dataclass
class ProceduralClip:
    texture: np.ndarray
    velocity_x: float
    velocity_y: float
    rectangles: list[tuple[float, float, float, float, float, float, float]]

    def frame(self, t: float) -> np.ndarray:
        shifted = np.roll(self.texture, (int(round(t * self.velocity_y)), int(round(t * self.velocity_x))), axis=(0, 1)).copy()
        h, w = shifted.shape
        for x0, y0, rw, rh, value, vx, vy in self.rectangles:
            x = int(round((x0 + t * vx) % max(w, 1)))
            y = int(round((y0 + t * vy) % max(h, 1)))
            x1 = min(x + int(round(rw)), w)
            y1 = min(y + int(round(rh)), h)
            shifted[y:y1, x:x1] = value
        return np.clip(np.rint(shifted), 0, 255).astype(np.uint8)


class SourceFramePool:
    def __init__(self, source_dirs: list[str], height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.sequences = []
        for path in source_dirs:
            if Path(path).exists():
                try:
                    self.sequences.append(iter_image_paths(path))
                except FileNotFoundError:
                    continue
        self.sequences = [sequence for sequence in self.sequences if sequence]

    @property
    def available(self) -> bool:
        return bool(self.sequences)

    def sample_frames(self, rng: np.random.Generator, count: int) -> list[np.ndarray]:
        sequence = self.sequences[int(rng.integers(0, len(self.sequences)))]
        start_max = max(len(sequence) - count, 0)
        start = int(rng.integers(0, start_max + 1)) if start_max > 0 else 0
        return [load_luma_image(sequence[(start + i) % len(sequence)], size=(self.height, self.width)) for i in range(count)]


def make_procedural_clip(rng: np.random.Generator, height: int, width: int, motion: bool = True) -> ProceduralClip:
    small_h = max(height // 16, 4)
    small_w = max(width // 16, 4)
    small = rng.integers(32, 224, size=(small_h, small_w), dtype=np.uint8)
    texture = np.asarray(Image.fromarray(small, mode="L").resize((width, height), Image.Resampling.BICUBIC), dtype=np.float32)
    vx = float(rng.uniform(-2.0, 2.0)) if motion else 0.0
    vy = float(rng.uniform(-1.0, 1.0)) if motion else 0.0
    rectangles: list[tuple[float, float, float, float, float, float, float]] = []
    for _ in range(int(rng.integers(1, 4)) if motion else 0):
        rw = float(rng.uniform(width * 0.08, width * 0.25))
        rh = float(rng.uniform(height * 0.08, height * 0.25))
        rectangles.append((float(rng.uniform(0, width)), float(rng.uniform(0, height)), rw, rh, float(rng.uniform(32, 224)), float(rng.uniform(-2.5, 2.5)), float(rng.uniform(-1.5, 1.5))))
    return ProceduralClip(texture=texture, velocity_x=vx, velocity_y=vy, rectangles=rectangles)


def add_noise(rng: np.random.Generator, fields: np.ndarray, noise_std: float) -> np.ndarray:
    if noise_std <= 0:
        return fields.astype(np.uint8)
    noise = rng.normal(0.0, noise_std, size=fields.shape)
    return np.clip(np.rint(fields.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def generate_telecine_frames(rng: np.random.Generator, height: int, width: int, target_phase: int, field_order: FieldOrder, source_pool: SourceFramePool | None, window_frames: int = 11) -> list[np.ndarray]:
    window_frames = validate_window_frames(window_frames)
    radius = window_frames // 2
    target_video_index = 10 + target_phase
    video_indices = list(range(target_video_index - radius, target_video_index + radius + 1))
    pairs = [telecine_pair_for_frame(index) for index in video_indices]
    min_film = min(min(pair) for pair in pairs)
    max_film = max(max(pair) for pair in pairs)
    count = max_film - min_film + 1
    if source_pool is not None and source_pool.available:
        progressive_frames = source_pool.sample_frames(rng, count)
    else:
        clip = make_procedural_clip(rng, height, width, motion=True)
        progressive_frames = [clip.frame(i) for i in range(count)]
    frames = []
    for first_index, second_index in pairs:
        first = progressive_frames[first_index - min_film]
        second = progressive_frames[second_index - min_film]
        frames.append(weave_field_pair(first, second, field_order))
    return frames


def generate_telecine_window(rng: np.random.Generator, height: int, width: int, target_phase: int, field_order: FieldOrder, source_pool: SourceFramePool | None, window_frames: int = 11) -> np.ndarray:
    frames = generate_telecine_frames(rng, height, width, target_phase, field_order, source_pool, window_frames)
    return frames_to_field_tensor(frames, field_order)


def generate_video_frames(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> list[np.ndarray]:
    window_frames = validate_window_frames(window_frames)
    clip = make_procedural_clip(rng, height, width, motion=True)
    return [weave_field_pair(clip.frame(i * 2), clip.frame(i * 2 + 1), field_order) for i in range(window_frames)]


def generate_video_window(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> np.ndarray:
    frames = generate_video_frames(rng, height, width, field_order, window_frames)
    return frames_to_field_tensor(frames, field_order)


def generate_blend_frames(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, source_pool: SourceFramePool | None, window_frames: int = 11) -> list[np.ndarray]:
    first = generate_telecine_frames(rng, height, width, int(rng.integers(0, 5)), field_order, source_pool, window_frames)
    second = generate_telecine_frames(rng, height, width, int(rng.integers(0, 5)), field_order, source_pool, window_frames)
    alpha = float(rng.uniform(0.35, 0.65))
    return [np.clip(np.rint(a.astype(np.float32) * alpha + b.astype(np.float32) * (1.0 - alpha)), 0, 255).astype(np.uint8) for a, b in zip(first, second, strict=True)]


def generate_blend_window(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, source_pool: SourceFramePool | None, window_frames: int = 11) -> np.ndarray:
    frames = generate_blend_frames(rng, height, width, field_order, source_pool, window_frames)
    return frames_to_field_tensor(frames, field_order)


def generate_scene_cut_frames(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> list[np.ndarray]:
    window_frames = validate_window_frames(window_frames)
    first_clip = make_procedural_clip(rng, height, width, motion=True)
    second_clip = make_procedural_clip(rng, height, width, motion=True)
    cut_at = int(rng.integers(max(1, window_frames // 3), max(2, window_frames - window_frames // 3)))
    frames = []
    for i in range(window_frames):
        clip = first_clip if i < cut_at else second_clip
        frames.append(weave_field_pair(clip.frame(i * 2), clip.frame(i * 2 + 1), field_order))
    return frames


def generate_scene_cut_window(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> np.ndarray:
    frames = generate_scene_cut_frames(rng, height, width, field_order, window_frames)
    return frames_to_field_tensor(frames, field_order)


def generate_unknown_frames(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> list[np.ndarray]:
    window_frames = validate_window_frames(window_frames)
    clip = make_procedural_clip(rng, height, width, motion=False)
    frame = clip.frame(0)
    return [weave_field_pair(frame, frame, field_order) for _ in range(window_frames)]


def generate_unknown_window(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, window_frames: int = 11) -> np.ndarray:
    frames = generate_unknown_frames(rng, height, width, field_order, window_frames)
    return frames_to_field_tensor(frames, field_order)


def sample_class_index(rng: np.random.Generator, distribution: dict[str, float]) -> int:
    names = list(distribution.keys())
    unknown_names = [name for name in names if name not in CLASS_TO_INDEX]
    if unknown_names:
        raise KeyError(f"Unknown class names in class_distribution: {unknown_names}")
    weights = np.asarray([float(distribution[name]) for name in names], dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("class_distribution weights must be non-negative and sum to a positive value")
    weights = weights / weights.sum()
    name = names[int(rng.choice(len(names), p=weights))]
    return CLASS_TO_INDEX[name]


def resolve_worker_count(value: Any = "auto") -> int:
    if value is None or value == "auto":
        return max(os.cpu_count() or 1, 1)
    workers = int(value)
    if workers < 1:
        raise ValueError(f"num_workers must be >= 1 or 'auto', got {value!r}")
    return workers


def generate_sample_frames(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, label: int, source_pool: SourceFramePool | None, noise_std: float, window_frames: int = 11) -> list[np.ndarray]:
    if label in range(5):
        frames = generate_telecine_frames(rng, height, width, label, field_order, source_pool, window_frames)
    elif CLASS_NAMES[label] == "video":
        frames = generate_video_frames(rng, height, width, field_order, window_frames)
    elif CLASS_NAMES[label] == "blend":
        frames = generate_blend_frames(rng, height, width, field_order, source_pool, window_frames)
    elif CLASS_NAMES[label] == "scene_cut":
        frames = generate_scene_cut_frames(rng, height, width, field_order, window_frames)
    else:
        frames = generate_unknown_frames(rng, height, width, field_order, window_frames)
    return [add_noise(rng, frame, noise_std) for frame in frames]


def generate_sample(rng: np.random.Generator, height: int, width: int, field_order: FieldOrder, label: int, source_pool: SourceFramePool | None, noise_std: float, window_frames: int = 11) -> np.ndarray:
    frames = generate_sample_frames(rng, height, width, field_order, label, source_pool, noise_std, window_frames)
    return frames_to_field_tensor(frames, field_order)


def write_sample(dataset_dir: Path, split: str, index: int, seed: int, label: int, height: int, width: int, field_order: FieldOrder, source_pool: SourceFramePool, noise_std: float, window_frames: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    frames = generate_sample_frames(rng, height, width, field_order, label, source_pool, noise_std, window_frames)
    sample_dir_rel = f"{split}/{index:06d}"
    frame_paths = []
    for frame_index, frame in enumerate(frames):
        frame_rel = f"{sample_dir_rel}/frame_{frame_index:02d}.png"
        save_luma_image(dataset_dir / frame_rel, frame)
        frame_paths.append(frame_rel)
    return {"sample_dir": sample_dir_rel, "frames": frame_paths, "field_order": field_order, "label": int(label), "class_id": class_id(label), "class_name": CLASS_NAMES[label], "frame_index": index}


def write_split(config: dict[str, Any], split: str, count: int, rng: np.random.Generator, source_pool: SourceFramePool, overwrite: bool, num_workers: int) -> None:
    dataset_dir = Path(config["paths"]["dataset_dir"])
    split_dir = dataset_dir / split
    manifest_path = dataset_dir / f"{split}_manifest.jsonl"
    if split_dir.exists() and any(split_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{split_dir} already contains samples. Pass --overwrite to replace matching files.")
    if split_dir.exists() and overwrite:
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    height = int(config["data"]["height"])
    width = int(config["data"]["width"])
    window_frames = validate_window_frames(int(config["data"]["window_frames"]))
    field_order = validate_field_order(str(config["data"]["field_order"]).lower())
    distribution = config["data"]["class_distribution"]
    noise_std = float(config["data"].get("noise_std", 0.0))
    labels = [sample_class_index(rng, distribution) for _ in range(count)]
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(count)]
    with manifest_path.open("w", encoding="utf-8") as manifest:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            records = executor.map(
                lambda args: write_sample(dataset_dir, split, args[0], args[1], args[2], height, width, field_order, source_pool, noise_std, window_frames),
                zip(range(count), seeds, labels, strict=True),
            )
            for record in records:
                manifest.write(json.dumps(record, separators=(",", ":")) + "\n")


def make_synthetic_dataset(config: dict[str, Any], overwrite: bool = False, num_workers: int | None = None) -> None:
    resolved_workers = resolve_worker_count(num_workers if num_workers is not None else config["data"].get("num_workers", "auto"))
    print(f"Using {resolved_workers} synthetic data worker threads.")
    dataset_dir = Path(config["paths"]["dataset_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    data_config = config["data"]
    height = int(data_config["height"])
    width = int(data_config["width"])
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError("Synthetic height and width must be even")
    source_pool = SourceFramePool([str(path) for path in data_config.get("source_dirs", [])], height, width)
    root_rng = np.random.default_rng(int(config.get("seed", 1234)))
    write_split(config, "train", int(data_config["train_samples"]), np.random.default_rng(int(root_rng.integers(0, 2**31 - 1))), source_pool, overwrite, resolved_workers)
    write_split(config, "val", int(data_config["val_samples"]), np.random.default_rng(int(root_rng.integers(0, 2**31 - 1))), source_pool, overwrite, resolved_workers)
    classes_path = dataset_dir / "classes.json"
    classes_path.write_text(json.dumps({"class_names": CLASS_NAMES, "class_ids": [class_id(i) for i in range(len(CLASS_NAMES))]}, indent=4), encoding="utf-8")
