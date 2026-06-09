from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from trainner_ivtc.labels import FILM_CLASS_INDICES


GRID_IGNORE_INDEX = -100
CONFIDENCE_SCALE = 255


def dense_downsample_steps(channel_mult: tuple[int, ...] | list[int]) -> int:
    return max(len(channel_mult) - 1, 0)


def dense_downsample_factor(channel_mult: tuple[int, ...] | list[int]) -> int:
    return 2 ** dense_downsample_steps(channel_mult)


def downsampled_size(size: int, steps: int) -> int:
    for _ in range(steps):
        size = math.ceil(size / 2)
    return size


def dense_grid_shape(field_height: int, width: int, channel_mult: tuple[int, ...] | list[int]) -> tuple[int, int]:
    steps = dense_downsample_steps(channel_mult)
    return downsampled_size(field_height, steps), downsampled_size(width, steps)


def global_logits_from_dense(dense_logits: Tensor) -> Tensor:
    if dense_logits.ndim != 4:
        raise ValueError(f"Expected dense logits shape [B, C, H, W], got {tuple(dense_logits.shape)}")
    return dense_logits.mean(dim=(2, 3))


def dense_targets_from_labels(labels: Tensor, grid_height: int, grid_width: int) -> Tensor:
    if labels.ndim != 1:
        raise ValueError(f"Expected labels shape [B], got {tuple(labels.shape)}")
    return labels.long().view(-1, 1, 1).expand(-1, grid_height, grid_width)


def encode_probability_map(probabilities: Tensor) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shape [C, H, W], got {tuple(probabilities.shape)}")
    probabilities = probabilities.detach().float().cpu()
    confidence, best_class = torch.max(probabilities, dim=0)
    film_confidence = probabilities[list(FILM_CLASS_INDICES)].sum(dim=0).clamp(0.0, 1.0)
    encoded = torch.empty((probabilities.shape[1], probabilities.shape[2], 3), dtype=torch.uint8)
    encoded[:, :, 0] = best_class.clamp(0, 255).to(torch.uint8)
    encoded[:, :, 1] = confidence.clamp(0.0, 1.0).mul(CONFIDENCE_SCALE).round().to(torch.uint8)
    encoded[:, :, 2] = film_confidence.mul(CONFIDENCE_SCALE).round().to(torch.uint8)
    return encoded.numpy()


def decode_probability_map(encoded: np.ndarray) -> dict[str, np.ndarray]:
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected encoded RGB map shape [H, W, 3], got {tuple(encoded.shape)}")
    encoded = encoded.astype(np.uint8, copy=False)
    return {
        "class_index": encoded[:, :, 0].copy(),
        "confidence": encoded[:, :, 1].astype(np.float32) / CONFIDENCE_SCALE,
        "film_confidence": encoded[:, :, 2].astype(np.float32) / CONFIDENCE_SCALE,
    }


def grid_map_metadata(source_frame_shape: tuple[int, int], grid_shape: tuple[int, int], channel_mult: tuple[int, ...] | list[int], field_order: str, checkpoint_path: str, class_names: tuple[str, ...], class_ids: tuple[str, ...]) -> dict[str, Any]:
    field_downsample = dense_downsample_factor(channel_mult)
    return {
        "format": "trainner_ivtc_grid_rgb_v1",
        "class_names": list(class_names),
        "class_ids": list(class_ids),
        "source_frame_size": {"height": int(source_frame_shape[0]), "width": int(source_frame_shape[1])},
        "grid_size": {"height": int(grid_shape[0]), "width": int(grid_shape[1])},
        "field_downsample": int(field_downsample),
        "source_cell_size": {"height": int(field_downsample * 2), "width": int(field_downsample)},
        "field_order": field_order,
        "checkpoint": checkpoint_path,
        "confidence_scale": CONFIDENCE_SCALE,
        "rgb_channels": {"r": "class_index", "g": "confidence", "b": "film_confidence"},
    }
