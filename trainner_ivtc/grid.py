from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

from trainner_ivtc.labels import FILM_CLASS_INDICES


GRID_IGNORE_INDEX = -100
CONFIDENCE_SCALE = 255
CLASS_INDEX_SCALE = 20
HSV_HUE_STEP_DEGREES = 30


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


def encode_probability_map(probabilities: Tensor, confidence_cutoff: float = 0.0) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shape [C, H, W], got {tuple(probabilities.shape)}")
    if confidence_cutoff < 0.0 or confidence_cutoff > 1.0:
        raise ValueError(f"confidence_cutoff must be between 0.0 and 1.0, got {confidence_cutoff}")
    probabilities = probabilities.detach().float().cpu()
    confidence, best_class = torch.max(probabilities, dim=0)
    film_confidence = probabilities[list(FILM_CLASS_INDICES)].sum(dim=0).clamp(0.0, 1.0)
    encoded = torch.empty((probabilities.shape[1], probabilities.shape[2], 3), dtype=torch.uint8)
    encoded[:, :, 0] = best_class.mul(CLASS_INDEX_SCALE).clamp(0, 255).to(torch.uint8)
    encoded[:, :, 1] = confidence.clamp(0.0, 1.0).mul(CONFIDENCE_SCALE).round().to(torch.uint8)
    encoded[:, :, 2] = film_confidence.mul(CONFIDENCE_SCALE).round().to(torch.uint8)
    if confidence_cutoff > 0.0:
        low_confidence = confidence < confidence_cutoff
        encoded[:, :, 1][low_confidence] = 0
        encoded[:, :, 2][low_confidence] = 0
    return encoded.numpy()


def encode_hsv_probability_map(probabilities: Tensor, confidence_cutoff: float = 0.0) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shape [C, H, W], got {tuple(probabilities.shape)}")
    if confidence_cutoff < 0.0 or confidence_cutoff > 1.0:
        raise ValueError(f"confidence_cutoff must be between 0.0 and 1.0, got {confidence_cutoff}")
    probabilities = probabilities.detach().float().cpu()
    confidence, best_class = torch.max(probabilities, dim=0)
    value = confidence.clamp(0.0, 1.0)
    if confidence_cutoff > 0.0:
        value = value.clone()
        value[confidence < confidence_cutoff] = 0.0
    hue = best_class.float().mul(float(HSV_HUE_STEP_DEGREES)).remainder(360.0)
    hue_sector = torch.floor(hue / 60.0).long()
    x = value * (1.0 - torch.abs(torch.remainder(hue / 60.0, 2.0) - 1.0))
    red = torch.zeros_like(value)
    green = torch.zeros_like(value)
    blue = torch.zeros_like(value)
    mask = hue_sector == 0
    red[mask] = value[mask]
    green[mask] = x[mask]
    mask = hue_sector == 1
    red[mask] = x[mask]
    green[mask] = value[mask]
    mask = hue_sector == 2
    green[mask] = value[mask]
    blue[mask] = x[mask]
    mask = hue_sector == 3
    green[mask] = x[mask]
    blue[mask] = value[mask]
    mask = hue_sector == 4
    red[mask] = x[mask]
    blue[mask] = value[mask]
    mask = hue_sector == 5
    red[mask] = value[mask]
    blue[mask] = x[mask]
    return torch.stack((red, green, blue), dim=2).mul(CONFIDENCE_SCALE).round().to(torch.uint8).numpy()


def cutoff_prevalence_probabilities(probabilities: Tensor, confidence_cutoff: float) -> Tensor:
    if probabilities.ndim != 3:
        raise ValueError(f"Expected probabilities shape [C, H, W], got {tuple(probabilities.shape)}")
    if confidence_cutoff < 0.0 or confidence_cutoff > 1.0:
        raise ValueError(f"confidence_cutoff must be between 0.0 and 1.0, got {confidence_cutoff}")
    probabilities = probabilities.detach().float().cpu()
    confidence, best_class = torch.max(probabilities, dim=0)
    valid = confidence >= confidence_cutoff
    counts = torch.zeros(probabilities.shape[0], dtype=torch.float32)
    if bool(valid.any()):
        counts.scatter_add_(0, best_class[valid].reshape(-1), torch.ones(int(valid.sum().item()), dtype=torch.float32))
    return counts / float(confidence.numel())


def decode_probability_map(encoded: np.ndarray) -> dict[str, np.ndarray]:
    if encoded.ndim != 3 or encoded.shape[2] != 3:
        raise ValueError(f"Expected encoded RGB map shape [H, W, 3], got {tuple(encoded.shape)}")
    encoded = encoded.astype(np.uint8, copy=False)
    return {
        "class_index": np.rint(encoded[:, :, 0].astype(np.float32) / CLASS_INDEX_SCALE).astype(np.uint8),
        "confidence": encoded[:, :, 1].astype(np.float32) / CONFIDENCE_SCALE,
        "film_confidence": encoded[:, :, 2].astype(np.float32) / CONFIDENCE_SCALE,
    }


def grid_map_metadata(source_frame_shape: tuple[int, int], grid_shape: tuple[int, int], channel_mult: tuple[int, ...] | list[int], field_order: str, checkpoint_path: str, class_names: tuple[str, ...], class_ids: tuple[str, ...], hsv: bool = False) -> dict[str, Any]:
    field_downsample = dense_downsample_factor(channel_mult)
    metadata = {
        "format": "trainner_ivtc_grid_hsv_v1" if hsv else "trainner_ivtc_grid_rgb_v2",
        "class_names": list(class_names),
        "class_ids": list(class_ids),
        "source_frame_size": {"height": int(source_frame_shape[0]), "width": int(source_frame_shape[1])},
        "grid_size": {"height": int(grid_shape[0]), "width": int(grid_shape[1])},
        "field_downsample": int(field_downsample),
        "source_cell_size": {"height": int(field_downsample * 2), "width": int(field_downsample)},
        "field_order": field_order,
        "checkpoint": checkpoint_path,
        "confidence_scale": CONFIDENCE_SCALE,
    }
    if hsv:
        metadata.update({
            "map_encoding": "hsv_to_rgb",
            "hue_start_degrees": 0,
            "hue_step_degrees": HSV_HUE_STEP_DEGREES,
            "saturation": 1.0,
            "value": "confidence",
            "rgb_channels": {"r": "hsv_to_rgb_red", "g": "hsv_to_rgb_green", "b": "hsv_to_rgb_blue"},
        })
    else:
        metadata.update({
            "map_encoding": "rgb_channels",
            "class_index_scale": CLASS_INDEX_SCALE,
            "rgb_channels": {"r": "class_index * class_index_scale", "g": "confidence", "b": "film_confidence"},
        })
    return metadata
