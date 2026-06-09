from __future__ import annotations

from typing import Literal

import numpy as np


FieldOrder = Literal["tff", "bff"]
PULLDOWN_FIELD_PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (1, 2), (2, 3), (3, 3))


def validate_field_order(field_order: str) -> FieldOrder:
    if field_order not in {"tff", "bff"}:
        raise ValueError(f"field_order must be 'tff' or 'bff', got {field_order!r}")
    return field_order  # type: ignore[return-value]


def validate_window_frames(window_frames: int) -> int:
    if window_frames <= 0 or window_frames % 2 == 0:
        raise ValueError(f"window_frames must be a positive odd integer, got {window_frames}")
    return window_frames


def ensure_even_luma(luma: np.ndarray) -> np.ndarray:
    if luma.ndim != 2:
        raise ValueError(f"Expected 2D luma image, got shape {luma.shape}")
    h, w = luma.shape
    if h < 2 or w < 2:
        raise ValueError(f"Luma image is too small: {luma.shape}")
    return luma[: h - (h % 2), : w - (w % 2)]


def split_frame_to_fields(luma: np.ndarray, field_order: FieldOrder = "tff") -> tuple[np.ndarray, np.ndarray]:
    luma = ensure_even_luma(luma)
    field_order = validate_field_order(field_order)
    top = luma[0::2, :]
    bottom = luma[1::2, :]
    return (top, bottom) if field_order == "tff" else (bottom, top)


def weave_field_pair(first_luma: np.ndarray, second_luma: np.ndarray, field_order: FieldOrder = "tff") -> np.ndarray:
    first_luma = ensure_even_luma(first_luma)
    second_luma = ensure_even_luma(second_luma)
    if first_luma.shape != second_luma.shape:
        raise ValueError(f"Field source shapes must match, got {first_luma.shape} and {second_luma.shape}")
    field_order = validate_field_order(field_order)
    out = np.empty_like(first_luma)
    if field_order == "tff":
        out[0::2, :] = first_luma[0::2, :]
        out[1::2, :] = second_luma[1::2, :]
    else:
        out[1::2, :] = first_luma[1::2, :]
        out[0::2, :] = second_luma[0::2, :]
    return out


def frames_to_field_tensor(frames: list[np.ndarray], field_order: FieldOrder = "tff") -> np.ndarray:
    fields: list[np.ndarray] = []
    expected_shape: tuple[int, int] | None = None
    for frame in frames:
        first, second = split_frame_to_fields(frame, field_order)
        if expected_shape is None:
            expected_shape = first.shape
        elif first.shape != expected_shape or second.shape != expected_shape:
            raise ValueError(f"All fields must have shape {expected_shape}, got {first.shape} and {second.shape}")
        fields.append(first)
        fields.append(second)
    return np.stack(fields, axis=0)


def telecine_phase_for_frame(video_frame_index: int, phase_offset: int = 0) -> int:
    return (video_frame_index + phase_offset) % 5


def telecine_pair_for_frame(video_frame_index: int, phase_offset: int = 0) -> tuple[int, int]:
    phase = telecine_phase_for_frame(video_frame_index, phase_offset)
    cycle = (video_frame_index + phase_offset) // 5
    first_offset, second_offset = PULLDOWN_FIELD_PAIRS[phase]
    return cycle * 4 + first_offset, cycle * 4 + second_offset


def clamped_window_indices(center: int, total: int, window_frames: int) -> list[int]:
    if total <= 0:
        raise ValueError("total must be greater than zero")
    window_frames = validate_window_frames(window_frames)
    radius = window_frames // 2
    return [min(max(center + offset, 0), total - 1) for offset in range(-radius, radius + 1)]
