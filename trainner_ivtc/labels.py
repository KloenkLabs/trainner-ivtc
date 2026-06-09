from __future__ import annotations

from typing import Any

import torch


CLASS_NAMES: tuple[str, ...] = (
    "film_phase_0",
    "film_phase_1",
    "film_phase_2",
    "film_phase_3",
    "film_phase_4",
    "video",
    "blend",
    "unknown",
)
CLASS_IDS: tuple[str, ...] = ("0", "1", "2", "3", "4", "V", "B", "U")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_INDEX = {class_id: i for i, class_id in enumerate(CLASS_IDS)}
FILM_CLASS_INDICES = tuple(range(5))
VIDEO_CLASS_INDEX = CLASS_TO_INDEX["video"]


def output_class_name(name: str) -> str:
    if name.startswith("film_phase_"):
        return "pd_" + name.removeprefix("film_phase_")
    return name


def rounded_float(value: float) -> float:
    return round(float(value), 6)


def class_name(index: int) -> str:
    return CLASS_NAMES[index]


def class_id(index: int) -> str:
    return CLASS_IDS[index]


def class_index(name_or_id: str) -> int:
    if name_or_id in CLASS_TO_INDEX:
        return CLASS_TO_INDEX[name_or_id]
    if name_or_id in ID_TO_INDEX:
        return ID_TO_INDEX[name_or_id]
    raise KeyError(f"Unknown class name or id: {name_or_id}")


def probabilities_to_dict(probabilities: torch.Tensor) -> dict[str, float]:
    values = probabilities.detach().cpu().tolist()
    return {output_class_name(name): rounded_float(values[i]) for i, name in enumerate(CLASS_NAMES)}


def prediction_to_json(frame_index: int, probabilities: torch.Tensor) -> dict[str, Any]:
    probabilities = probabilities.detach().float().cpu()
    confidence, best_index_tensor = torch.max(probabilities, dim=0)
    best_index = int(best_index_tensor.item())
    film_confidence = float(probabilities[list(FILM_CLASS_INDICES)].sum().item())
    video_confidence = float(probabilities[VIDEO_CLASS_INDEX].item())
    best_name = class_name(best_index)
    return {
        "idx": int(frame_index),
        "class_id": class_id(best_index),
        "class_name": output_class_name(best_name),
        "conf": rounded_float(confidence.item()),
        "film_conf": rounded_float(film_confidence),
        "video_conf": rounded_float(video_confidence),
        "probs": probabilities_to_dict(probabilities),
    }
