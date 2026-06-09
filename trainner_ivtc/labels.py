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
    "scene_cut",
    "unknown",
)
CLASS_IDS: tuple[str, ...] = ("0", "1", "2", "3", "4", "V", "B", "C", "U")
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_INDEX = {class_id: i for i, class_id in enumerate(CLASS_IDS)}
FILM_CLASS_INDICES = tuple(range(5))
VIDEO_CLASS_INDEX = CLASS_TO_INDEX["video"]


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
    return {name: float(values[i]) for i, name in enumerate(CLASS_NAMES)}


def recommended_action(best_name: str, film_confidence: float, video_confidence: float) -> str:
    if best_name.startswith("film_phase_") and film_confidence >= 0.5:
        return "ivtc_global"
    if best_name == "video" and video_confidence >= 0.5:
        return "deinterlace_or_preserve_30000_1001"
    if best_name == "scene_cut":
        return "reset_cadence"
    if best_name == "blend":
        return "blend_or_transition_fallback"
    return "heuristic_fallback"


def prediction_to_json(frame_index: int, probabilities: torch.Tensor) -> dict[str, Any]:
    probabilities = probabilities.detach().float().cpu()
    confidence, best_index_tensor = torch.max(probabilities, dim=0)
    best_index = int(best_index_tensor.item())
    film_confidence = float(probabilities[list(FILM_CLASS_INDICES)].sum().item())
    video_confidence = float(probabilities[VIDEO_CLASS_INDEX].item())
    best_name = class_name(best_index)
    return {
        "frame_index": int(frame_index),
        "class_id": class_id(best_index),
        "class_name": best_name,
        "confidence": float(confidence.item()),
        "film_confidence": film_confidence,
        "video_confidence": video_confidence,
        "probabilities": probabilities_to_dict(probabilities),
        "recommended_action": recommended_action(best_name, film_confidence, video_confidence),
    }
