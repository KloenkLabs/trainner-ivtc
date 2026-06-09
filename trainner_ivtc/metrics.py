from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from trainner_ivtc.labels import CLASS_NAMES, FILM_CLASS_INDICES, VIDEO_CLASS_INDEX


def confusion_matrix(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.cpu(), predictions.cpu(), strict=False):
        matrix[int(target), int(prediction)] += 1
    return matrix


def summarize_classification(logits: Tensor, targets: Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits.float(), dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    num_classes = len(CLASS_NAMES)
    matrix = confusion_matrix(predictions, targets, num_classes)
    total = int(targets.numel())
    correct = int((predictions == targets).sum().item())
    recalls: dict[str, float] = {}
    f1_values: list[float] = []
    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i].float()
        support = matrix[i, :].sum().float()
        predicted = matrix[:, i].sum().float()
        recall = float((tp / support).item()) if support.item() > 0 else 0.0
        precision = float((tp / predicted).item()) if predicted.item() > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls[name] = recall
        f1_values.append(f1)
    film_mask = torch.isin(targets.cpu(), torch.tensor(FILM_CLASS_INDICES))
    video_mask = targets.cpu() == VIDEO_CLASS_INDEX
    film_confidence = probabilities[:, list(FILM_CLASS_INDICES)].sum(dim=1).detach().cpu()
    video_confidence = probabilities[:, VIDEO_CLASS_INDEX].detach().cpu()
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values),
        "recall": recalls,
        "confusion_matrix": matrix.tolist(),
        "film_confidence_on_film": float(film_confidence[film_mask].mean().item()) if film_mask.any() else 0.0,
        "video_confidence_on_video": float(video_confidence[video_mask].mean().item()) if video_mask.any() else 0.0,
    }
