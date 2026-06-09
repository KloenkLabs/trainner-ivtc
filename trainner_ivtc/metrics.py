from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from trainner_ivtc.grid import GRID_IGNORE_INDEX, global_logits_from_dense
from trainner_ivtc.labels import CLASS_NAMES, CLASS_TO_INDEX, FILM_CLASS_INDICES, VIDEO_CLASS_INDEX


ACTIVE_CLASS_WEIGHT_MIN = 0.001


def confusion_matrix(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.cpu(), predictions.cpu(), strict=False):
        matrix[int(target), int(prediction)] += 1
    return matrix


def metrics_from_confusion_matrix(matrix: Tensor, total: int, correct: int, active_class_indices: tuple[int, ...]) -> dict[str, Any]:
    recalls: dict[str, float] = {}
    f1_by_class: list[float] = []
    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i, i].float()
        support = matrix[i, :].sum().float()
        predicted = matrix[:, i].sum().float()
        recall = float((tp / support).item()) if support.item() > 0 else 0.0
        precision = float((tp / predicted).item()) if predicted.item() > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        recalls[name] = recall
        f1_by_class.append(f1)
    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_f1": sum(f1_by_class[i] for i in active_class_indices) / len(active_class_indices),
        "macro_f1_classes": [CLASS_NAMES[i] for i in active_class_indices],
        "recall": recalls,
        "confusion_matrix": matrix.tolist(),
    }


def active_class_indices_from_distribution(distribution: dict[str, float], min_weight: float = ACTIVE_CLASS_WEIGHT_MIN) -> tuple[int, ...]:
    unknown_names = [name for name in distribution if name not in CLASS_TO_INDEX]
    if unknown_names:
        raise KeyError(f"Unknown class names in class_distribution: {unknown_names}")
    active_indices = tuple(CLASS_TO_INDEX[name] for name, weight in distribution.items() if float(weight) >= min_weight)
    if not active_indices:
        raise ValueError(f"class_distribution must contain at least one class with weight >= {min_weight}")
    return active_indices


def summarize_class_matches(metrics: dict[str, Any], active_class_indices: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    matrix = metrics["confusion_matrix"]
    active_class_indices = active_class_indices if active_class_indices is not None else tuple(range(len(CLASS_NAMES)))
    matches: list[dict[str, Any]] = []
    for i in active_class_indices:
        row = matrix[i]
        support = int(sum(int(value) for value in row))
        correct = int(row[i])
        matches.append({"class_name": CLASS_NAMES[i], "correct": correct, "support": support, "recall": correct / support if support > 0 else 0.0})
    return matches


def summarize_classification(logits: Tensor, targets: Tensor, active_class_indices: tuple[int, ...] | None = None) -> dict[str, Any]:
    if logits.ndim == 4:
        logits = global_logits_from_dense(logits)
    probabilities = torch.softmax(logits.float(), dim=1)
    predictions = torch.argmax(probabilities, dim=1)
    num_classes = len(CLASS_NAMES)
    matrix = confusion_matrix(predictions, targets, num_classes)
    total = int(targets.numel())
    correct = int((predictions == targets).sum().item())
    active_class_indices = active_class_indices if active_class_indices is not None else tuple(range(num_classes))
    metrics = metrics_from_confusion_matrix(matrix, total, correct, active_class_indices)
    film_mask = torch.isin(targets.cpu(), torch.tensor(FILM_CLASS_INDICES))
    video_mask = targets.cpu() == VIDEO_CLASS_INDEX
    film_confidence = probabilities[:, list(FILM_CLASS_INDICES)].sum(dim=1).detach().cpu()
    video_confidence = probabilities[:, VIDEO_CLASS_INDEX].detach().cpu()
    metrics["film_confidence_on_film"] = float(film_confidence[film_mask].mean().item()) if film_mask.any() else 0.0
    metrics["video_confidence_on_video"] = float(video_confidence[video_mask].mean().item()) if video_mask.any() else 0.0
    return metrics


def summarize_grid_predictions(predictions: Tensor, targets: Tensor, active_class_indices: tuple[int, ...] | None = None, ignore_index: int = GRID_IGNORE_INDEX) -> dict[str, Any]:
    if predictions.shape != targets.shape:
        raise ValueError(f"Grid predictions and targets must have the same shape, got {tuple(predictions.shape)} and {tuple(targets.shape)}")
    num_classes = len(CLASS_NAMES)
    active_class_indices = active_class_indices if active_class_indices is not None else tuple(range(num_classes))
    predictions = predictions.detach().cpu().long()
    targets = targets.detach().cpu().long()
    valid = targets != ignore_index
    valid_predictions = predictions[valid]
    valid_targets = targets[valid]
    matrix = confusion_matrix(valid_predictions, valid_targets, num_classes) if int(valid_targets.numel()) > 0 else torch.zeros((num_classes, num_classes), dtype=torch.long)
    metrics = metrics_from_confusion_matrix(matrix, int(valid_targets.numel()), int((valid_predictions == valid_targets).sum().item()), active_class_indices)
    total_cells = int(targets.numel())
    ignored_cells = total_cells - int(valid_targets.numel())
    mixed_samples = 0
    mixed_recall_sum = 0.0
    mixed_exact = 0
    for sample_predictions, sample_targets in zip(predictions, targets, strict=False):
        sample_valid = sample_targets != ignore_index
        sample_classes = torch.unique(sample_targets[sample_valid])
        if int(sample_classes.numel()) <= 1:
            continue
        mixed_samples += 1
        matched = 0
        for class_index in sample_classes.tolist():
            class_mask = sample_targets == int(class_index)
            if bool((sample_predictions[class_mask] == int(class_index)).any()):
                matched += 1
        mixed_recall = matched / int(sample_classes.numel())
        mixed_recall_sum += mixed_recall
        mixed_exact += 1 if matched == int(sample_classes.numel()) else 0
    metrics["ignored_fraction"] = ignored_cells / total_cells if total_cells > 0 else 0.0
    metrics["valid_cells"] = int(valid_targets.numel())
    metrics["ignored_cells"] = ignored_cells
    metrics["mixed_samples"] = mixed_samples
    metrics["mixed_class_recall"] = mixed_recall_sum / mixed_samples if mixed_samples > 0 else 0.0
    metrics["mixed_sample_recall"] = mixed_exact / mixed_samples if mixed_samples > 0 else 0.0
    return metrics
