from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Dataset

from trainner_ivtc.config import load_config, save_config
from trainner_ivtc.dataset import CadenceFrameDataset, OnlineSyntheticCadenceDataset, manifest_path
from trainner_ivtc.labels import CLASS_NAMES
from trainner_ivtc.metrics import active_class_indices_from_distribution, summarize_class_matches, summarize_classification
from trainner_ivtc.model import build_model


LOGGER_NAME = "trainner_ivtc.train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the global luma cadence classifier.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def make_loader(dataset: Dataset, batch_size: int, num_workers: int, shuffle: bool, prefetch_factor: int = 2) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def prepare_fields(fields: torch.Tensor, device: torch.device) -> torch.Tensor:
    fields = fields.to(device, non_blocking=True)
    if fields.dtype == torch.uint8:
        return fields.float().div_(255.0)
    return fields.float()


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def format_class_match_summary(matches: list[dict[str, Any]]) -> str:
    if not matches:
        return "none"
    return ", ".join(f"{row['class_name']} correct={int(row['correct'])}/{int(row['support'])} recall={float(row['recall']):.4f}" for row in matches)


def setup_logger(output_dir: Path) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.info("Logging to %s", log_path)
    return logger, log_path


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool, active_class_indices: tuple[int, ...]) -> dict[str, Any]:
    model.eval()
    logits_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in loader:
            fields = prepare_fields(batch["fields"], device)
            targets = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(fields)
            logits_list.append(logits.detach().cpu())
            targets_list.append(targets.detach().cpu())
    return summarize_classification(torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0), active_class_indices)


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], epoch: int, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "model": config["model"],
            "window_frames": int(config["data"]["window_frames"]),
            "field_order": str(config["data"]["field_order"]),
            "class_names": CLASS_NAMES,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def train(config: dict[str, Any]) -> None:
    seed_everything(int(config.get("seed", 1234)))
    dataset_dir = Path(config["paths"]["dataset_dir"])
    output_dir = Path(config["paths"]["output_dir"])
    logger, log_path = setup_logger(output_dir)
    save_config(config, output_dir / "config_resolved.yaml")
    dataset_mode = str(config["data"].get("dataset_mode", "online"))
    logger.info("Preparing datasets: mode=%s source_cache_mode=%s", dataset_mode, config["data"].get("source_cache_mode", "lru"))
    if dataset_mode == "manifest":
        train_manifest = manifest_path(dataset_dir, "train")
        val_manifest = manifest_path(dataset_dir, "val")
        if not train_manifest.exists() or not val_manifest.exists():
            raise FileNotFoundError(f"Missing synthetic manifests in {dataset_dir}. Run python -m trainner_ivtc.data.make_synthetic --config {config['config_path']} first.")
        train_dataset = CadenceFrameDataset(train_manifest)
        val_dataset = CadenceFrameDataset(val_manifest)
    else:
        train_dataset = OnlineSyntheticCadenceDataset(config, "train")
        val_dataset = OnlineSyntheticCadenceDataset(config, "val")
    training = config["training"]
    device = resolve_device(str(training.get("device", "cuda")))
    amp = bool(training.get("amp", True))
    prefetch_factor = int(training.get("prefetch_factor", 2))
    train_loader = make_loader(train_dataset, int(training["batch_size"]), int(training["num_workers"]), shuffle=True, prefetch_factor=prefetch_factor)
    val_loader = make_loader(val_dataset, int(training["batch_size"]), int(training["num_workers"]), shuffle=False, prefetch_factor=prefetch_factor)
    model = build_model(config["model"], in_channels=int(config["data"]["window_frames"]) * 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device="cuda", enabled=amp and device.type == "cuda")
    active_class_indices = active_class_indices_from_distribution(config["data"]["class_distribution"])
    active_class_names = [CLASS_NAMES[i] for i in active_class_indices]
    best_macro_f1 = -1.0
    best_checkpoint_path = output_dir / "checkpoints" / "best.pt"
    last_checkpoint_path = output_dir / "checkpoints" / "last.pt"
    total_iters = 0
    start_time = time.perf_counter()
    final_metrics: dict[str, Any] | None = None
    logger.info("Starting training: train_samples=%d val_samples=%d epochs=%d batch_size=%d window_frames=%d", len(train_dataset), len(val_dataset), int(training["epochs"]), int(training["batch_size"]), int(config["data"]["window_frames"]))
    logger.info("Augmentation: noise_std=%.3f class_distribution=%s", float(config["data"].get("noise_std", 0.0)), json.dumps(config["data"].get("class_distribution", {}), separators=(",", ":")))
    logger.info("Macro F1 active classes: %s", ",".join(active_class_names))
    for epoch in range(1, int(training["epochs"]) + 1):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        epoch_start_time = time.perf_counter()
        last_log_time = epoch_start_time
        last_log_step = 0
        wait_start_time = epoch_start_time
        recent_data_wait = 0.0
        recent_step_time = 0.0
        recent_loss_vals = []
        for step, batch in enumerate(train_loader, start=1):
            batch_ready_time = time.perf_counter()
            recent_data_wait += batch_ready_time - wait_start_time
            step_start_time = batch_ready_time
            total_iters += 1
            fields = prepare_fields(batch["fields"], device)
            targets = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
                logits = model(fields)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            current_loss = float(loss.detach().item())
            running_loss += current_loss
            recent_loss_vals.append(current_loss)
            step_end_time = time.perf_counter()
            recent_step_time += step_end_time - step_start_time
            if step == 10 or step % int(training["print_freq"]) == 0:
                now = time.perf_counter()
                recent_steps = step - last_log_step
                recent_it_s = recent_steps / max(now - last_log_time, 1e-9)
                recent_avg_loss = sum(recent_loss_vals) / len(recent_loss_vals)
                recent_loss_vals.clear()
                logger.info("epoch=%02d  step=%05d/%05d  loss=%.5f  loss_now=%.5f  it/s=%.1f", epoch, step, len(train_loader), running_loss / step, recent_avg_loss, recent_it_s)
                last_log_time = now
                last_log_step = step
                recent_data_wait = 0.0
                recent_step_time = 0.0
            wait_start_time = time.perf_counter()
        train_elapsed = time.perf_counter() - epoch_start_time
        train_it_s = len(train_loader) / max(train_elapsed, 1e-9)
        metrics = evaluate(model, val_loader, device, amp, active_class_indices)
        final_metrics = metrics
        train_loss = running_loss / max(len(train_loader), 1)
        logger.info("epoch=%02d  train_loss=%.5f  train_it_s=%.3f  val_accuracy=%.4f  val_macro_f1=%.4f", epoch, train_loss, train_it_s, float(metrics["accuracy"]), float(metrics["macro_f1"]))
        logger.debug("epoch_metrics_json=%s", json.dumps({"epoch": epoch, "loss": train_loss, "val": metrics}, separators=(",", ":")))
        save_checkpoint(last_checkpoint_path, model, config, epoch, metrics)
        if float(metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(metrics["macro_f1"])
            save_checkpoint(best_checkpoint_path, model, config, epoch, metrics)
    elapsed = time.perf_counter() - start_time
    avg_iter_per_second = total_iters / elapsed if elapsed > 0 else 0.0
    final_accuracy = float(final_metrics["accuracy"]) if final_metrics is not None else 0.0
    final_macro_f1 = float(final_metrics["macro_f1"]) if final_metrics is not None else 0.0
    logger.info("Training complete:\n%d iters in %s - avg_it_s=%.2f  best_macro_f1=%.4f  final_accuracy=%.4f  final_macro_f1=%.4f", total_iters, format_duration(elapsed), avg_iter_per_second, best_macro_f1, final_accuracy, final_macro_f1)
    if final_metrics is not None:
        class_matches = summarize_class_matches(final_metrics, active_class_indices)
        most_matched = sorted(class_matches, key=lambda row: (int(row["correct"]), float(row["recall"])), reverse=True)
        least_matched = sorted(class_matches, key=lambda row: (int(row["correct"]), float(row["recall"])))
        logger.info("Most correctly matched classes: %s", format_class_match_summary(most_matched))
        logger.info("Least correctly matched classes: %s", format_class_match_summary(least_matched))
    logger.info("Best checkpoint (macro_f1=%.4f) saved to %s", best_macro_f1, best_checkpoint_path)
    logger.info("Last checkpoint (macro_f1=%.4f) saved to %s", final_macro_f1, last_checkpoint_path)
    logger.info("Log was written to %s", log_path)


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
