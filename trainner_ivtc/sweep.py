from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from trainner_ivtc.config import load_config, save_config
from trainner_ivtc.data.synthetic import make_synthetic_dataset
from trainner_ivtc.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a training sweep for global luma cadence classifier configs.")
    parser.add_argument("--base-config", required=True, help="Base YAML config to copy from.")
    parser.add_argument("--sweep-dir", default="experiments/sweeps/voy_intro_luma_v1", help="Directory for generated configs and result summaries.")
    parser.add_argument("--overwrite-datasets", action="store_true", help="Regenerate sweep datasets even if manifests already exist.")
    parser.add_argument("--workers", type=int, default=None, help="Override synthetic generation workers.")
    return parser.parse_args()


def dataset_ready(dataset_dir: Path) -> bool:
    return (dataset_dir / "train_manifest.jsonl").exists() and (dataset_dir / "val_manifest.jsonl").exists()


def read_checkpoint_metrics(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metrics = checkpoint["metrics"]
    return {
        "epoch": int(checkpoint["epoch"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "film_confidence_on_film": float(metrics.get("film_confidence_on_film", 0.0)),
        "video_confidence_on_video": float(metrics.get("video_confidence_on_video", 0.0)),
    }


def build_config(base: dict[str, Any], sweep_root: Path, window_frames: int, batch_size: int, epochs: int) -> dict[str, Any]:
    config = deepcopy(base)
    combo_name = f"wf{window_frames}_bs{batch_size}_ep{epochs}"
    config["name"] = combo_name
    config["paths"]["dataset_dir"] = f"datasets/sweeps/voy_intro_luma_v1_wf{window_frames}"
    config["paths"]["output_dir"] = str(sweep_root / combo_name)
    config["data"]["window_frames"] = window_frames
    config["training"]["batch_size"] = batch_size
    config["training"]["epochs"] = epochs
    config["inference"]["window_frames"] = window_frames
    config["inference"]["batch_size"] = batch_size
    config["config_path"] = str(sweep_root / "configs" / f"{combo_name}.yaml")
    return config


def main() -> None:
    args = parse_args()
    base = load_config(args.base_config)
    sweep_root = Path(args.sweep_dir)
    config_dir = sweep_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for window_frames in (7, 11, 15):
        dataset_config = build_config(base, sweep_root, window_frames, 8, 8)
        dataset_dir = Path(dataset_config["paths"]["dataset_dir"])
        if args.overwrite_datasets or not dataset_ready(dataset_dir):
            make_synthetic_dataset(dataset_config, overwrite=args.overwrite_datasets, num_workers=args.workers)
        for batch_size in (4, 8, 16):
            for epochs in (8, 12, 16):
                config = build_config(base, sweep_root, window_frames, batch_size, epochs)
                config_path = config_dir / f"wf{window_frames}_bs{batch_size}_ep{epochs}.yaml"
                config["config_path"] = str(config_path)
                save_config(config, config_path)
                train(config)
                best_metrics = read_checkpoint_metrics(Path(config["paths"]["output_dir"]) / "checkpoints" / "best.pt")
                last_metrics = read_checkpoint_metrics(Path(config["paths"]["output_dir"]) / "checkpoints" / "last.pt")
                result = {
                    "window_frames": window_frames,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "output_dir": config["paths"]["output_dir"],
                    "best_epoch": best_metrics["epoch"],
                    "best_accuracy": best_metrics["accuracy"],
                    "best_macro_f1": best_metrics["macro_f1"],
                    "last_accuracy": last_metrics["accuracy"],
                    "last_macro_f1": last_metrics["macro_f1"],
                    "best_film_confidence_on_film": best_metrics["film_confidence_on_film"],
                    "best_video_confidence_on_video": best_metrics["video_confidence_on_video"],
                }
                results.append(result)
                results_path = sweep_root / "results.json"
                results_path.write_text(json.dumps(results, indent=4), encoding="utf-8")
                with (sweep_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                    writer.writeheader()
                    writer.writerows(results)
                best = max(results, key=lambda item: item["best_macro_f1"])
                print(f"current_best wf={best['window_frames']} bs={best['batch_size']} ep={best['epochs']} macro_f1={best['best_macro_f1']:.4f} acc={best['best_accuracy']:.4f}")


if __name__ == "__main__":
    main()
