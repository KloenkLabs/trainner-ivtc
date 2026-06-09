from __future__ import annotations

import argparse

from trainner_ivtc.config import load_config
from trainner_ivtc.data.synthetic import make_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic cadence classification samples.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite matching generated sample files.")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker threads. Defaults to data.num_workers, or CPU core count when set to auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    make_synthetic_dataset(config, overwrite=args.overwrite, num_workers=args.workers)
    print(f"Wrote synthetic dataset to {config['paths']['dataset_dir']}")


if __name__ == "__main__":
    main()
