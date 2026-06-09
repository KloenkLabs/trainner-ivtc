from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def iter_image_paths(folder: str | Path) -> list[Path]:
    root = Path(folder)
    paths = [path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    paths.sort(key=lambda path: (path.name.casefold(), path.name))
    if not paths:
        raise FileNotFoundError(f"No image files found in {root}")
    return paths


def load_luma_image(path: str | Path, size: tuple[int, int] | None = None) -> np.ndarray:
    with Image.open(path) as image:
        if size is not None:
            target_size = (size[1], size[0])
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.BICUBIC)
        if image.mode == "L":
            return np.asarray(image, dtype=np.uint8)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return np.clip(np.rint(luma), 0, 255).astype(np.uint8)


def save_luma_image(path: str | Path, luma: np.ndarray) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(luma.astype(np.uint8), mode="L").save(out_path)
