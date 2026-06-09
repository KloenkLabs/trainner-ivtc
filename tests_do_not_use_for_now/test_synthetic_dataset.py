import numpy as np

from trainner_ivtc.data.synthetic import generate_sample, generate_sample_frames, make_synthetic_dataset, resolve_worker_count, split_sequence_paths
from trainner_ivtc.image_io import save_luma_image
from trainner_ivtc.labels import CLASS_TO_INDEX


def test_generate_sample_tensor_shape() -> None:
    rng = np.random.default_rng(1)
    fields = generate_sample(rng, 32, 48, "tff", CLASS_TO_INDEX["film_phase_2"], None, 0.0)
    assert fields.shape == (22, 16, 48)
    assert fields.dtype == np.uint8


def test_generate_video_sample_tensor_shape() -> None:
    rng = np.random.default_rng(2)
    fields = generate_sample(rng, 32, 48, "bff", CLASS_TO_INDEX["video"], None, 0.0)
    assert fields.shape == (22, 16, 48)
    assert fields.dtype == np.uint8


def test_generate_sample_frames_shape() -> None:
    rng = np.random.default_rng(3)
    frames = generate_sample_frames(rng, 32, 48, "tff", CLASS_TO_INDEX["blend"], None, 0.0)
    assert len(frames) == 11
    assert frames[0].shape == (32, 48)
    assert frames[0].dtype == np.uint8


def test_resolve_worker_count() -> None:
    assert resolve_worker_count(1) == 1
    assert resolve_worker_count("auto") >= 1


def test_split_sequence_paths_uses_percentage_boundary(tmp_path) -> None:
    paths = [tmp_path / f"{i:04d}.png" for i in range(10)]
    train_paths, val_paths = split_sequence_paths(paths, 90)
    assert len(train_paths) == 9
    assert len(val_paths) == 1
    assert train_paths[-1].name == "0008.png"
    assert val_paths[0].name == "0009.png"


def test_make_synthetic_dataset_uses_percentage_counts(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for i in range(10):
        save_luma_image(source / f"{i:04d}.png", np.full((32, 48), i, dtype=np.uint8))
    config = {
        "seed": 1234,
        "paths": {"dataset_dir": str(tmp_path / "dataset")},
        "data": {
            "source_dirs": [str(source)],
            "field_order": "tff",
            "window_frames": 11,
            "height": 32,
            "width": 48,
            "train_samples_pct": 90,
            "val_samples_pct": 10,
            "source_cache_size": 2,
            "noise_std": 0.0,
            "class_distribution": {"film_phase_0": 1.0},
        },
    }
    make_synthetic_dataset(config, overwrite=True, num_workers=1)
    train_lines = (tmp_path / "dataset" / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    val_lines = (tmp_path / "dataset" / "val_manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(train_lines) == 9
    assert len(val_lines) == 1
