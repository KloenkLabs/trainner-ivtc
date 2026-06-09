import numpy as np

from trainner_ivtc.data.synthetic import generate_sample, generate_sample_frames, resolve_worker_count
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
