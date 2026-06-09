import numpy as np
import torch

from trainner_ivtc.grid import decode_probability_map, encode_probability_map
from trainner_ivtc.labels import CLASS_NAMES


def test_rgb_grid_map_decode_reconstructs_encoded_arrays() -> None:
    probabilities = torch.zeros(len(CLASS_NAMES), 2, 3)
    probabilities[2, :, :] = 0.80
    probabilities[5, 0, 1] = 0.90
    probabilities[2, 0, 1] = 0.05
    encoded = encode_probability_map(probabilities)
    decoded = decode_probability_map(encoded)
    expected_confidence = encoded[:, :, 1].astype(np.float32) / 255.0
    expected_film_confidence = encoded[:, :, 2].astype(np.float32) / 255.0
    assert np.array_equal(decoded["class_index"], encoded[:, :, 0])
    assert np.array_equal(decoded["confidence"], expected_confidence)
    assert np.array_equal(decoded["film_confidence"], expected_film_confidence)
