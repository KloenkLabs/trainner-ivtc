import torch

from trainner_ivtc.labels import CLASS_NAMES, class_id, class_index, prediction_to_json


def test_class_mapping() -> None:
    assert class_id(0) == "0"
    assert class_id(5) == "V"
    assert class_index("film_phase_3") == 3
    assert class_index("U") == 8


def test_prediction_to_json_schema() -> None:
    probabilities = torch.zeros(len(CLASS_NAMES))
    probabilities[2] = 0.9
    probabilities[5] = 0.1
    result = prediction_to_json(123, probabilities)
    assert result["frame_index"] == 123
    assert result["class_id"] == "2"
    assert result["class_name"] == "film_phase_2"
    assert result["recommended_action"] == "ivtc_global"
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
