import torch

from trainner_ivtc.labels import CLASS_NAMES, class_id, class_index, prediction_to_json


def test_class_mapping() -> None:
    assert class_id(0) == "0"
    assert class_id(5) == "V"
    assert class_index("film_phase_3") == 3
    assert class_index("U") == 7


def test_prediction_to_json_schema() -> None:
    probabilities = torch.zeros(len(CLASS_NAMES))
    probabilities[2] = 0.91234567
    probabilities[5] = 0.08765433
    result = prediction_to_json(123, probabilities)
    assert result["idx"] == 123
    assert result["class_id"] == "2"
    assert result["class_name"] == "pd_2"
    assert result["conf"] == 0.912346
    assert result["film_conf"] == 0.912346
    assert result["video_conf"] == 0.087654
    assert "recommended_action" not in result
    assert set(result["probs"].keys()) == {"pd_0", "pd_1", "pd_2", "pd_3", "pd_4", "video", "blend", "unknown"}
    assert result["probs"]["pd_2"] == 0.912346
