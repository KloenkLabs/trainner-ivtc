import numpy as np

from trainner_ivtc.fields import frames_to_field_tensor, split_frame_to_fields, telecine_pair_for_frame, telecine_phase_for_frame, validate_window_frames, weave_field_pair


def test_split_frame_to_fields_tff_and_bff() -> None:
    frame = np.arange(24, dtype=np.uint8).reshape(6, 4)
    first, second = split_frame_to_fields(frame, "tff")
    assert np.array_equal(first, frame[0::2])
    assert np.array_equal(second, frame[1::2])
    first, second = split_frame_to_fields(frame, "bff")
    assert np.array_equal(first, frame[1::2])
    assert np.array_equal(second, frame[0::2])


def test_weave_field_pair_respects_order() -> None:
    first = np.full((4, 4), 10, dtype=np.uint8)
    second = np.full((4, 4), 20, dtype=np.uint8)
    tff = weave_field_pair(first, second, "tff")
    bff = weave_field_pair(first, second, "bff")
    assert np.all(tff[0::2] == 10)
    assert np.all(tff[1::2] == 20)
    assert np.all(bff[1::2] == 10)
    assert np.all(bff[0::2] == 20)


def test_frames_to_field_tensor_shape() -> None:
    frames = [np.full((8, 10), i, dtype=np.uint8) for i in range(11)]
    fields = frames_to_field_tensor(frames, "tff")
    assert fields.shape == (22, 4, 10)


def test_telecine_phase_and_pairs() -> None:
    assert [telecine_phase_for_frame(i) for i in range(5)] == [0, 1, 2, 3, 4]
    assert [telecine_pair_for_frame(i) for i in range(5)] == [(0, 0), (1, 1), (1, 2), (2, 3), (3, 3)]


def test_validate_window_frames() -> None:
    assert validate_window_frames(11) == 11
