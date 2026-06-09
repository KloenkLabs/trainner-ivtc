import torch

from trainner_ivtc.labels import CLASS_NAMES
from trainner_ivtc.model import GlobalCadenceClassifier


def test_model_forward_variable_sizes() -> None:
    model = GlobalCadenceClassifier(in_channels=22, base_channels=4, channel_mult=(1, 2), dropout=0.0)
    for shape, grid_shape in [((2, 22, 24, 32), (12, 16)), ((2, 22, 31, 45), (16, 23))]:
        output = model(torch.rand(shape))
        assert output.shape == (2, len(CLASS_NAMES), *grid_shape)
        assert model.forward_global(torch.rand(shape)).shape == (2, len(CLASS_NAMES))
