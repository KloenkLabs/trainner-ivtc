import torch

from trainner_ivtc.model import GlobalCadenceClassifier


def test_model_forward_variable_sizes() -> None:
    model = GlobalCadenceClassifier(in_channels=22, base_channels=4, channel_mult=(1, 2), dropout=0.0)
    for shape in [(2, 22, 24, 32), (2, 22, 31, 45)]:
        output = model(torch.rand(shape))
        assert output.shape == (2, 9)
