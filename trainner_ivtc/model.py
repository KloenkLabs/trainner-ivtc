from __future__ import annotations

import torch
from torch import Tensor, nn

from trainner_ivtc.grid import global_logits_from_dense
from trainner_ivtc.labels import CLASS_NAMES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.body(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            ResidualBlock(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class GlobalCadenceClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 22,
        num_classes: int = len(CLASS_NAMES),
        base_channels: int = 32,
        channel_mult: tuple[int, ...] | list[int] = (1, 2, 4, 4),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be greater than zero")
        channels = [base_channels * mult for mult in channel_mult]
        if not channels:
            raise ValueError("channel_mult must contain at least one value")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
            ResidualBlock(channels[0]),
        )
        blocks: list[nn.Module] = []
        for in_ch, out_ch in zip(channels, channels[1:], strict=False):
            blocks.append(DownsampleBlock(in_ch, out_ch))
        self.encoder = nn.Sequential(*blocks)
        self.dense_head = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(channels[-1], num_classes, 1),
        )

    def forward_dense(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.shape[1]}")
        x = self.stem(x)
        x = self.encoder(x)
        return self.dense_head(x)

    def forward_global(self, x: Tensor) -> Tensor:
        return global_logits_from_dense(self.forward_dense(x))

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_dense(x)


def build_model(model_config: dict, in_channels: int = 22) -> GlobalCadenceClassifier:
    return GlobalCadenceClassifier(
        in_channels=in_channels,
        base_channels=int(model_config.get("base_channels", 32)),
        channel_mult=tuple(model_config.get("channel_mult", [1, 2, 4, 4])),
        dropout=float(model_config.get("dropout", 0.1)),
    )


def upgrade_legacy_global_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if "dense_head.1.weight" in state_dict:
        return state_dict
    if "head.3.weight" not in state_dict or "head.3.bias" not in state_dict:
        return state_dict
    upgraded = {key: value for key, value in state_dict.items() if not key.startswith("head.")}
    upgraded["dense_head.1.weight"] = state_dict["head.3.weight"].view(state_dict["head.3.weight"].shape[0], state_dict["head.3.weight"].shape[1], 1, 1)
    upgraded["dense_head.1.bias"] = state_dict["head.3.bias"]
    return upgraded
