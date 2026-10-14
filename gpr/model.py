"""CNN for 32x32 single-channel GPR patches.

Three blocks of two 3x3 convolutions (16, 32, 64 channels), each followed by
max pooling, then a linear classifier. The same network, with a 10-class head,
is used for CIFAR-10 pretraining so its convolutional layers can be transferred.
"""
from __future__ import annotations

import torch
from torch import nn


def _block(c_in: int, c_out: int) -> list[nn.Module]:
    return [
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    ]


class GPRNet(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(*_block(1, 16), *_block(16, 32), *_block(32, 64))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64 * 4 * 4, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def transfer_conv_layers(target: GPRNet, source_state: dict[str, torch.Tensor], n_layers: int) -> int:
    """Copy the first ``n_layers`` conv layers (and their batch norms) from a pretrained state dict.

    Returns the number of conv layers actually copied.
    """
    copied = 0
    modules = list(target.features)
    for idx, m in enumerate(modules):
        if copied >= n_layers:
            break
        if isinstance(m, nn.Conv2d):
            for prefix in (f"features.{idx}.", f"features.{idx + 1}."):  # conv, then its BatchNorm
                for key, value in source_state.items():
                    if key.startswith(prefix):
                        target.state_dict()[key].copy_(value)
            copied += 1
    return copied
