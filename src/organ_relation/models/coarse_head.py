"""Coarse prediction on the unchanged feature grid; legacy softmax or region sigmoid."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class CoarsePrediction(NamedTuple):
    logits: Tensor  # [B,outputs,Df,Hf,Wf]; legacy default has 16 classes.
    probabilities: Tensor  # Same shape, with the explicitly selected activation.


class CoarseHead(nn.Module):
    """A single 1x1x1 convolution followed by softmax or independent sigmoid.

    Bias is explicit so a test choice does not freeze the formal configuration.
    The caller moves/converts this module and its input together.
    """

    def __init__(self, in_channels: int, *, bias: bool, out_channels: int = 16, activation: str = "softmax"):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError('in_channels must be a positive integer')
        if type(bias) is not bool:
            raise ValueError('bias must be an explicit boolean')
        if type(out_channels) is not int or out_channels < 1 or activation not in ('softmax', 'sigmoid'):
            raise ValueError('positive out_channels and softmax/sigmoid activation required')
        self.activation = activation
        self.projection = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, features: Tensor) -> CoarsePrediction:
        if not isinstance(features, Tensor) or features.ndim != 5:
            raise ValueError('features must have shape [B,C,Df,Hf,Wf]')
        if min(features.shape) < 1 or features.shape[1] != self.projection.in_channels:
            raise ValueError('features require positive sizes and configured channels')
        if not features.is_floating_point():
            raise ValueError('features must use a floating dtype')
        logits = self.projection(features)
        return CoarsePrediction(logits, logits.sigmoid() if self.activation == "sigmoid" else torch.softmax(logits, dim=1))
