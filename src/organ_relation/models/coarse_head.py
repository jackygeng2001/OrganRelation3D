"""Sixteen-class coarse prediction at the unchanged deepest feature grid."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


class CoarsePrediction(NamedTuple):
    logits: Tensor  # [B,16,Df,Hf,Wf], background at class 0.
    probabilities: Tensor  # Same shape; softmax over all 16 classes.


class CoarseHead(nn.Module):
    """A single 1x1x1 convolution followed by class softmax.

    Bias is explicit so a test choice does not freeze the formal configuration.
    The caller moves/converts this module and its input together.
    """

    def __init__(self, in_channels: int, *, bias: bool):
        super().__init__()
        if type(in_channels) is not int or in_channels < 1:
            raise ValueError('in_channels must be a positive integer')
        if type(bias) is not bool:
            raise ValueError('bias must be an explicit boolean')
        self.projection = nn.Conv3d(in_channels, 16, kernel_size=1, bias=bias)

    def forward(self, features: Tensor) -> CoarsePrediction:
        if not isinstance(features, Tensor) or features.ndim != 5:
            raise ValueError('features must have shape [B,C,Df,Hf,Wf]')
        if min(features.shape) < 1 or features.shape[1] != self.projection.in_channels:
            raise ValueError('features require positive sizes and configured channels')
        if not features.is_floating_point():
            raise ValueError('features must use a floating dtype')
        logits = self.projection(features)
        return CoarsePrediction(logits, torch.softmax(logits, dim=1))
