"""METHOD_SPEC residual fusion: Fprime = F + Conv1x1x1(G), nothing else."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class ResidualFusion(nn.Module):
    def __init__(self, channels: int, *, content_channels: int, bias: bool):
        super().__init__()
        for name, value in (('channels', channels), ('content_channels', content_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(bias) is not bool:
            raise ValueError('bias must be an explicit boolean')
        self.channels = channels
        self.content_channels = content_channels
        self.phi = nn.Conv3d(content_channels, channels, kernel_size=1, bias=bias)

    def forward(self, features: Tensor, content: Tensor) -> Tensor:
        if not isinstance(features, Tensor) or features.ndim != 5 or min(features.shape) < 1:
            raise ValueError('features must have positive shape [B,C,Df,Hf,Wf]')
        if features.shape[1] != self.channels:
            raise ValueError('features must match configured channels')
        if not isinstance(content, Tensor) or tuple(content.shape) != (
            features.shape[0], self.content_channels, *features.shape[2:]
        ):
            raise ValueError('content must have shape [B,Cg,Df,Hf,Wf] matching the feature batch/grid')
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError('ResidualFusion requires float32 or float64; AMP is not validated')
        if content.dtype != features.dtype or content.device != features.device:
            raise ValueError('features and content must share device and dtype')
        if torch.is_autocast_enabled(features.device.type):
            raise ValueError('ResidualFusion requires autocast disabled until AMP is validated')
        if any(p.dtype != features.dtype or p.device != features.device for p in self.parameters()):
            raise ValueError('module parameters and inputs must share device and dtype')
        for name, value in (('features', features), ('content', content)):
            if not torch.isfinite(value).all():
                raise ValueError(f'{name} must contain only finite values')
        fused = features + self.phi(content)
        if not torch.isfinite(fused).all():
            raise ValueError('residual fusion produced non-finite values')
        return fused
