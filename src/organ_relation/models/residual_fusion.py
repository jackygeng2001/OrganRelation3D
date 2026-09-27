"""Legacy residual fusion, with an explicit opt-in learnable scalar after phi."""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn


class ResidualFusion(nn.Module):
    def __init__(self, channels: int, *, content_channels: int, bias: bool,
                 learnable_relation_scale: bool = False, relation_scale_init: float = 1.0):
        super().__init__()
        for name, value in (('channels', channels), ('content_channels', content_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if type(bias) is not bool:
            raise ValueError('bias must be an explicit boolean')
        if type(learnable_relation_scale) is not bool:
            raise ValueError('learnable_relation_scale must be boolean')
        if isinstance(relation_scale_init, bool) or not isinstance(relation_scale_init, (float, int)) or not math.isfinite(relation_scale_init):
            raise ValueError('relation_scale_init must be finite')
        if not learnable_relation_scale and relation_scale_init != 1.0:
            raise ValueError('legacy fixed relation scale must be 1.0')
        self.channels = channels
        self.content_channels = content_channels
        self.phi = nn.Conv3d(content_channels, channels, kernel_size=1, bias=bias)
        self.learnable_relation_scale = learnable_relation_scale
        # Constant initialization consumes no RNG. Legacy state_dict has no new key.
        self.register_parameter('relation_scale', nn.Parameter(self.phi.weight.new_tensor(float(relation_scale_init)))
                                if learnable_relation_scale else None)

    def forward(self, features: Tensor, content: Tensor) -> Tensor:
        return self._forward(features, content, diagnostics_epsilon=None)[0]

    def forward_with_diagnostics(self, features: Tensor, content: Tensor, *, epsilon: float):
        """Same forward; detached whole-tensor L2 norms, FP64 reduction, scalars only."""
        if isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('diagnostics epsilon must be finite and positive')
        return self._forward(features, content, diagnostics_epsilon=epsilon)

    def _forward(self, features, content, *, diagnostics_epsilon):
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
        writeback = self.phi(content)
        residual = writeback if self.relation_scale is None else self.relation_scale * writeback
        fused = features + residual
        if not torch.isfinite(fused).all():
            raise ValueError('residual fusion produced non-finite values')
        stats = {}
        if diagnostics_epsilon is not None and self.relation_scale is not None:
            with torch.no_grad():
                denominator = torch.linalg.vector_norm(features.detach(), dtype=torch.float64) + diagnostics_epsilon
                stats = dict(gamma=self.relation_scale.detach().item(),
                    writeback_to_feature_norm=(torch.linalg.vector_norm(writeback.detach(), dtype=torch.float64) / denominator).item(),
                    scaled_writeback_to_feature_norm=(torch.linalg.vector_norm(residual.detach(), dtype=torch.float64) / denominator).item(),
                    norm_definition='whole_tensor_L2_fp64_reduction', epsilon=diagnostics_epsilon)
        return fused, stats
