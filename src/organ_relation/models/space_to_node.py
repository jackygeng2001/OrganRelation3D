"""METHOD_SPEC Space-to-Node formulas, with no labels or hard node masks."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


ORGAN_LABEL_IDS = tuple(range(1, 16))
CENTROID_AXES = ('D', 'H', 'W')


class OrganNodes(NamedTuple):
    """Nodes follow the caller's fixed channel order, excluding optional background.

    Centroid components follow feature tensor D,H,W (not world/mm axes).
    mass/centroid/size/confidence are separate named attributes, not channels
    mixed into z0. All outputs retain autograd history for future reasoning.
    """

    z0: Tensor  # [B,nodes,C]
    mass: Tensor  # [B,nodes,1]
    centroid: Tensor  # [B,nodes,3], normalized D,H,W coordinates
    size: Tensor  # [B,nodes,1], mass / (Df*Hf*Wf)
    confidence: Tensor  # [B,nodes,1], sum(P_i**2) / (mass + epsilon)


class SpaceToNode(nn.Module):
    """Parameter-free soft pooling on the full feature grid.

    epsilon is required: load its current value from the explicit baseline config.
    Inputs must share device, dtype and grid. This stage supports FP32/FP64;
    autocast must be disabled; AMP accumulation needs separate validation. P is the
    legacy 16-class softmax or independent region output: this module does not renormalize, clip, binarize,
    or infer organ presence. Probability values are the caller's contract.
    """

    def __init__(self, *, epsilon: float, node_count: int = 15, include_background: bool = True):
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise ValueError('epsilon must be a finite positive number')
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be a finite positive number')
        if type(node_count) is not int or node_count < 1 or type(include_background) is not bool:
            raise ValueError('positive node_count and explicit background-channel flag required')
        self.node_count = node_count
        self.include_background = include_background
        self.epsilon = float(epsilon)

    def extra_repr(self) -> str:
        return f'epsilon={self.epsilon!r}'

    def forward(self, features: Tensor, probabilities: Tensor) -> OrganNodes:
        if not isinstance(features, Tensor) or features.ndim != 5 or min(features.shape) < 1:
            raise ValueError('features must have positive shape [B,C,Df,Hf,Wf]')
        if not isinstance(probabilities, Tensor) or probabilities.ndim != 5:
            raise ValueError(f'probabilities must have shape [B,{self.node_count + int(self.include_background)},Df,Hf,Wf]')
        expected = (features.shape[0], self.node_count + int(self.include_background), *features.shape[2:])
        if tuple(probabilities.shape) != expected:
            raise ValueError(f'probabilities must match the feature batch/grid and have {expected[1]} classes')
        if probabilities.device != features.device or probabilities.dtype != features.dtype:
            raise ValueError('features and probabilities must share device and dtype')
        if features.dtype not in (torch.float32, torch.float64):
            raise ValueError('SpaceToNode currently requires float32 or float64; AMP is not validated')
        if torch.is_autocast_enabled(features.device.type):
            raise ValueError('SpaceToNode requires autocast disabled until AMP is validated')
        limits = torch.finfo(features.dtype)
        if not limits.tiny * limits.eps <= self.epsilon <= limits.max:
            raise ValueError('epsilon must be representable and positive in the input dtype')

        foreground = probabilities[:, int(self.include_background):]  # [B,nodes,Df,Hf,Wf]; no background node.
        weights = foreground.flatten(start_dim=2)  # [B,nodes,N]
        mass = weights.sum(dim=-1, keepdim=True)
        denominator = mass + self.epsilon
        # PF exactly once. Never allocate [B,nodes,C,N] or an organ feature volume.
        numerator = torch.bmm(weights, features.flatten(start_dim=2).transpose(1, 2))
        z0 = numerator / denominator

        # Marginalize the other two axes to avoid a dense [N,3] coordinate grid.
        components = []
        for axis, length in enumerate(features.shape[2:]):
            other_axes = tuple(2 + a for a in range(3) if a != axis)
            marginal = foreground.sum(dim=other_axes)
            coordinate = torch.arange(length, device=features.device, dtype=features.dtype)
            coordinate = coordinate / max(length - 1, 1)
            components.append((marginal * coordinate).sum(dim=-1, keepdim=True) / denominator)
        centroid = torch.cat(components, dim=-1)
        size = mass / weights.shape[-1]
        confidence = weights.square().sum(dim=-1, keepdim=True) / denominator
        return OrganNodes(z0, mass, centroid, size, confidence)
