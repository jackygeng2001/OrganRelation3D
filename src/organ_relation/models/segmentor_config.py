"""Explicit full-forward configuration; no formal experiment defaults or torch import."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from .backbone_config import BackboneConfig


@dataclass(frozen=True)
class SegmentorConfig:
    backbone: BackboneConfig
    coarse_bias: bool
    fusion_bias: bool
    epsilon: float
    relation_channels: int
    rounds: int
    attention_channels: int
    content_channels: int
    beta_init: float

    def __post_init__(self):
        if isinstance(self.backbone, dict):
            object.__setattr__(self, 'backbone', BackboneConfig(**self.backbone))
        if not isinstance(self.backbone, BackboneConfig):
            raise ValueError('backbone must be a BackboneConfig or its field dictionary')
        for name in ('coarse_bias', 'fusion_bias'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be an explicit boolean')
        for name in ('relation_channels', 'rounds', 'attention_channels', 'content_channels'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('epsilon', 'beta_init'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{name} must be a finite number')
        if self.epsilon <= 0:
            raise ValueError('epsilon must be positive')

    def to_dict(self):
        return asdict(self)
