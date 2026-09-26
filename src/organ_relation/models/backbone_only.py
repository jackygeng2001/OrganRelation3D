"""Isolated overfit diagnostic, NOT the OrganRelation3D scientific model."""
from typing import NamedTuple

import torch
from torch import Tensor, nn

from .segmentor import Segmentor
from .segmentor_config import SegmentorConfig


class BackboneOnlyOutput(NamedTuple):
    final_logits: Tensor


class BackboneOnly(nn.Module):
    """Reuse the full model's exact encoder/decoder initialization and operators.

    Construct on CPU after the same seed as Segmentor, before device transfer.
    Building the reference consumes the same RNG sequence, including discarded
    graph modules. Only encoder/decoder are retained, registered and optimized.
    The full reference config is required to reproduce that sequence.
    """

    def __init__(self, config: SegmentorConfig):
        super().__init__()
        reference = Segmentor(config)
        self.config = config
        self.encoder = reference.encoder
        self.decoder = reference.decoder

    def forward(self, image: Tensor) -> BackboneOnlyOutput:
        if not isinstance(image, Tensor) or image.ndim != 5 or min(image.shape) < 1 or image.shape[1] != 1:
            raise ValueError('image must have positive shape [B,1,D,H,W]')
        parameter = next(self.parameters())
        if (image.dtype not in (torch.float32, torch.float64)
                or image.dtype != parameter.dtype or image.device != parameter.device):
            raise ValueError('image and parameters require matching FP32/FP64 dtype and device')
        if torch.is_autocast_enabled(image.device.type) or not torch.isfinite(image).all():
            raise ValueError('finite image and disabled autocast required')
        features = self.encoder(image)
        logits = self.decoder(features.deepest, features.skips)
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite final logits')
        return BackboneOnlyOutput(logits)
