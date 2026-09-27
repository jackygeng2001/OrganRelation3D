"""External whole-volume MONAI reference; not the proposed model or its ablation."""
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ReferenceOutput(NamedTuple):
    final_logits: Tensor


def padding_geometry(shape):
    """Minimal high-end padding, independent of image values and labels."""
    original = list(shape)
    high = [(-n) % 16 for n in original]
    return dict(original_shape=original, padding_dhw=[[0, p] for p in high],
                padded_shape=[n + p for n, p in zip(original, high)])


class MonaiReferenceUNet(nn.Module):
    def __init__(self, constructor, input_processing):
        super().__init__()
        from monai.networks.nets import UNet
        if input_processing != dict(clip_hu=[-1000, 1000], scale_to=[-1, 1],
                                    padding_multiple=16, padding_side='high', padding_value=-1):
            raise ValueError('MONAI reference requires the explicit HU scaling and padding protocol')
        self.constructor = dict(constructor)
        self.input_processing = dict(input_processing)
        self.network = UNet(**constructor)

    def prepare_image(self, image):
        # The shared loader has already applied each NIfTI's physical HU scaling.
        normalized = image.clamp(-1000, 1000) / 1000
        plan = padding_geometry(image.shape[2:])
        pad = tuple(v for pair in reversed(plan['padding_dhw']) for v in pair)
        padded = F.pad(normalized, pad, mode='constant', value=-1)
        if image.is_contiguous(memory_format=torch.channels_last_3d):
            padded = padded.contiguous(memory_format=torch.channels_last_3d)
        return padded

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 1 or image.dtype != torch.float32:
            raise ValueError('MONAI reference requires FP32 [B,1,D,H,W]')
        if torch.is_autocast_enabled(image.device.type) or not torch.isfinite(image).all():
            raise ValueError('finite FP32 image and disabled autocast required')
        shape = image.shape[2:]
        logits = self.network(self.prepare_image(image))  # ONE whole-volume forward.
        # Remove only the added boundary, never crop original scan coverage.
        logits = logits[:, :, :shape[0], :shape[1], :shape[2]]
        if tuple(logits.shape) != (image.shape[0], 16, *shape) or not torch.isfinite(logits).all():
            raise ValueError('invalid MONAI reference output')
        return ReferenceOutput(logits)
