"""Light training-only augmentation for BraTS region volumes (user-approved 2026-11-06).

flip_intensity_v1: each spatial axis is mirrored with probability p (image and
label together); each modality's nonzero voxels are scaled by U(1-s, 1+s) and
shifted by U(-o, o) in z-score units. Zero (outside-brain) voxels stay zero.
Random draws come from a generator seeded by (training seed, global step), so
no augmentation state needs checkpointing, resume reproduces every step, and no
other RNG stream is consumed. Labels never affect the image transform; dev,
validation and inference inputs are never augmented.
"""
import hashlib
import math

import torch

KIND = 'flip_intensity_v1'
_KEYS = {'kind', 'flip_probability', 'intensity_scale', 'intensity_shift'}


def validate_augmentation(spec):
    if spec is None:
        return
    if not isinstance(spec, dict) or set(spec) != _KEYS or spec['kind'] != KIND:
        raise ValueError(f'augmentation must be null or {{kind: {KIND}, flip_probability, intensity_scale, intensity_shift}}')
    for key, upper in (('flip_probability', 1.0), ('intensity_scale', 1.0), ('intensity_shift', math.inf)):
        value = spec[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or value < 0 or value > upper or (key == 'intensity_scale' and value >= 1):
            raise ValueError(f'augmentation {key} out of range')


def step_generator(seed, step):
    """Independent CPU generator for one optimizer step (global_step counted from 1)."""
    digest = hashlib.sha256(f'{KIND}:{seed}:{step}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], 'little') & ((1 << 63) - 1))


def draw(spec, generator, channels):
    flip = (torch.rand(3, generator=generator, dtype=torch.float64) < spec['flip_probability']).tolist()
    scale = 1 + (2 * torch.rand(channels, generator=generator, dtype=torch.float64) - 1) * spec['intensity_scale']
    shift = (2 * torch.rand(channels, generator=generator, dtype=torch.float64) - 1) * spec['intensity_shift']
    return dict(flip=flip, scale=scale.tolist(), shift=shift.tolist())


def apply(image, label, params):
    """image [B,C,D,H,W] float, label [B,D,H,W]; returns new tensors, inputs untouched."""
    if image.ndim != 5 or label.ndim != 4 or image.shape[0] != label.shape[0] or image.shape[2:] != label.shape[1:]:
        raise ValueError('augmentation expects image [B,C,D,H,W] and label [B,D,H,W] on the same grid')
    if len(params['scale']) != image.shape[1] or len(params['shift']) != image.shape[1]:
        raise ValueError('one intensity scale/shift per modality required')
    view = (1, image.shape[1], 1, 1, 1)
    scale = torch.tensor(params['scale'], device=image.device, dtype=image.dtype).view(view)
    shift = torch.tensor(params['shift'], device=image.device, dtype=image.dtype).view(view)
    image = torch.where(image != 0, image * scale + shift, image)
    axes = [axis for axis, flip in enumerate(params['flip']) if flip]
    if axes:
        image = image.flip([2 + axis for axis in axes])
        label = label.flip([1 + axis for axis in axes])
    return image, label
