"""Frozen probability CE + per-case foreground Dice, with explicit configuration."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BranchLoss(NamedTuple):
    ce: Tensor  # [B], includes background, averaged over this case's voxels.
    dice_per_class: Tensor  # [B,15], class axis corresponds to labels 1..15.
    dice_loss: Tensor  # [B], 1 - mean over all 15 foreground classes.
    segmentation: Tensor  # [B], ce + dice_loss, both coefficients exactly 1.


class JointLossResult(NamedTuple):
    total: Tensor  # Scalar, mean over batch AFTER combining per-case branches.
    per_case: Tensor  # [B], final.segmentation + lambda_c * coarse.segmentation.
    coarse: BranchLoss
    final: BranchLoss


class JointLoss(nn.Module):
    """Separate supervised module; never passes label to the Segmentor.

    No default epsilon, lambda_c or align_corners: their formal values have not
    been frozen. The caller must use the same epsilon convention as the model.
    Returns only scalar/small per-case statistics, not full probability volumes.
    FP32/FP64 outside autocast are supported. Config is not in state_dict.
    """

    def __init__(self, *, epsilon: float, lambda_c: float, align_corners: bool):
        super().__init__()
        for name, value in (('epsilon', epsilon), ('lambda_c', lambda_c)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if type(align_corners) is not bool:
            raise ValueError('align_corners must be an explicit boolean')
        self.epsilon = float(epsilon)
        self.lambda_c = float(lambda_c)
        self.align_corners = align_corners

    def extra_repr(self) -> str:
        return f'epsilon={self.epsilon!r}, lambda_c={self.lambda_c!r}, align_corners={self.align_corners!r}'

    def _validate(self, coarse_logits: Tensor, final_logits: Tensor, label: Tensor) -> None:
        for name, value in (('coarse_logits', coarse_logits), ('final_logits', final_logits)):
            if not isinstance(value, Tensor) or value.ndim != 5 or min(value.shape) < 1 or value.shape[1] != 16:
                raise ValueError(f'{name} must have positive shape [B,16,D,H,W]')
            if value.dtype not in (torch.float32, torch.float64):
                raise ValueError(f'{name} requires float32 or float64; AMP is not validated')
        if coarse_logits.shape[0] != final_logits.shape[0] or coarse_logits.device != final_logits.device or coarse_logits.dtype != final_logits.dtype:
            raise ValueError('coarse/final logits must share batch, device and dtype')
        if not isinstance(label, Tensor) or tuple(label.shape) != (final_logits.shape[0], *final_logits.shape[2:]):
            raise ValueError('label must have shape [B,D,H,W] matching final_logits exactly')
        if label.dtype != torch.int64:
            raise ValueError('label must use torch.int64 class indices, without one-hot encoding or gradients')
        if label.device != final_logits.device:
            raise ValueError('label and logits must share device')
        if torch.is_autocast_enabled(final_logits.device.type):
            raise ValueError('JointLoss requires autocast disabled until AMP is validated')
        for name, value in (('coarse_logits', coarse_logits), ('final_logits', final_logits)):
            if not torch.isfinite(value).all():
                raise ValueError(f'{name} must contain only finite values')
        if ((label < 0) | (label > 15)).any():
            raise ValueError('label class indices must be in 0..15; no ignore_index is defined')
        limits = torch.finfo(final_logits.dtype)
        for name, value in (('epsilon', self.epsilon), ('lambda_c', self.lambda_c)):
            if not limits.tiny * limits.eps <= value <= limits.max:
                raise ValueError(f'{name} must be representable and positive in the logits dtype')

    def _branch(self, probabilities: Tensor, label: Tensor, target_count: Tensor) -> BranchLoss:
        flat = probabilities.flatten(start_dim=2)  # [B,16,N]
        indices = label.flatten(start_dim=1)  # [B,N]
        matched = flat.gather(1, indices.unsqueeze(1)).squeeze(1)  # S at the true class.
        ce = -torch.log(matched + self.epsilon).mean(dim=1)
        # Equivalent to sum_x S_c(x)*T_c(x), without dense [B,16,D,H,W] one-hot GT.
        intersection = probabilities.new_zeros(probabilities.shape[0], 16).scatter_add(1, indices, matched)
        predicted_count = flat.sum(dim=2)
        dice = (2 * intersection[:, 1:] + self.epsilon) / (
            predicted_count[:, 1:] + target_count[:, 1:] + self.epsilon)
        dice_loss = 1 - dice.mean(dim=1)
        return BranchLoss(ce, dice, dice_loss, ce + dice_loss)

    def forward(self, coarse_logits: Tensor, final_logits: Tensor, label: Tensor) -> JointLossResult:
        self._validate(coarse_logits, final_logits, label)
        # Resize LOGITS exactly once; neither final_logits nor label is resized.
        coarse_up = functional.interpolate(coarse_logits, size=tuple(final_logits.shape[2:]),
                                           mode='trilinear', align_corners=self.align_corners)
        if not torch.isfinite(coarse_up).all():
            raise ValueError('interpolated coarse logits must contain only finite values')
        # Accumulate GT counts as int64 before conversion, avoiding repeated FP32
        # unit increments for large classes. Labels never carry gradients.
        indices = label.flatten(start_dim=1)
        target_count = label.new_zeros(label.shape[0], 16).scatter_add(
            1, indices, torch.ones_like(indices)).to(dtype=final_logits.dtype)
        coarse = self._branch(torch.softmax(coarse_up, dim=1), label, target_count)
        final = self._branch(torch.softmax(final_logits, dim=1), label, target_count)
        per_case = final.segmentation + self.lambda_c * coarse.segmentation
        total = per_case.mean()
        if not torch.isfinite(total):
            raise ValueError('joint loss produced a non-finite value')
        return JointLossResult(total, per_case, coarse, final)
