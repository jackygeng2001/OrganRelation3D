"""Frozen probability CE + per-case foreground Dice, with explicit configuration."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BranchLoss(NamedTuple):
    ce: Tensor  # [B], selected per-case reduction; voxel mean by default.
    dice_per_class: Tensor  # [B,15], class axis corresponds to labels 1..15.
    dice_loss: Tensor  # [B], 1 - mean over all 15 foreground classes.
    segmentation: Tensor  # [B], ce + dice_loss, both coefficients exactly 1.


class JointLossResult(NamedTuple):
    total: Tensor  # Scalar, mean over batch AFTER combining per-case branches.
    per_case: Tensor  # [B], final.segmentation + lambda_c * coarse.segmentation.
    coarse: BranchLoss
    final: BranchLoss


class FinalLossResult(NamedTuple):
    total: Tensor
    per_case: Tensor
    final: BranchLoss


CE_REDUCTION_MODES = ('voxel_mean', 'foreground_background_balanced')
FOREGROUND_CE_REDUCTIONS = ('voxel_mean', 'class_macro_mean')


def validate_foreground_reduction(reduction, ce_reduction_mode):
    if reduction not in FOREGROUND_CE_REDUCTIONS:
        raise ValueError('unsupported foreground_ce_reduction')
    if reduction == 'class_macro_mean' and ce_reduction_mode != 'foreground_background_balanced':
        raise ValueError('class_macro_mean requires foreground_background_balanced CE')
    return reduction


def foreground_class_ce_means(voxel_loss: Tensor, indices: Tensor):
    """[B,N] -> means [B,15], macro [B], present count [B].

    Absent classes are NaN in diagnostic means, excluded from macro. No dense
    one-hot GT; integer counts and per-class sums use scatter_add. A wholly
    absent foreground is undefined (NaN), rejected by balanced loss as before.
    """
    counts = indices.new_zeros(indices.shape[0], 16).scatter_add(
        1, indices, torch.ones_like(indices))[:, 1:]
    sums = voxel_loss.new_zeros(indices.shape[0], 16).scatter_add(1, indices, voxel_loss)[:, 1:]
    present = counts > 0
    means = sums / counts.clamp_min(1)
    number = present.sum(1)
    macro = torch.where(present, means, 0).sum(1) / number.clamp_min(1)
    return means.masked_fill(~present, float('nan')), macro.masked_fill(number == 0, float('nan')), number


def resolve_ce_weights(background=0.5, foreground=0.5):
    """Validate, never renormalize. Tolerance only accommodates float rounding."""
    for value in (background, foreground):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError('CE group weights must be finite numbers strictly between 0 and 1')
    if not math.isclose(background + foreground, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError('CE group weights must sum to 1')
    return float(background), float(foreground)


def foreground_background_ce_means(voxel_loss: Tensor, foreground: Tensor) -> tuple[Tensor, Tensor]:
    """Per-case group means from [B,N] losses; NaN denotes an absent group.

    Absence is exposed for logging, never silently reweighted in training.
    Counts are accumulated as integers; epsilon is only inside log(S+epsilon).
    """
    fg_count = foreground.sum(dim=1)
    bg_count = (~foreground).sum(dim=1)
    fg = torch.where(foreground, voxel_loss, 0).sum(dim=1) / fg_count.clamp_min(1)
    bg = torch.where(~foreground, voxel_loss, 0).sum(dim=1) / bg_count.clamp_min(1)
    return (bg.masked_fill(bg_count == 0, float('nan')),
            fg.masked_fill(fg_count == 0, float('nan')))


class SegmentationLoss(nn.Module):
    """Final-only CE + foreground Dice; voxel-mean CE unless explicitly selected.

    No resizing, coarse branch or auxiliary coefficient. Supervision remains
    outside the model; all 15 foreground classes participate in each case.
    """

    def __init__(self, *, epsilon: float, ce_reduction_mode: str = 'voxel_mean',
                 ce_background_weight: float = 0.5, ce_foreground_weight: float = 0.5,
                 foreground_ce_reduction: str = 'voxel_mean'):
        super().__init__()
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('epsilon must be a finite positive number')
        self.epsilon = float(epsilon)
        if ce_reduction_mode not in CE_REDUCTION_MODES:
            raise ValueError('unsupported ce_reduction_mode')
        self.ce_reduction_mode = ce_reduction_mode
        self.foreground_ce_reduction = validate_foreground_reduction(foreground_ce_reduction, ce_reduction_mode)
        self.ce_background_weight, self.ce_foreground_weight = resolve_ce_weights(
            ce_background_weight, ce_foreground_weight)

    def forward(self, final_logits: Tensor, label: Tensor) -> FinalLossResult:
        if not isinstance(final_logits, Tensor) or final_logits.ndim != 5 or min(final_logits.shape) < 1 or final_logits.shape[1] != 16:
            raise ValueError('final_logits must have positive shape [B,16,D,H,W]')
        if final_logits.dtype not in (torch.float32, torch.float64):
            raise ValueError('final_logits requires float32 or float64')
        if not isinstance(label, Tensor) or tuple(label.shape) != (final_logits.shape[0], *final_logits.shape[2:]):
            raise ValueError('label must match final_logits [B,D,H,W]')
        if label.dtype != torch.int64 or label.device != final_logits.device:
            raise ValueError('label requires int64 on the logits device')
        if torch.is_autocast_enabled(final_logits.device.type):
            raise ValueError('SegmentationLoss requires autocast disabled')
        if not torch.isfinite(final_logits).all() or ((label < 0) | (label > 15)).any():
            raise ValueError('finite logits and labels in 0..15 required')
        limits = torch.finfo(final_logits.dtype)
        if not limits.tiny * limits.eps <= self.epsilon <= limits.max:
            raise ValueError('epsilon must be representable and positive in the logits dtype')
        indices = label.flatten(start_dim=1)
        target_count = label.new_zeros(label.shape[0], 16).scatter_add(
            1, indices, torch.ones_like(indices)).to(dtype=final_logits.dtype)
        final = _segmentation_branch(torch.softmax(final_logits, dim=1), label, target_count,
                                     self.epsilon, self.ce_reduction_mode,
                                     self.ce_background_weight, self.ce_foreground_weight, self.foreground_ce_reduction)
        total = final.segmentation.mean()
        if not torch.isfinite(total):
            raise ValueError('segmentation loss produced a non-finite value')
        return FinalLossResult(total, final.segmentation, final)


def _segmentation_branch(probabilities: Tensor, label: Tensor, target_count: Tensor, epsilon: float,
                         ce_reduction_mode: str = 'voxel_mean',
                         ce_background_weight: float = 0.5, ce_foreground_weight: float = 0.5,
                         foreground_ce_reduction: str = 'voxel_mean') -> BranchLoss:
    flat = probabilities.flatten(start_dim=2)  # [B,16,N]
    indices = label.flatten(start_dim=1)  # [B,N]
    matched = flat.gather(1, indices.unsqueeze(1)).squeeze(1)  # S at the true class.
    if ce_reduction_mode == 'voxel_mean':
        ce = -torch.log(matched + epsilon).mean(dim=1)  # Historical operation order unchanged.
    elif ce_reduction_mode == 'foreground_background_balanced':
        bg, fg = foreground_background_ce_means(-torch.log(matched + epsilon), indices > 0)
        if foreground_ce_reduction == 'class_macro_mean':
            _, fg, _ = foreground_class_ce_means(-torch.log(matched + epsilon), indices)
        if not (torch.isfinite(bg).all() and torch.isfinite(fg).all()):
            raise ValueError('balanced CE requires both background and foreground in every case')
        ce = ce_background_weight * bg + ce_foreground_weight * fg
    else:
        raise ValueError('unsupported ce_reduction_mode')
    # Equivalent to sum_x S_c(x)*T_c(x), without dense [B,16,D,H,W] one-hot GT.
    intersection = probabilities.new_zeros(probabilities.shape[0], 16).scatter_add(1, indices, matched)
    predicted_count = flat.sum(dim=2)
    dice = (2 * intersection[:, 1:] + epsilon) / (
        predicted_count[:, 1:] + target_count[:, 1:] + epsilon)
    dice_loss = 1 - dice.mean(dim=1)
    return BranchLoss(ce, dice, dice_loss, ce + dice_loss)


class JointLoss(nn.Module):
    """Separate supervised module; never passes label to the Segmentor.

    Load epsilon/lambda_c/align_corners explicitly from the baseline config.
    CE defaults to historical voxel mean; balanced reduction requires opt-in.
    The caller must use the same epsilon as the model.
    Returns only scalar/small per-case statistics, not full probability volumes.
    FP32/FP64 outside autocast are supported. Config is not in state_dict.
    """

    def __init__(self, *, epsilon: float, lambda_c: float, align_corners: bool,
                 ce_reduction_mode: str = 'voxel_mean',
                 ce_background_weight: float = 0.5, ce_foreground_weight: float = 0.5,
                 foreground_ce_reduction: str = 'voxel_mean'):
        super().__init__()
        for name, value in (('epsilon', epsilon), ('lambda_c', lambda_c)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        if type(align_corners) is not bool:
            raise ValueError('align_corners must be an explicit boolean')
        self.epsilon = float(epsilon)
        self.lambda_c = float(lambda_c)
        self.align_corners = align_corners
        if ce_reduction_mode not in CE_REDUCTION_MODES:
            raise ValueError('unsupported ce_reduction_mode')
        self.ce_reduction_mode = ce_reduction_mode
        self.foreground_ce_reduction = validate_foreground_reduction(foreground_ce_reduction, ce_reduction_mode)
        self.ce_background_weight, self.ce_foreground_weight = resolve_ce_weights(
            ce_background_weight, ce_foreground_weight)

    def extra_repr(self) -> str:
        return (f'epsilon={self.epsilon!r}, lambda_c={self.lambda_c!r}, align_corners={self.align_corners!r}, '
                f'ce_reduction_mode={self.ce_reduction_mode!r}')

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
        return _segmentation_branch(probabilities, label, target_count, self.epsilon, self.ce_reduction_mode,
                                     self.ce_background_weight, self.ce_foreground_weight, self.foreground_ce_reduction)

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
