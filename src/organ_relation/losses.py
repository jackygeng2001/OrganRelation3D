"""WT/TC/ET 三区域联合损失：概率 BCE + 逐病例 soft Dice，粗分支加权，最终分支可加层级一致性项（式 27–33）。"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


class BranchLoss(NamedTuple):
    ce: Tensor  # [B] 三区域、全体素平均的概率 BCE（字段名沿用历史日志格式）
    dice_per_class: Tensor  # [B,3] 依次为 WT/TC/ET 的 soft Dice
    dice_loss: Tensor  # [B] 1 - 三区域 soft Dice 平均
    segmentation: Tensor  # [B] ce + dice_loss，两项系数均为 1


class JointLossResult(NamedTuple):
    total: Tensor  # Scalar, mean over batch AFTER combining per-case branches.
    per_case: Tensor  # [B], final.segmentation + lambda_c * coarse.segmentation.
    coarse: BranchLoss
    final: BranchLoss


REGION_NAMES = ('WT', 'TC', 'ET')


def region_targets(label: Tensor) -> Tensor:
    """Canonical BraTS 0/1/2/3 -> overlapping WT/TC/ET. Never used by models."""
    if label.ndim != 4 or label.dtype != torch.int64 or ((label < 0) | (label > 3)).any():
        raise ValueError('canonical int64 [B,D,H,W] labels in 0..3 required')
    return torch.stack((label > 0, (label == 1) | (label == 3), label == 3), 1)


class RegionLossResult(NamedTuple):
    total: Tensor  # Scalar batch mean of per_case.
    per_case: Tensor  # [B] final + lambda_c*coarse + lambda_hier*hierarchy, eq. (33).
    coarse: BranchLoss
    final: BranchLoss
    hierarchy: Tensor  # [B] eq. (32) on FINAL probabilities, before lambda_hier.


class ContainmentLossResult(NamedTuple):
    total: Tensor  # Scalar batch mean of per_case.
    per_case: Tensor  # [B] 式 (33) 加 lambda_contain * containment。
    coarse: BranchLoss
    final: BranchLoss
    hierarchy: Tensor  # [B] eq. (32) on FINAL probabilities, before lambda_hier.
    containment: Tensor  # [B] 包含比例损失，未乘 lambda_contain。


def containment_targets(target: Tensor) -> tuple[Tensor, Tensor]:
    """真值包含比例 [B,2]（|TC|/|WT|、|ET|/|TC|）与有效标记（外层真值非空）。只用于损失，不进入模型。"""
    volume = target.flatten(2).sum(2).to(torch.float64)  # [B,3] WT/TC/ET 真值体素数
    outer, inner = volume[:, :2], volume[:, 1:]
    valid = outer > 0
    return (inner / outer.clamp_min(1)).clamp(0, 1), valid


def containment_loss(logits: Tensor, target: Tensor) -> Tensor:
    """[B]：两条相邻嵌套边（WT→TC、TC→ET）读出的监督，对外层真值非空的边取平均；都无效时为 0。

    logits [B,2]：包含比例 logit，软目标二元交叉熵（目标为真值体素比 |内层|/|外层|）。
    logits [B,2,2]：最后一维为（比例 logit，存在 logit）；存在 logit 另加二元交叉熵（目标为内层真值是否非空），两项相加。
    """
    ratio, valid = containment_targets(target)
    bce = functional.binary_cross_entropy_with_logits
    if logits.ndim == 2:
        loss = bce(logits, ratio.to(logits.dtype), reduction='none')
    else:
        loss = (bce(logits[..., 0], ratio.to(logits.dtype), reduction='none')
                + bce(logits[..., 1], (ratio > 0).to(logits.dtype), reduction='none'))
    return (loss * valid).sum(1) / valid.sum(1).clamp_min(1)


def hierarchy_violation(logits: Tensor) -> Tensor:
    """Eq. (32): (1/(2|Omega_I|)) sum_x [ReLU(P_TC-P_WT) + ReLU(P_ET-P_TC)], per case."""
    probability = logits.sigmoid()
    wt, tc, et = probability[:, 0], probability[:, 1], probability[:, 2]
    return (torch.relu(tc - wt) + torch.relu(et - tc)).flatten(1).mean(1) / 2


class RegionJointLoss(nn.Module):
    """Revised method (27)-(31) and hierarchy revision (32)-(33); probability BCE(+epsilon).

    BranchLoss.ce is the shared engine container's historical field name; here
    it stores binary CE averaged across all THREE regions and all voxels.
    lambda_hier=None keeps the earlier (27)-(32) objective and result type
    bit-for-bit; a number (>=0) adds the final-only hierarchy term of (33).
    """
    def __init__(self, *, epsilon, lambda_c, align_corners, lambda_hier=None, lambda_contain=None):
        super().__init__()
        if (isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0
                or isinstance(lambda_c, bool) or not math.isfinite(lambda_c) or lambda_c <= 0
                or type(align_corners) is not bool):
            raise ValueError('positive finite epsilon/lambda_c and explicit align_corners required')
        if lambda_hier is not None and (type(lambda_hier) not in (int, float)
                                        or not math.isfinite(lambda_hier) or lambda_hier < 0):
            raise ValueError('lambda_hier must be null or a finite number >= 0')
        self.epsilon, self.lambda_c, self.align_corners = epsilon, lambda_c, align_corners
        self.lambda_hier = None if lambda_hier is None else float(lambda_hier)
        # 关系推理 v2 的包含比例监督（2026-11-11）：缺省 None 时目标函数与返回结构不变。
        if lambda_contain is not None and (type(lambda_contain) not in (int, float) or not math.isfinite(lambda_contain)
                                           or lambda_contain < 0 or lambda_hier is None):
            raise ValueError('lambda_contain must be null or a finite number >= 0, and requires lambda_hier')
        self.lambda_contain = None if lambda_contain is None else float(lambda_contain)

    def branch(self, logits, target):
        probability = logits.sigmoid()
        target = target.to(dtype=logits.dtype)
        bce = -(target * (probability+self.epsilon).log()
                + (1-target) * (1-probability+self.epsilon).log()).flatten(1).mean(1)
        intersection = (probability*target).flatten(2).sum(2)
        dice = (2*intersection+self.epsilon)/(probability.flatten(2).sum(2)+target.flatten(2).sum(2)+self.epsilon)
        dice_loss = 1-dice.mean(1)
        return BranchLoss(bce,dice,dice_loss,bce+dice_loss)

    def align_coarse(self, logits, label):
        return functional.interpolate(logits,size=label.shape[1:],mode='trilinear',align_corners=self.align_corners)

    def forward(self, coarse_logits, final_logits, label, containment_logits=None):
        target = region_targets(label)
        if (self.lambda_contain is None) != (containment_logits is None):
            raise ValueError('containment logits and loss.lambda_contain must be configured together')
        for logits in (coarse_logits, final_logits):
            if (logits.ndim != 5 or min(logits.shape) < 1 or logits.shape[:2] != (label.shape[0],3)
                    or logits.dtype not in (torch.float32,torch.float64)
                    or logits.device != label.device or not torch.isfinite(logits).all()):
                raise ValueError('finite three-region FP32/FP64 logits required')
        if final_logits.shape != target.shape or coarse_logits.dtype != final_logits.dtype:
            raise ValueError('final logits must match original GT grid and coarse dtype')
        if torch.is_autocast_enabled(final_logits.device.type):
            raise ValueError('autocast disabled required')
        limits = torch.finfo(final_logits.dtype)
        for value in (self.epsilon, self.lambda_c):
            if not limits.tiny * limits.eps <= value <= limits.max:
                raise ValueError('epsilon/lambda_c must be positive and representable in logits dtype')
        final = self.branch(final_logits,target)
        coarse = self.branch(self.align_coarse(coarse_logits,label),target)
        per_case = final.segmentation+self.lambda_c*coarse.segmentation
        if self.lambda_hier is None:
            if not torch.isfinite(per_case).all():
                raise ValueError('nonfinite region loss')
            return JointLossResult(per_case.mean(),per_case,coarse,final)
        hierarchy = hierarchy_violation(final_logits)  # Final branch only; coarse keeps L_seg.
        per_case = per_case+self.lambda_hier*hierarchy
        if self.lambda_contain is not None:
            if (containment_logits.shape not in ((label.shape[0],2),(label.shape[0],2,2))
                    or containment_logits.dtype != final_logits.dtype or not torch.isfinite(containment_logits).all()):
                raise ValueError('finite [B,2] or [B,2,2] containment logits in the final-logit dtype required')
            containment = containment_loss(containment_logits,target)
            per_case = per_case+self.lambda_contain*containment
            if not torch.isfinite(per_case).all():
                raise ValueError('nonfinite region loss')
            return ContainmentLossResult(per_case.mean(),per_case,coarse,final,hierarchy,containment)
        if not torch.isfinite(per_case).all():
            raise ValueError('nonfinite region loss')
        return RegionLossResult(per_case.mean(),per_case,coarse,final,hierarchy)
