"""单病例样本容器。

BraTS 数据集（data/brats.py）返回这个结构。data/brats.py 的源码哈希是 npy 缓存键的一部分，
因此它的导入路径保持不变，这里只保留容器本身。
"""
from typing import NamedTuple

import torch


class FullScanSample(NamedTuple):
    image: torch.Tensor  # [4,D,H,W] float32，四模态 MRI，标签不参与任何图像处理
    label: torch.Tensor  # [D,H,W] int64，只用于监督与评估
    metadata: dict  # 仿射、裁剪框、归一化统计等元数据
