"""Independent sigmoid spatial matching and updated-node content writeback."""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class NodeToSpaceResult(NamedTuple):
    """Optional read-only diagnostics; all tensors retain autograd history."""

    G: Tensor  # [B,Cg,Df,Hf,Wf]
    Q: Tensor  # [B,nodes,da], updated-node queries
    K: Tensor  # [B,N,da], keys from original F; W axis varies fastest
    V: Tensor  # [B,nodes,Cg], updated-node contents
    A: Tensor  # [B,nodes,N], independent gates, NOT segmentation probabilities


def _finite(value: Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')


class NodeToSpace(nn.Module):
    """METHOD_SPEC Node-to-Space, without residual fusion or supervision.

    The caller supplies original deepest F and post-reasoning zK. Their source
    cannot be inferred from shape alone. No coarse probabilities, labels or
    external attention enter forward. FP32/FP64 outside autocast are supported.
    Standard Linear initialization and zero beta are engineering initializers,
    not frozen formal experiment settings; beta remains independently trainable.
    """

    def __init__(self, channels: int, *, attention_channels: int, content_channels: int,
                 beta_init: float = 0.0, node_count: int = 15, key_channels: int | None = None):
        super().__init__()
        for name, value in (('channels', channels), ('attention_channels', attention_channels),
                            ('content_channels', content_channels)):
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(beta_init, bool) or not isinstance(beta_init, (int, float)) or not math.isfinite(beta_init):
            raise ValueError('beta_init must be a finite number')
        key_channels = channels if key_channels is None else key_channels
        if type(node_count) is not int or node_count < 1 or type(key_channels) is not int or key_channels < 1:
            raise ValueError('positive node_count/key_channels required')
        self.node_count, self.key_channels = node_count, key_channels
        self.channels = channels
        self.attention_channels = attention_channels
        self.content_channels = content_channels
        self.W_Q = nn.Linear(channels, attention_channels, bias=False)
        self.W_K = nn.Linear(key_channels, attention_channels, bias=False)
        self.W_V = nn.Linear(channels, content_channels, bias=False)
        if abs(beta_init) > torch.finfo(self.W_Q.weight.dtype).max:
            raise ValueError('beta_init must be representable in the parameter dtype')
        self.beta = nn.Parameter(self.W_Q.weight.new_full((node_count,), float(beta_init)))
        _finite(self.beta, 'beta_init in parameter dtype')

    def extra_repr(self) -> str:
        return (f'channels={self.channels}, attention_channels={self.attention_channels}, '
                f'content_channels={self.content_channels}')

    def _validate(self, features: Tensor, zK: Tensor, *, reduced_features: bool = False) -> None:
        if not isinstance(features, Tensor) or features.ndim != 5:
            raise ValueError('features must have shape [B,C,Df,Hf,Wf]')
        if min(features.shape) < 1 or features.shape[1] != self.key_channels:
            raise ValueError('features require positive sizes and configured channels')
        if not isinstance(zK, Tensor) or tuple(zK.shape) != (features.shape[0], self.node_count, self.channels):
            raise ValueError(f'zK must have shape [B,{self.node_count},C], matching the feature batch/channels')
        # reduced_features: whole-network mixed precision stores the decoder skip
        # volume in bfloat16 while nodes and parameters stay FP32.
        dtype = zK.dtype if reduced_features and features.dtype == torch.bfloat16 else features.dtype
        if dtype not in (torch.float32, torch.float64) or (dtype != features.dtype and dtype != torch.float32):
            raise ValueError('NodeToSpace requires float32 or float64; AMP is not validated')
        if zK.device != features.device or zK.dtype != dtype:
            raise ValueError('features and zK must share device and dtype')
        if torch.is_autocast_enabled(features.device.type):
            raise ValueError('NodeToSpace requires autocast disabled until AMP is validated')
        if any(p.device != features.device or p.dtype != dtype for p in self.parameters()):
            raise ValueError('module parameters and inputs must share device and dtype')
        _finite(features, 'features')
        _finite(zK, 'zK')

    def forward(self, features: Tensor, zK: Tensor, *, return_diagnostics: bool = False) -> Tensor | NodeToSpaceResult:
        return self._run(features, zK, return_diagnostics, None)

    def forward_with_scale_weights(self, features: Tensor, zK: Tensor, region_weights: Tensor,
                                   *, return_diagnostics: bool = False) -> Tensor | NodeToSpaceResult:
        """Revised decoder: multiply each region's content by its learned scale weight."""
        return self._run(features, zK, return_diagnostics, region_weights)

    def _run(self, features, zK, return_diagnostics, region_weights):
        self._validate(features, zK)
        if type(return_diagnostics) is not bool:
            raise ValueError('return_diagnostics must be a boolean')
        spatial = features.shape[2:]
        Q = self.W_Q(zK)
        K = self.W_K(features.flatten(start_dim=2).transpose(1, 2))
        V = self.W_V(zK)
        for name, value in (('Q', Q), ('K', K), ('V', V)):
            _finite(value, name)
        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.attention_channels)
        scores = scores + self.beta.view(1, self.node_count, 1)
        _finite(scores, 'matching scores')
        A = torch.sigmoid(scores)
        # Sum organ contents at each position; never allocate [B,nodes,Cg,D,H,W].
        weighted = A
        if region_weights is not None:
            if region_weights.shape != (features.shape[0], self.node_count, 1) or region_weights.dtype != features.dtype or region_weights.device != features.device:
                raise ValueError('region_weights must match [B,node_count,1], device and dtype')
            _finite(region_weights, 'region_weights')
            weighted = A * region_weights
        G = torch.bmm(V.transpose(1, 2), weighted).reshape(features.shape[0], self.content_channels, *spatial)
        _finite(G, 'G')
        return NodeToSpaceResult(G, Q, K, V, A) if return_diagnostics else G
