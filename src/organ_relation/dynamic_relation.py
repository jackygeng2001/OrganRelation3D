"""Dynamic directed organ reasoning and the explicit METHOD_SPEC GRU.

Edge axes are always [batch, sender, receiver]. There are exactly 15 nodes;
only their semantic states change between rounds. No labels enter this module.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


NODE_COUNT = 15


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _check_tensor(value: Tensor, name: str, channels: int, reference: Tensor | None = None) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f'{name} must have shape [B,15,{channels}]')
    if value.shape[0] < 1 or tuple(value.shape[1:]) != (NODE_COUNT, channels):
        raise ValueError(f'{name} must have shape [B,15,{channels}] with B>=1')
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} requires float32 or float64; AMP is not validated')
    if reference is not None and (
        value.shape[0] != reference.shape[0]
        or value.device != reference.device
        or value.dtype != reference.dtype
    ):
        raise ValueError(f'{name} must share batch, device and dtype with the semantic states')
    if torch.is_autocast_enabled(value.device.type):
        raise ValueError('DynamicRelation/FormulaGRU require autocast disabled until AMP is validated')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')


def _check_module(module: nn.Module, reference: Tensor) -> None:
    if any(p.device != reference.device or p.dtype != reference.dtype for p in module.parameters()):
        raise ValueError('module parameters and inputs must share device and dtype; convert at the entry point')


class FormulaGRU(nn.Module):
    """Explicit reset-before-linear GRU; u is the candidate write proportion.

    Each W has its single specified bias; each U is bias-free. nn.Linear uses
    x @ weight.T, which is the batched row-vector form of the spec's W x.
    """

    def __init__(self, channels: int):
        super().__init__()
        _positive_integer(channels, 'channels')
        self.channels = channels
        self.W_u = nn.Linear(channels, channels, bias=True)
        self.U_u = nn.Linear(channels, channels, bias=False)
        self.W_rho = nn.Linear(channels, channels, bias=True)
        self.U_rho = nn.Linear(channels, channels, bias=False)
        self.W_z = nn.Linear(channels, channels, bias=True)
        self.U_z = nn.Linear(channels, channels, bias=False)

    def forward(self, message: Tensor, z: Tensor) -> Tensor:
        _check_tensor(z, 'z', self.channels)
        _check_tensor(message, 'message', self.channels, z)
        _check_module(self, z)
        u = torch.sigmoid(self.W_u(message) + self.U_u(z))
        rho = torch.sigmoid(self.W_rho(message) + self.U_rho(z))
        candidate = torch.tanh(self.W_z(message) + self.U_z(rho * z))
        return (1 - u) * z + u * candidate


class RelationRound(NamedTuple):
    """One round: alpha/messages from z^(t); z is z^(t+1).

    Diagnostics preserve autograd and are read-only by contract; no snapshots
    are detached, and no extra round history is retained when not requested.
    """

    alpha: Tensor  # [B,15,15], alpha[b,i,j] is i -> j; diagonal exactly zero.
    messages: Tensor  # [B,15,C], receiver axis; sum over senders, not mean.
    z: Tensor  # [B,15,C], all nodes updated synchronously.


class RelationResult(NamedTuple):
    zK: Tensor  # [B,15,C]
    rounds: tuple[RelationRound, ...]  # Index t describes transition t -> t+1.


class DynamicRelation(nn.Module):
    """K shared-parameter rounds on the complete directed graph without loops.

    C, Cr and K are explicit configuration, not formal experiment defaults.
    Fixed centroid/size/confidence inputs are reused unchanged, with gradients.
    Only standard PyTorch operators are used; this stage supports FP32/FP64
    outside autocast. Parameter initialization follows nn.Linear defaults.
    """

    def __init__(self, channels: int, *, relation_channels: int, rounds: int):
        super().__init__()
        for name, value in (('channels', channels), ('relation_channels', relation_channels), ('rounds', rounds)):
            _positive_integer(value, name)
        self.channels = channels
        self.relation_channels = relation_channels
        self.rounds = rounds
        self.relation_hidden = nn.Linear(2 * channels + 7, relation_channels, bias=True)
        self.relation_output = nn.Linear(relation_channels, 1, bias=True)
        self.W_m = nn.Linear(channels, channels, bias=False)
        self.gru = FormulaGRU(channels)

    def extra_repr(self) -> str:
        return f'channels={self.channels}, relation_channels={self.relation_channels}, rounds={self.rounds}'

    def _relation_descriptors(self, z: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor) -> Tensor:
        def sender(x: Tensor) -> Tensor:
            return x.unsqueeze(2).expand(-1, -1, NODE_COUNT, -1)

        def receiver(x: Tensor) -> Tensor:
            return x.unsqueeze(1).expand(-1, NODE_COUNT, -1, -1)

        return torch.cat((sender(z), receiver(z), receiver(centroid) - sender(centroid),
                          sender(size), receiver(size), sender(confidence), receiver(confidence)), dim=-1)

    def _edge_weights(self, z: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor) -> Tensor:
        descriptors = self._relation_descriptors(z, centroid, size, confidence)
        scores = self.relation_output(torch.relu(self.relation_hidden(descriptors))).squeeze(-1)
        alpha = torch.sigmoid(scores)
        self_edges = torch.eye(NODE_COUNT, device=z.device, dtype=torch.bool)
        return alpha.masked_fill(self_edges, 0)  # Architectural no-self-loop mask only.

    def _aggregate_messages(self, z: Tensor, alpha: Tensor) -> Tensor:
        # [B,receiver,sender] @ [B,sender,C]; W_m has no bias.
        return torch.bmm(alpha.transpose(1, 2), self.W_m(z))

    def forward(self, z0: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor,
                *, return_diagnostics: bool = False) -> Tensor | RelationResult:
        _check_tensor(z0, 'z0', self.channels)
        for name, value, channels in (('centroid', centroid, 3), ('size', size, 1), ('confidence', confidence, 1)):
            _check_tensor(value, name, channels, z0)
        _check_module(self, z0)
        if type(return_diagnostics) is not bool:
            raise ValueError('return_diagnostics must be a boolean')

        z = z0
        history = []
        for _ in range(self.rounds):
            alpha = self._edge_weights(z, centroid, size, confidence)
            message = self._aggregate_messages(z, alpha)
            z = self.gru(message, z)  # One new tensor for all nodes; no in-place node updates.
            if not torch.isfinite(z).all():
                raise ValueError('relation update produced non-finite semantic states')
            if return_diagnostics:
                history.append(RelationRound(alpha, message, z))
        return RelationResult(z, tuple(history)) if return_diagnostics else z
