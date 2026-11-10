"""Dynamic directed organ reasoning and the explicit METHOD_SPEC GRU.

Edge axes are always [batch, sender, receiver]. Node count is explicit (legacy: 15);
only their semantic states change between rounds. No labels enter this module.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn


NODE_COUNT = 15
GRAPHS = ('complete', 'nested_chain')
# 嵌套双向链（2026-11-10 老师批准）：节点顺序 [WT, TC, ET] 由外到内，WT 与 ET 之间没有直接的边。
# 向内边（外层→内层）表示包含关系，用外层区域约束内层的范围；向外边（内层→外层）用内层区域帮助外层定位。
# 两类边各有一套边门控 MLP 与消息矩阵；GRU 节点更新两类共用。边的轴为 [sender, receiver]。
NESTED_CHAIN_EDGES = {'inward': ((0, 1), (1, 2)), 'outward': ((1, 0), (2, 1))}


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _check_tensor(value: Tensor, name: str, channels: int, reference: Tensor | None = None, node_count: int = NODE_COUNT) -> None:
    if not isinstance(value, Tensor) or value.ndim != 3:
        raise ValueError(f'{name} must have shape [B,{node_count},{channels}]')
    if value.shape[0] < 1 or tuple(value.shape[1:]) != (node_count, channels):
        raise ValueError(f'{name} must have shape [B,{node_count},{channels}] with B>=1')
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

    def __init__(self, channels: int, *, node_count: int = NODE_COUNT):
        super().__init__()
        _positive_integer(channels, 'channels')
        self.channels = channels
        _positive_integer(node_count, "node_count")
        self.node_count = node_count
        self.W_u = nn.Linear(channels, channels, bias=True)
        self.U_u = nn.Linear(channels, channels, bias=False)
        self.W_rho = nn.Linear(channels, channels, bias=True)
        self.U_rho = nn.Linear(channels, channels, bias=False)
        self.W_z = nn.Linear(channels, channels, bias=True)
        self.U_z = nn.Linear(channels, channels, bias=False)

    def forward(self, message: Tensor, z: Tensor) -> Tensor:
        _check_tensor(z, 'z', self.channels, node_count=self.node_count)
        _check_tensor(message, 'message', self.channels, z, self.node_count)
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

    alpha: Tensor  # [B,nodes,nodes], alpha[b,i,j] is i -> j; diagonal exactly zero.
    messages: Tensor  # [B,nodes,C], receiver axis; sum over senders, not mean.
    z: Tensor  # [B,nodes,C], all nodes updated synchronously.


class RelationResult(NamedTuple):
    zK: Tensor  # [B,nodes,C]
    rounds: tuple[RelationRound, ...]  # Index t describes transition t -> t+1.


class RelationState(NamedTuple):
    zK: Tensor
    last_alpha: Tensor  # Gates computed from z^(K-1), not a new K+1 round.


class DynamicRelation(nn.Module):
    """K 轮共享参数的有向关系推理，无自环。

    graph='complete'（缺省，历史实验 E1/A3）：全部 N(N-1) 条有向边共用一套边门控与消息矩阵，
    参数名、构建顺序与计算保持不变。graph='nested_chain'：三节点 [WT,TC,ET] 的嵌套双向链，
    只有 NESTED_CHAIN_EDGES 中的四条边，向内/向外两类边各有一套边门控 MLP 与消息矩阵；
    不存在的边由固定的结构掩码置 0（与无自环掩码同类），不是由标签决定。
    C, Cr and K are explicit configuration, not formal experiment defaults.
    Fixed centroid/size/confidence inputs are reused unchanged, with gradients.
    Only standard PyTorch operators are used; this stage supports FP32/FP64
    outside autocast. Parameter initialization follows nn.Linear defaults.
    """

    def __init__(self, channels: int, *, relation_channels: int, rounds: int, node_count: int = NODE_COUNT,
                 graph: str = 'complete'):
        super().__init__()
        for name, value in (('channels', channels), ('relation_channels', relation_channels), ('rounds', rounds)):
            _positive_integer(value, name)
        self.channels = channels
        _positive_integer(node_count, "node_count")
        if graph not in GRAPHS:
            raise ValueError(f'graph must be one of {GRAPHS}')
        if graph == 'nested_chain' and node_count != 3:
            raise ValueError('nested_chain is defined only for the three nodes [WT, TC, ET]')
        self.node_count = node_count
        self.relation_channels = relation_channels
        self.rounds = rounds
        self.graph = graph
        if graph == 'complete':
            self.relation_hidden = nn.Linear(2 * channels + 7, relation_channels, bias=True)
            self.relation_output = nn.Linear(relation_channels, 1, bias=True)
            self.W_m = nn.Linear(channels, channels, bias=False)
        else:
            # 向内边（包含：外层约束内层）与向外边（定位：内层帮助外层）各一套参数，构建顺序固定。
            self.inward_hidden = nn.Linear(2 * channels + 7, relation_channels, bias=True)
            self.inward_output = nn.Linear(relation_channels, 1, bias=True)
            self.W_inward = nn.Linear(channels, channels, bias=False)
            self.outward_hidden = nn.Linear(2 * channels + 7, relation_channels, bias=True)
            self.outward_output = nn.Linear(relation_channels, 1, bias=True)
            self.W_outward = nn.Linear(channels, channels, bias=False)
            # 固定的结构掩码：非持久 buffer，随模型移动设备，不写入 state_dict/checkpoint，不消耗随机数。
            for kind, pairs in NESTED_CHAIN_EDGES.items():
                mask = torch.zeros(3, 3, dtype=torch.bool)
                for sender, receiver in pairs:
                    mask[sender, receiver] = True
                self.register_buffer(f'{kind}_mask', mask, persistent=False)
        self.gru = FormulaGRU(channels, node_count=node_count)

    def extra_repr(self) -> str:
        graph = '' if self.graph == 'complete' else f', graph={self.graph}'
        return f'channels={self.channels}, relation_channels={self.relation_channels}, rounds={self.rounds}{graph}'

    def edge_masks(self, device=None) -> dict[str, Tensor]:
        """各类边的固定结构掩码 [sender, receiver]（bool）。complete 图只有一类 'all'（去掉自环），
        在 device 上新建；nested_chain 直接返回随模型所在设备的 buffer。"""
        if self.graph == 'complete':
            return {'all': ~torch.eye(self.node_count, device=device, dtype=torch.bool)}
        return {kind: getattr(self, f'{kind}_mask') for kind in NESTED_CHAIN_EDGES}

    def _typed_parameters(self, kind: str):
        return {'inward': (self.inward_hidden, self.inward_output, self.W_inward),
                'outward': (self.outward_hidden, self.outward_output, self.W_outward)}[kind]

    def _relation_descriptors(self, z: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor) -> Tensor:
        def sender(x: Tensor) -> Tensor:
            return x.unsqueeze(2).expand(-1, -1, self.node_count, -1)

        def receiver(x: Tensor) -> Tensor:
            return x.unsqueeze(1).expand(-1, self.node_count, -1, -1)

        return torch.cat((sender(z), receiver(z), receiver(centroid) - sender(centroid),
                          sender(size), receiver(size), sender(confidence), receiver(confidence)), dim=-1)

    def _edge_weights(self, z: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor) -> Tensor:
        descriptors = self._relation_descriptors(z, centroid, size, confidence)
        if self.graph == 'complete':
            scores = self.relation_output(torch.relu(self.relation_hidden(descriptors))).squeeze(-1)
            alpha = torch.sigmoid(scores)
            self_edges = torch.eye(self.node_count, device=z.device, dtype=torch.bool)
            return alpha.masked_fill(self_edges, 0)  # Architectural no-self-loop mask only.
        # 每类边用自己的门控 MLP 打分，只保留该类的边；两类掩码互不重叠，相加即完整的 [B,3,3] 边权。
        alpha = None
        for kind, mask in self.edge_masks(z.device).items():
            hidden, output, _ = self._typed_parameters(kind)
            typed = torch.sigmoid(output(torch.relu(hidden(descriptors))).squeeze(-1)).masked_fill(~mask, 0)
            alpha = typed if alpha is None else alpha + typed
        return alpha

    def _aggregate_messages(self, z: Tensor, alpha: Tensor) -> Tensor:
        # [B,receiver,sender] @ [B,sender,C]; W_m has no bias.
        if self.graph == 'complete':
            return torch.bmm(alpha.transpose(1, 2), self.W_m(z))
        # 按边的类别拆开边权，各用自己的消息矩阵，再在接收端求和（不是平均）。
        message = None
        for kind, mask in self.edge_masks(z.device).items():
            typed = torch.bmm(alpha.masked_fill(~mask, 0).transpose(1, 2), self._typed_parameters(kind)[2](z))
            message = typed if message is None else message + typed
        return message

    def forward(self, z0: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor,
                *, return_diagnostics: bool = False) -> Tensor | RelationResult:
        return self._run(z0, centroid, size, confidence, return_diagnostics, False)

    def forward_with_last_edges(self, z0: Tensor, centroid: Tensor, size: Tensor,
                                confidence: Tensor) -> RelationState:
        """Decoder state without retaining a diagnostic history or recomputing gates."""
        return self._run(z0, centroid, size, confidence, False, True)

    def _run(self, z0, centroid, size, confidence, return_diagnostics, return_last_edges):
        _check_tensor(z0, 'z0', self.channels, node_count=self.node_count)
        for name, value, channels in (('centroid', centroid, 3), ('size', size, 1), ('confidence', confidence, 1)):
            _check_tensor(value, name, channels, z0, self.node_count)
        _check_module(self, z0)
        if type(return_diagnostics) is not bool:
            raise ValueError('return_diagnostics must be a boolean')

        if type(return_last_edges) is not bool or (return_last_edges and return_diagnostics):
            raise ValueError('choose last-edge output or full diagnostics, not both')
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
        if return_last_edges:
            return RelationState(z, alpha)
        return RelationResult(z, tuple(history)) if return_diagnostics else z
