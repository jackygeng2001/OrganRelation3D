"""Dynamic directed organ reasoning and the explicit METHOD_SPEC GRU.

Edge axes are always [batch, sender, receiver]. Node count is explicit (legacy: 15);
only their semantic states change between rounds. No labels enter this module.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

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
    # 包含关系读出（WT→TC、TC→ET 两条边）；未启用时为 None。[B,2]：包含比例 logit；
    # 启用存在性读出时为 [B,2,2]，最后一维依次是包含比例 logit、内层区域存在 logit。
    containment: Optional[Tensor] = None


CONDITIONINGS = ('none', 'layernorm_residual')
# 相邻嵌套的两条向内边 [sender, receiver]：WT→TC、TC→ET。包含比例读出只定义在这两条边上。
CONTAINMENT_EDGES = ((0, 1), (1, 2))
# 存在 logit 的初始偏置：sigmoid(4) ≈ 0.98，起点近似"都存在"，门控几乎不改变输出。
EXISTENCE_BIAS_INIT = 4.0


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

    conditioning='none'（缺省）：式 12–16 直接作用在 z(0) 上，历史行为、参数与初始化不变。
    conditioning='layernorm_residual'（2026-11-11）：h(0) = LayerNorm(z(0))，边门控、消息与 GRU 都在
    归一化后的状态 h 上计算（公式不变），读出 z' = z(0) + W_o h(K)，W_o 无偏置、零初始化。
    原因：z(0) 是未归一化的池化特征（训练中范数 20–50，分量可到 ±15），直接送入门控 MLP 与 GRU 会使
    sigmoid/tanh 饱和，边权停在 0 或 1、梯度消失（EXPERIMENTS.md 5.5）。零初始化使起点与"不做关系推理"相同。
    containment=True（需要 layernorm_residual）：对 CONTAINMENT_EDGES 的两条边，用末轮之后的状态
    h(K) 与同一组几何量，经独立的两层 MLP 输出包含比例 logit（|内层|/|外层| 的估计），供损失监督与先验平移使用。
    existence=True（需要 containment）：同一个 MLP 另输出"内层区域是否存在"的 logit，供存在性门控使用。
    新增模块都排在原有参数之后构建，不改变原有参数的初始化。
    """

    def __init__(self, channels: int, *, relation_channels: int, rounds: int, node_count: int = NODE_COUNT,
                 graph: str = 'complete', conditioning: str = 'none', containment: bool = False,
                 existence: bool = False):
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
        if conditioning not in CONDITIONINGS or type(containment) is not bool or type(existence) is not bool:
            raise ValueError(f'conditioning must be one of {CONDITIONINGS}; containment and existence are booleans')
        if existence and not containment:
            raise ValueError('the existence readout is part of the containment readout')
        self.existence = existence
        if containment and (conditioning != 'layernorm_residual' or node_count != 3):
            raise ValueError('the containment readout needs layernorm_residual conditioning and the three nodes [WT, TC, ET]')
        self.conditioning, self.containment = conditioning, containment
        if conditioning == 'layernorm_residual':
            self.state_norm = nn.LayerNorm(channels)
            self.readout = nn.Linear(channels, channels, bias=False)
            nn.init.zeros_(self.readout.weight)  # 起点 z' = z(0)
        if containment:
            self.containment_hidden = nn.Linear(2 * channels + 7, relation_channels, bias=True)
            self.containment_output = nn.Linear(relation_channels, 2 if existence else 1, bias=True)
            if existence:
                with torch.no_grad():
                    self.containment_output.bias[1] = EXISTENCE_BIAS_INIT

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
                                confidence: Tensor, extra: Optional[Tensor] = None) -> RelationState:
        """Decoder state without retaining a diagnostic history or recomputing gates.

        extra [B,N,C]（仅 layernorm_residual）：加到 h(0) 上的额外节点输入（其他尺度、模态强度等），
        即 h(0) = LayerNorm(z(0)) + extra；它只进入关系推理的状态，残差读出仍以 z(0) 为底。
        """
        return self._run(z0, centroid, size, confidence, False, True, extra)

    def initial_state(self, z0: Tensor, extra: Optional[Tensor] = None) -> Tensor:
        """关系推理的起始状态：历史行为为 z(0)；layernorm_residual 为 LayerNorm(z(0)) [+ extra]。"""
        if self.conditioning != 'layernorm_residual':
            if extra is not None:
                raise ValueError('extra node inputs need layernorm_residual conditioning')
            return z0
        state = self.state_norm(z0)
        if extra is not None:
            _check_tensor(extra, 'extra', self.channels, z0, self.node_count)
            state = state + extra
        return state

    def _run(self, z0, centroid, size, confidence, return_diagnostics, return_last_edges, extra=None):
        _check_tensor(z0, 'z0', self.channels, node_count=self.node_count)
        for name, value, channels in (('centroid', centroid, 3), ('size', size, 1), ('confidence', confidence, 1)):
            _check_tensor(value, name, channels, z0, self.node_count)
        _check_module(self, z0)
        if type(return_diagnostics) is not bool:
            raise ValueError('return_diagnostics must be a boolean')

        if type(return_last_edges) is not bool or (return_last_edges and return_diagnostics):
            raise ValueError('choose last-edge output or full diagnostics, not both')
        conditioned = self.conditioning == 'layernorm_residual'
        z = self.initial_state(z0, extra)
        history = []
        for _ in range(self.rounds):
            alpha = self._edge_weights(z, centroid, size, confidence)
            message = self._aggregate_messages(z, alpha)
            z = self.gru(message, z)  # One new tensor for all nodes; no in-place node updates.
            if not torch.isfinite(z).all():
                raise ValueError('relation update produced non-finite semantic states')
            if return_diagnostics:
                history.append(RelationRound(alpha, message, z))
        containment = self.containment_logits(z, centroid, size, confidence) if self.containment else None
        if conditioned:
            z = z0 + self.readout(z)  # 残差读出；history 中记录的是归一化空间里的状态 h(t)
        if return_last_edges:
            return RelationState(z, alpha, containment)
        return RelationResult(z, tuple(history)) if return_diagnostics else z

    def containment_logits(self, h: Tensor, centroid: Tensor, size: Tensor, confidence: Tensor) -> Tensor:
        """WT→TC、TC→ET 两条边的读出，由末轮之后的 h(K) 与几何量经独立 MLP 得到。

        [B,2]：包含比例 logit；existence=True 时 [B,2,2]，最后一维为（包含比例 logit，内层存在 logit）。
        """
        descriptors = self._relation_descriptors(h, centroid, size, confidence)
        scores = self.containment_output(torch.relu(self.containment_hidden(descriptors)))  # [B,N,N,1 或 2]
        edges = torch.stack([scores[:, sender, receiver] for sender, receiver in CONTAINMENT_EDGES], dim=1)
        return edges if self.existence else edges.squeeze(-1)
