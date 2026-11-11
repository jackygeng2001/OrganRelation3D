"""Revised brain-tumor method, equations (1)-(26); image-only whole-volume forward.

Ablation variants (user-approved 2026-11-09, explicit `model.variant`; omitted =
the full document method, whose identity and construction are unchanged):
  no_relation  -- nodes are built and used as in the full model, but no relation
                  rounds run: z(K) := z(0) and the scale descriptor's incoming
                  edge gates are fixed to 0 (there are no edges).
  drf_backbone -- dual-receptive-field encoder + plain U-Net decoder (resize,
                  concatenate the skip, two units); no nodes/writeback/queries.
  unet         -- as drf_backbone, but each encoder stage is two ordinary units.
Every variant keeps the coarse head on the deepest encoder feature, so all
ablations train with the identical objective (27)-(33); in the two plain
variants it is auxiliary deep supervision only and feeds nothing else.

关系图（显式 `model.graph`）：省略 = 三节点完全图（六条有向边，历史实验 E1/A3，identity 不变）；
'nested_chain' = 2026-11-10 老师批准的嵌套双向链 WT⇄TC⇄ET（四条有向边，向内/向外两类边分开设参，
见 dynamic_relation.NESTED_CHAIN_EDGES）。没有关系模块的变体不受 graph 影响。
"""
from dataclasses import asdict, dataclass
import math
from typing import NamedTuple, Optional

import torch
from torch import nn
from torch.nn import functional as functional

from .coarse_head import CoarseHead
from .dynamic_relation import DynamicRelation
from .depth_sliced_conv import DepthSlicedConv3d
from .group_norm import ChannelsLastGroupNorm
from .node_to_space import NodeToSpace
from .residual_fusion import ResidualFusion
from .space_to_node import SpaceToNode


class SegmentorOutput(NamedTuple):
    coarse_logits: torch.Tensor  # [B,3,Df,Hf,Wf] 最深层粗预测，未上采样
    final_logits: torch.Tensor  # [B,3,D,H,W] 最终预测，未经 sigmoid


class ContainmentSegmentorOutput(NamedTuple):
    """启用 model.containment 时的输出；未启用时仍返回两个字段的 SegmentorOutput（历史解包方式不变）。"""
    coarse_logits: torch.Tensor
    final_logits: torch.Tensor  # 启用先验平移时已含平移
    # WT→TC、TC→ET 两条边的读出：[B,2] 包含比例 logit；containment='gate' 时 [B,2,2]（比例 logit，存在 logit）
    containment_logits: torch.Tensor


def existence_gate(logits, existence):
    """存在性门控：P'_j(x) = pi_j * sigmoid(L_j(x))，返回 logit(P')；WT 通道不变。

    existence [B,2] 是 WT→TC、TC→ET 两条包含边的存在 logit。沿嵌套链相乘：pi_TC = sigmoid(e_TC)，
    pi_ET = pi_TC * sigmoid(e_ET)（TC 不存在则 ET 也不存在）。全程用 logsigmoid / logaddexp 计算，
    任何有限输入都给出有限 logit；pi → 1 时 logit(P') → L，pi → 0 时整块区域被关掉。
    """
    if existence.shape != (logits.shape[0], 2) or logits.shape[1] != 3:
        raise ValueError('existence gate expects [B,3,...] logits and [B,2] existence logits')
    log_sigmoid = functional.logsigmoid
    e_tc, e_et = existence[:, 0], existence[:, 1]
    shape = (logits.shape[0], 2, 1, 1, 1)
    log_pi = torch.stack((log_sigmoid(e_tc), log_sigmoid(e_tc) + log_sigmoid(e_et)), 1).view(shape)
    log_not_pi = torch.stack((log_sigmoid(-e_tc), torch.logaddexp(log_sigmoid(-e_tc), log_sigmoid(e_tc) + log_sigmoid(-e_et))),
                             1).view(shape)
    inner = logits[:, 1:]
    log_p = log_pi + log_sigmoid(inner)                                                  # log P'
    log_not_p = torch.logaddexp(log_not_pi.expand_as(inner), log_pi + log_sigmoid(-inner))  # log(1 - P')
    return torch.cat((logits[:, :1], log_p - log_not_p), 1)


def nested_logits(logits):
    """逐体素嵌套分解：P_WT = sigmoid(L_WT)，P_TC = P_WT * sigmoid(L_TC)，P_ET = P_TC * sigmoid(L_ET)；返回 logit(P)。

    先定外层、内层只在外层之内细分（级联的单网络形式），输出必然满足 P_ET <= P_TC <= P_WT。
    全程用 logsigmoid / logaddexp，任何有限输入都得到有限 logit。
    """
    if logits.ndim != 5 or logits.shape[1] != 3:
        raise ValueError('nested output expects [B,3,D,H,W] logits')
    log_sigmoid = functional.logsigmoid
    wt, tc, et = logits[:, 0], logits[:, 1], logits[:, 2]
    log_tc = log_sigmoid(wt) + log_sigmoid(tc)
    log_not_tc = torch.logaddexp(log_sigmoid(-wt), log_sigmoid(wt) + log_sigmoid(-tc))
    log_et = log_tc + log_sigmoid(et)
    log_not_et = torch.logaddexp(log_not_tc, log_tc + log_sigmoid(-et))
    return torch.stack((wt, log_tc - log_not_tc, log_et - log_not_et), 1)


def containment_logits(output):
    """输出里的包含比例 logit；模型未启用时为 None。"""
    return getattr(output, 'containment_logits', None)


@dataclass(frozen=True)
class BrainTumorConfig:
    # Architecture is fixed by the revised document; unspecified choices are explicit.
    relation_channels: int
    rounds: int
    attention_channels: int
    content_channels: int
    epsilon: float
    gamma_init: float
    conv_bias: bool
    coarse_bias: bool
    fusion_bias: bool
    align_corners: bool
    variant: Optional[str] = None  # None = full method; see VARIANTS.
    graph: Optional[str] = None  # None = 历史完全图；'nested_chain' = 嵌套双向链；见 GRAPHS。
    # 关系推理 v2（2026-11-11，见 dynamic_relation.DynamicRelation 与 METHOD_SPEC）；省略即历史行为，identity 不变。
    relation_conditioning: Optional[str] = None  # 'layernorm_residual'：归一化状态上推理 + 零初始化残差读出
    # 'supervised'：包含比例读出（只由损失监督）；'prior_shift'：另按比例平移内层 logit；
    # 'gate'：读出另含内层存在 logit，最终概率 = 存在概率 × 体素概率（见 existence_gate）。
    containment: Optional[str] = None
    # 额外的节点输入（需要 relation_conditioning；按字母序的列表，取自 NODE_INPUTS）：
    # 'multiscale'：同一组粗掩码在 1/4、1/2 分辨率编码特征上的池化；'intensity'：各区域内四个模态的强度均值与标准差。
    node_inputs: Optional[list] = None
    # 'nested'：逐体素的嵌套分解 P_TC = P_WT * sigmoid(L_TC)，P_ET = P_TC * sigmoid(L_ET)（先外后内）。
    output_hierarchy: Optional[str] = None

    def __post_init__(self):
        for name in ('relation_channels', 'rounds', 'attention_channels', 'content_channels'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive integer required: ' + name)
        if (isinstance(self.epsilon, bool) or isinstance(self.gamma_init, bool)
                or not math.isfinite(self.epsilon) or self.epsilon <= 0 or not math.isfinite(self.gamma_init)):
            raise ValueError('finite gamma and positive epsilon required')
        if any(type(getattr(self, k)) is not bool for k in ('conv_bias','coarse_bias','fusion_bias','align_corners')):
            raise ValueError('explicit boolean settings required')
        if self.variant not in VARIANTS:
            raise ValueError(f'model variant must be omitted (full method) or one of {VARIANTS[1:]}')
        if self.graph not in GRAPHS:
            raise ValueError(f'model graph must be omitted (complete graph) or one of {GRAPHS[1:]}')
        if self.relation_conditioning not in (None, 'layernorm_residual'):
            raise ValueError('relation_conditioning must be omitted or layernorm_residual')
        if self.containment not in CONTAINMENTS:
            raise ValueError(f'containment must be omitted or one of {CONTAINMENTS[1:]}')
        if (self.relation_conditioning or self.containment) and self.variant is not None:
            raise ValueError('relation_conditioning/containment apply only to the full method (no variant)')
        if self.containment and self.relation_conditioning != 'layernorm_residual':
            raise ValueError('containment requires relation_conditioning=layernorm_residual')
        if self.node_inputs is not None and (
                not isinstance(self.node_inputs, list) or not self.node_inputs
                or self.node_inputs != sorted(set(self.node_inputs)) or not set(self.node_inputs) <= set(NODE_INPUTS)
                or self.relation_conditioning != 'layernorm_residual'):
            raise ValueError(f'node_inputs must be a sorted non-empty list from {NODE_INPUTS} and needs layernorm_residual')
        if self.output_hierarchy not in (None, 'nested'):
            raise ValueError('output_hierarchy must be omitted or nested')

    def to_dict(self):
        result = asdict(self)
        if result['variant'] is None:  # The full method keeps its pre-ablation identity.
            del result['variant']
        if result['graph'] is None:  # 完全图保持引入 graph 之前的 identity。
            del result['graph']
        for key in ('relation_conditioning', 'containment', 'node_inputs', 'output_hierarchy'):  # 省略时 identity 不变。
            if result[key] is None:
                del result[key]
        return result


VARIANTS = (None, 'no_relation', 'drf_backbone', 'unet')
GRAPHS = (None, 'nested_chain')
CONTAINMENTS = (None, 'supervised', 'prior_shift', 'gate')
NODE_INPUTS = ('intensity', 'multiscale')
# 先验平移 delta_j = PRIOR_SHIFT_GAIN * a_j * clamp(u_j, ±PRIOR_SHIFT_CLAMP)。a_j 是零初始化的可学习标量；
# 固定增益只为让标量在 AdamW 3e-4 下学得动（每步至多改变 3e-4，放大后约 300 步可到 1）。
PRIOR_SHIFT_GAIN = 10.0
PRIOR_SHIFT_CLAMP = 6.0


def unit(a, b, *, kernel=3, stride=1, dilation=1, bias):
    return nn.Sequential(nn.Conv3d(a, b, kernel, stride=stride,
        padding=dilation if kernel == 3 else 0, dilation=dilation, bias=bias),
        # GN backward needs its input/statistics, not its output. Positive-slope
        # LeakyReLU can safely reuse that private output for its own backward.
        ChannelsLastGroupNorm(8, b), nn.LeakyReLU(0.01, inplace=True))


class DualReceptiveField(nn.Module):
    def __init__(self, channels, *, bias):
        super().__init__()
        self.local = unit(channels, channels, bias=bias)
        self.context = unit(channels, channels, dilation=2, bias=bias)
        self.gate = nn.Sequential(nn.Conv3d(3*channels, channels//4, 1, bias=bias),
                                  nn.LeakyReLU(0.01), nn.Conv3d(channels//4, 2, 1, bias=bias))
        self.projection = nn.Conv3d(channels, channels, 1, bias=bias)

    def forward(self, x):
        local, context = self.local(x), self.context(x)
        weights = self.gate(torch.cat((local, context, (local-context).abs()), 1)).softmax(1)
        return x + self.projection(weights[:, :1]*local + weights[:, 1:]*context)


class BrainEncoder(nn.Module):
    channels = (32, 64, 128, 256)

    def __init__(self, *, bias, dual_receptive_field=True):
        super().__init__()
        self.stem = nn.Sequential(unit(4, 32, bias=bias), unit(32, 32, bias=bias))
        block = DualReceptiveField if dual_receptive_field else (lambda b, bias: unit(b, b, bias=bias))
        self.stages = nn.ModuleList(nn.Sequential(unit(a,b,stride=2,bias=bias), block(b,bias=bias))
                                   for a,b in zip(self.channels[:-1], self.channels[1:]))

    def forward(self, image):
        features = [self.stem(image)]
        for stage in self.stages:
            features.append(stage(features[-1]))
        return tuple(features)


def scale_descriptor(zK, nodes, last_alpha):
    """[WT,TC,ET] receivers; sender slots ascending, last PRE-update gates."""
    if zK.ndim != 3 or zK.shape[1:] != (3,256) or last_alpha.shape != (zK.shape[0],3,3):
        raise ValueError('expected three 256-dimensional nodes and [B,3,3] gates')
    incoming = torch.stack([last_alpha[:, [i for i in range(3) if i != j], j] for j in range(3)], 1)
    identity = torch.eye(3,device=zK.device,dtype=zK.dtype).unsqueeze(0).expand(zK.shape[0],-1,-1)
    return torch.cat((zK,nodes.centroid,nodes.size,nodes.confidence,incoming,identity), -1)


class ProjectThenUpsample(nn.Sequential):
    """Commute only the pointwise affine projection with trilinear resize.

    GroupNorm and activation must remain AFTER resize. Numeric child names
    preserve the original up-block state_dict keys and parameter order.
    """

    def __init__(self, a, b, *, bias):
        super().__init__(*unit(a, b, kernel=1, bias=bias))

    def forward(self, x, *, size, align_corners):
        x = self[0](x)
        x = functional.interpolate(x, size=size, mode='trilinear', align_corners=align_corners)
        return self[2](self[1](x))


class ProjectedRegionQuery(NodeToSpace):
    """Associate decoder linear maps on three nodes, not on full volumes.

    Q (WK F) = (Q WK) F; eta Phi(sum_i A_i V_i) =
    sum_i A_i (eta Phi(V_i)). Phi is the existing bias-free 1x1x1 map.
    The inherited unprojected query remains available for formula diagnostics.
    """

    def forward_projected(self, features, zK, region_weights, projection, scale):
        """bf16 skip volumes (whole-network mixed precision): node-level maps, score
        bias, sigmoid gates and scale weights stay FP32; only the two volume
        products take bf16 operands with FP32 accumulation."""
        self._validate(features, zK, reduced_features=True)
        reduced = features.dtype == torch.bfloat16
        if (region_weights.shape != (features.shape[0], self.node_count, 1)
                or region_weights.dtype != zK.dtype or region_weights.device != features.device
                or not torch.isfinite(region_weights).all()):
            raise ValueError('finite scale weights must match [B,nodes,1], dtype and device')
        if (projection.bias is not None or projection.kernel_size != (1,1,1)
                or projection.stride != (1,1,1) or projection.padding != (0,0,0)
                or projection.groups != 1):
            raise ValueError('decoder projection must be a bias-free pointwise linear map')
        Q = self.W_Q(zK)
        query = functional.linear(Q, self.W_K.weight.transpose(0,1))
        V = self.W_V(zK)
        # Treat the three nodes as three positions for the existing Conv3d.
        values = projection(V.transpose(1,2).unsqueeze(-1).unsqueeze(-1))
        values = values.squeeze(-1).squeeze(-1).transpose(1,2) * scale
        if reduced:
            scores = torch.bmm(query.to(features.dtype), features.flatten(2)).float()
        else:
            scores = torch.bmm(query, features.flatten(2))
        scores = scores / math.sqrt(self.attention_channels)
        scores = scores + self.beta.view(1,self.node_count,1)
        for name, value in (('Q',Q),('projected query',query),('V',V),('projected values',values),('scores',scores)):
            if not torch.isfinite(value).all():
                raise ValueError(f'nonfinite decoder {name}')
        attention = scores.sigmoid() * region_weights
        if reduced:
            attention, values = attention.to(features.dtype), values.to(features.dtype)
        content = torch.bmm(attention.transpose(1,2),values)
        content = content.transpose(1,2).reshape(features.shape)
        if not torch.isfinite(content).all():
            raise ValueError('nonfinite decoder writeback')
        return content


class BrainDecoder(nn.Module):
    scale_order = (2,1,0)

    def __init__(self, *, bias, align_corners):
        super().__init__()
        self.align_corners = align_corners
        self.scale_mlp = nn.Sequential(nn.Linear(266,64),nn.LeakyReLU(.01),nn.Linear(64,3))
        self.queries = nn.ModuleList(ProjectedRegionQuery(256,key_channels=c,node_count=3,
            attention_channels=32,content_channels=32) for c in (128,64,32))
        self.skip_projection = nn.ModuleList(nn.Conv3d(32,c,1,bias=False) for c in (128,64,32))
        self.eta = nn.Parameter(torch.full((3,),1e-3))
        self.up = nn.ModuleList(ProjectThenUpsample(a,b,bias=bias) for a,b in ((256,128),(128,64),(64,32)))
        self.fuse = nn.ModuleList(nn.Sequential(unit(2*c,c,bias=bias),unit(c,c,bias=bias)) for c in (128,64,32))
        self.head = nn.Conv3d(32,3,1,bias=bias)

    def forward(self, deepest, features, zK, nodes, last_alpha, *, diagnostics=False):
        weights = self.scale_mlp(scale_descriptor(zK,nodes,last_alpha)).softmax(-1)
        decoded = deepest
        for index, level in enumerate(self.scale_order):
            skip = features[level]
            content = self.queries[index].forward_projected(skip,zK,weights[:,:,index:index+1],
                                                           self.skip_projection[index],self.eta[index])
            enhanced = skip + content
            del content  # Autograd retains anything required for backward.
            decoded = self.up[index](decoded,size=skip.shape[2:],align_corners=self.align_corners)
            first_unit, second_unit = self.fuse[index]
            if isinstance(first_unit[0], DepthSlicedConv3d):
                decoded = first_unit[0](decoded, enhanced)
                del enhanced
                decoded = first_unit[2](first_unit[1](decoded))
                decoded = second_unit(decoded)
            else:
                decoded = torch.cat((decoded,enhanced),1)
                del enhanced  # Cat backward needs sizes, not this extra Python reference.
                decoded = self.fuse[index](decoded)
        logits = self.head(decoded)
        return (logits,weights) if diagnostics else logits


class PlainDecoder(nn.Module):
    """Ablation decoder: the full decoder's resize/fuse/head without region
    queries or scale weights (enhanced skip := skip)."""
    scale_order = BrainDecoder.scale_order

    def __init__(self, *, bias, align_corners):
        super().__init__()
        self.align_corners = align_corners
        self.up = nn.ModuleList(ProjectThenUpsample(a,b,bias=bias) for a,b in ((256,128),(128,64),(64,32)))
        self.fuse = nn.ModuleList(nn.Sequential(unit(2*c,c,bias=bias),unit(c,c,bias=bias)) for c in (128,64,32))
        self.head = nn.Conv3d(32,3,1,bias=bias)

    def forward(self, deepest, features):
        decoded = deepest
        for index, level in enumerate(self.scale_order):
            skip = features[level]
            decoded = self.up[index](decoded,size=skip.shape[2:],align_corners=self.align_corners)
            first_unit, second_unit = self.fuse[index]
            if isinstance(first_unit[0], DepthSlicedConv3d):
                decoded = first_unit[2](first_unit[1](first_unit[0](decoded, skip)))
                decoded = second_unit(decoded)
            else:
                decoded = self.fuse[index](torch.cat((decoded,skip),1))
        return self.head(decoded)


class BrainDiagnostics(NamedTuple):
    output: SegmentorOutput
    features: tuple
    nodes: object
    zK: torch.Tensor
    last_alpha: torch.Tensor
    scale_weights: torch.Tensor


MIXED_PRECISION = {'bfloat16': torch.bfloat16}


class BrainTumorSegmentor(nn.Module):
    """mixed_precision='bfloat16' (whole-network mixed precision, depth_sliced only):
    encoder/decoder volumes, their convolutions and pointwise projections run in
    bf16 with FP32 accumulation; parameters, GroupNorm statistics and arithmetic,
    coarse head, Space-to-Node, relation rounds, deep writeback/fusion, scale
    weights, decoder score bias/gates and the logits given to the loss are FP32.
    convolution_compute_dtype is the narrower kernel-only option; not both.
    """

    def __init__(self, config: BrainTumorConfig, *, convolution_backend='native', convolution_compute_dtype=None,
                 mixed_precision=None):
        super().__init__()
        if convolution_backend not in ('native', 'depth_sliced'):
            raise ValueError('unsupported convolution execution backend')
        if convolution_compute_dtype is not None and convolution_backend != 'depth_sliced':
            raise ValueError('reduced-precision convolution arithmetic requires the depth_sliced backend')
        if mixed_precision is not None and (mixed_precision not in MIXED_PRECISION
                                            or convolution_backend != 'depth_sliced'
                                            or convolution_compute_dtype is not None):
            raise ValueError('mixed_precision must be bfloat16, with depth_sliced and no convolution_compute_dtype')
        self.volume_dtype = None if mixed_precision is None else MIXED_PRECISION[mixed_precision]
        self.config = config
        self.variant = config.variant
        self.encoder = BrainEncoder(bias=config.conv_bias, dual_receptive_field=config.variant != 'unet')
        self.coarse_head = CoarseHead(256,bias=config.coarse_bias,out_channels=3,activation='sigmoid')
        if config.variant in ('drf_backbone', 'unet'):
            self.decoder = PlainDecoder(bias=config.conv_bias,align_corners=config.align_corners)
        else:  # Full method and no_relation; module order (and initialization) of the full method unchanged.
            self.space_to_node = SpaceToNode(epsilon=config.epsilon,node_count=3,include_background=False)
            if config.variant is None:
                self.relation = DynamicRelation(256,relation_channels=config.relation_channels,
                                                rounds=config.rounds,node_count=3,graph=config.graph or 'complete',
                                                conditioning=config.relation_conditioning or 'none',
                                                containment=config.containment is not None,
                                                existence=config.containment == 'gate')
            self.node_to_space = NodeToSpace(256,attention_channels=config.attention_channels,
                                             content_channels=config.content_channels,node_count=3)
            self.fusion = ResidualFusion(256,content_channels=config.content_channels,bias=config.fusion_bias,
                                         learnable_relation_scale=True,relation_scale_init=config.gamma_init)
            self.decoder = BrainDecoder(bias=config.conv_bias,align_corners=config.align_corners)
            if config.containment == 'prior_shift':  # 常数初始化，不消耗随机数；[a_TC, a_ET]
                self.prior_shift_scale = nn.Parameter(torch.zeros(2))
            # 额外的节点输入：排在其余模块之后构建，不改变它们的初始化。
            inputs = config.node_inputs or []
            if 'multiscale' in inputs:  # 1/4（128 通道）与 1/2（64 通道）编码特征的区域池化 -> C 维
                self.scale_norms = nn.ModuleList(nn.LayerNorm(c) for c in (128, 64))
                self.scale_projections = nn.ModuleList(nn.Linear(c, 256, bias=False) for c in (128, 64))
            if 'intensity' in inputs:  # 每个区域 8 个量：四个模态的强度均值与标准差
                self.intensity_projection = nn.Sequential(nn.Linear(8, 64), nn.LeakyReLU(0.01), nn.Linear(64, 256))
        if convolution_backend == 'depth_sliced':
            from .depth_sliced_conv import enable_depth_sliced_convolutions
            enable_depth_sliced_convolutions(self, convolution_compute_dtype)

    def _run(self, image, diagnostics):
        if image.ndim != 5 or image.shape[1] != 4 or min(image.shape) < 1 or image.dtype not in (torch.float32,torch.float64):
            raise ValueError('image must be FP32/FP64 [B,4,D,H,W]')
        if torch.is_autocast_enabled(image.device.type) or not torch.isfinite(image).all():
            raise ValueError('finite input with autocast disabled required')
        reduced = self.volume_dtype is not None
        if reduced and image.dtype != torch.float32:
            raise ValueError('mixed precision takes an FP32 image')
        features = self.encoder(image.to(self.volume_dtype) if reduced else image)
        # The deepest grid is 1/512 of the volume: coarse head, nodes, relation
        # rounds and the deep writeback/fusion stay FP32 under mixed precision.
        F = features[-1].float() if reduced else features[-1]
        coarse = self.coarse_head(F)
        if isinstance(self.decoder, PlainDecoder):
            del F  # Plain ablations: the coarse head is auxiliary supervision only.
            nodes = zK = last_alpha = scale_weights = containment = None
            final = self.decoder(features[-1],features)
        else:
            nodes = self.space_to_node(F,coarse.probabilities)
            containment = None
            if self.variant == 'no_relation':  # No rounds: z(K) := z(0); no edges, so no incoming gates.
                zK, last_alpha = nodes.z0, nodes.z0.new_zeros(nodes.z0.shape[0],3,3)
            else:
                extra = self.node_extra(features, coarse.probabilities, image) if self.config.node_inputs else None
                relation = self.relation.forward_with_last_edges(nodes.z0,nodes.centroid,nodes.size,nodes.confidence,extra)
                zK, last_alpha, containment = relation.zK, relation.last_alpha, relation.containment
            content = self.node_to_space(F,zK)
            fused = self.fusion(F,content)
            if reduced:
                fused = fused.to(self.volume_dtype)
            decoded = self.decoder(fused,features,zK,nodes,last_alpha,diagnostics=diagnostics)
            final, scale_weights = decoded if diagnostics else (decoded, None)
        if reduced:
            final = final.float()
        final = self.output_logits(final, containment)
        if not torch.isfinite(final).all():
            raise ValueError('nonfinite final logits')
        output = (SegmentorOutput(coarse.logits,final) if containment is None
                  else ContainmentSegmentorOutput(coarse.logits,final,containment))
        return BrainDiagnostics(output,features,nodes,zK,last_alpha,scale_weights) if diagnostics else output

    def output_logits(self, logits, containment, *, readout=True):
        """解码器 logit -> 模型输出：可选的逐体素嵌套分解，然后是包含读出的先验平移或存在性门控。

        readout=False 只跳过平移 / 门控（只读探针做敲除用），嵌套分解照常。
        """
        if self.config.output_hierarchy == 'nested':
            logits = nested_logits(logits)
        if readout and self.config.containment == 'prior_shift':
            logits = logits + self.prior_shift(containment).view(logits.shape[0],3,1,1,1)
        elif readout and self.config.containment == 'gate':
            logits = existence_gate(logits, containment[..., 1])
        return logits

    def node_extra(self, features, probabilities, image):
        """[B,3,C]：加到关系状态 h(0) 上的额外节点输入，都用同一组粗概率 P 作权重（P 插值到相应网格）。

        multiscale：z_i^(l) = sum_x P_i(x) E_l(x) / (sum_x P_i(x) + eps)，l = 2、1（1/4、1/2 分辨率），
        各经 LayerNorm 与无偏置线性映射到 C 维。最深层网格上 TC 与 ET 的掩码几乎重合，较细的尺度能把它们分开。
        intensity：各区域内 T1/T1ce/T2/FLAIR 四个模态的加权均值与标准差（8 维），经两层 MLP 到 C 维；
        这是深层特征里已被混合的"各模态原始对比度"。池化都在 FP32 中累加。
        """
        epsilon, extra = self.config.epsilon, 0
        def pooled(volume, power=1):
            weights = functional.interpolate(probabilities, size=volume.shape[2:], mode='trilinear',
                                             align_corners=self.config.align_corners).flatten(2)
            values = volume.flatten(2).transpose(1, 2).to(probabilities.dtype)
            mass = weights.sum(-1, keepdim=True) + epsilon
            if power == 1:
                return torch.bmm(weights, values) / mass
            return torch.bmm(weights, values) / mass, torch.bmm(weights, values.square()) / mass
        inputs = self.config.node_inputs
        if 'multiscale' in inputs:
            for level, norm, projection in zip((2, 1), self.scale_norms, self.scale_projections):
                extra = extra + projection(norm(pooled(features[level])))
        if 'intensity' in inputs:
            mean, second = pooled(image, power=2)
            deviation = (second - mean.square()).clamp_min(0).add(epsilon).sqrt()
            extra = extra + self.intensity_projection(torch.cat((mean, deviation), -1))
        if not torch.isfinite(extra).all():
            raise ValueError('nonfinite extra node inputs')
        return extra

    def prior_shift(self, containment):
        """[B,3] 逐区域的 logit 平移：WT 为 0；TC、ET 为 gain * a_j * clamp(u_j)。

        u_j 是关系模块给出的包含比例 logit（|TC|/|WT|、|ET|/|TC| 的估计）。这里对 u 取 detach：
        u 只由包含比例监督学习，保持"比例估计"的含义；分割损失经这条通路只学习标量 a_j（信多少）。
        """
        shift = PRIOR_SHIFT_GAIN * self.prior_shift_scale * containment.detach().clamp(-PRIOR_SHIFT_CLAMP, PRIOR_SHIFT_CLAMP)
        return torch.cat((shift.new_zeros(shift.shape[0],1), shift), 1)

    def forward(self, image):
        return self._run(image,False)

    def forward_with_diagnostics(self, image):
        return self._run(image,True)
