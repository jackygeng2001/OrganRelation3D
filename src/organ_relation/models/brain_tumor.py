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
from .segmentor import SegmentorOutput
from .space_to_node import SpaceToNode


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

    def to_dict(self):
        result = asdict(self)
        if result['variant'] is None:  # The full method keeps its pre-ablation identity.
            del result['variant']
        return result


VARIANTS = (None, 'no_relation', 'drf_backbone', 'unet')


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
                                                rounds=config.rounds,node_count=3)
            self.node_to_space = NodeToSpace(256,attention_channels=config.attention_channels,
                                             content_channels=config.content_channels,node_count=3)
            self.fusion = ResidualFusion(256,content_channels=config.content_channels,bias=config.fusion_bias,
                                         learnable_relation_scale=True,relation_scale_init=config.gamma_init)
            self.decoder = BrainDecoder(bias=config.conv_bias,align_corners=config.align_corners)
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
            nodes = zK = last_alpha = scale_weights = None
            final = self.decoder(features[-1],features)
        else:
            nodes = self.space_to_node(F,coarse.probabilities)
            if self.variant == 'no_relation':  # No rounds: z(K) := z(0); no edges, so no incoming gates.
                zK, last_alpha = nodes.z0, nodes.z0.new_zeros(nodes.z0.shape[0],3,3)
            else:
                relation = self.relation.forward_with_last_edges(nodes.z0,nodes.centroid,nodes.size,nodes.confidence)
                zK, last_alpha = relation.zK, relation.last_alpha
            content = self.node_to_space(F,zK)
            fused = self.fusion(F,content)
            if reduced:
                fused = fused.to(self.volume_dtype)
            decoded = self.decoder(fused,features,zK,nodes,last_alpha,diagnostics=diagnostics)
            final, scale_weights = decoded if diagnostics else (decoded, None)
        if reduced:
            final = final.float()
        if not torch.isfinite(final).all():
            raise ValueError('nonfinite final logits')
        output = SegmentorOutput(coarse.logits,final)
        return BrainDiagnostics(output,features,nodes,zK,last_alpha,scale_weights) if diagnostics else output

    def forward(self, image):
        return self._run(image,False)

    def forward_with_diagnostics(self, image):
        return self._run(image,True)
