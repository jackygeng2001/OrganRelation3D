"""Revised brain-tumor method, equations (1)-(26); image-only whole-volume forward."""
from dataclasses import asdict, dataclass
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as functional

from .coarse_head import CoarseHead
from .dynamic_relation import DynamicRelation
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

    def __post_init__(self):
        for name in ('relation_channels', 'rounds', 'attention_channels', 'content_channels'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('positive integer required: ' + name)
        if (isinstance(self.epsilon, bool) or isinstance(self.gamma_init, bool)
                or not math.isfinite(self.epsilon) or self.epsilon <= 0 or not math.isfinite(self.gamma_init)):
            raise ValueError('finite gamma and positive epsilon required')
        if any(type(getattr(self, k)) is not bool for k in ('conv_bias','coarse_bias','fusion_bias','align_corners')):
            raise ValueError('explicit boolean settings required')

    def to_dict(self):
        return asdict(self)


def unit(a, b, *, kernel=3, stride=1, dilation=1, bias):
    return nn.Sequential(nn.Conv3d(a, b, kernel, stride=stride,
        padding=dilation if kernel == 3 else 0, dilation=dilation, bias=bias),
        nn.GroupNorm(8, b), nn.LeakyReLU(0.01, inplace=False))


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

    def __init__(self, *, bias):
        super().__init__()
        self.stem = nn.Sequential(unit(4, 32, bias=bias), unit(32, 32, bias=bias))
        self.stages = nn.ModuleList(nn.Sequential(unit(a,b,stride=2,bias=bias), DualReceptiveField(b,bias=bias))
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


class BrainDecoder(nn.Module):
    scale_order = (2,1,0)

    def __init__(self, *, bias, align_corners):
        super().__init__()
        self.align_corners = align_corners
        self.scale_mlp = nn.Sequential(nn.Linear(266,64),nn.LeakyReLU(.01),nn.Linear(64,3))
        self.queries = nn.ModuleList(NodeToSpace(256,key_channels=c,node_count=3,
            attention_channels=32,content_channels=32) for c in (128,64,32))
        self.skip_projection = nn.ModuleList(nn.Conv3d(32,c,1,bias=False) for c in (128,64,32))
        self.eta = nn.Parameter(torch.full((3,),1e-3))
        self.up = nn.ModuleList(unit(a,b,kernel=1,bias=bias) for a,b in ((256,128),(128,64),(64,32)))
        self.fuse = nn.ModuleList(nn.Sequential(unit(2*c,c,bias=bias),unit(c,c,bias=bias)) for c in (128,64,32))
        self.head = nn.Conv3d(32,3,1,bias=bias)

    def forward(self, deepest, features, zK, nodes, last_alpha, *, diagnostics=False):
        weights = self.scale_mlp(scale_descriptor(zK,nodes,last_alpha)).softmax(-1)
        decoded = deepest
        for index, level in enumerate(self.scale_order):
            skip = features[level]
            content = self.queries[index].forward_with_scale_weights(skip,zK,region_weights=weights[:,:,index:index+1])
            enhanced = skip + self.eta[index]*self.skip_projection[index](content)
            decoded = self.up[index](functional.interpolate(decoded,size=skip.shape[2:],
                                      mode='trilinear',align_corners=self.align_corners))
            decoded = self.fuse[index](torch.cat((decoded,enhanced),1))
        logits = self.head(decoded)
        return (logits,weights) if diagnostics else logits


class BrainDiagnostics(NamedTuple):
    output: SegmentorOutput
    features: tuple
    nodes: object
    zK: torch.Tensor
    last_alpha: torch.Tensor
    scale_weights: torch.Tensor


class BrainTumorSegmentor(nn.Module):
    def __init__(self, config: BrainTumorConfig):
        super().__init__()
        self.config = config
        self.encoder = BrainEncoder(bias=config.conv_bias)
        self.coarse_head = CoarseHead(256,bias=config.coarse_bias,out_channels=3,activation='sigmoid')
        self.space_to_node = SpaceToNode(epsilon=config.epsilon,node_count=3,include_background=False)
        self.relation = DynamicRelation(256,relation_channels=config.relation_channels,rounds=config.rounds,node_count=3)
        self.node_to_space = NodeToSpace(256,attention_channels=config.attention_channels,
                                         content_channels=config.content_channels,node_count=3)
        self.fusion = ResidualFusion(256,content_channels=config.content_channels,bias=config.fusion_bias,
                                     learnable_relation_scale=True,relation_scale_init=config.gamma_init)
        self.decoder = BrainDecoder(bias=config.conv_bias,align_corners=config.align_corners)

    def _run(self, image, diagnostics):
        if image.ndim != 5 or image.shape[1] != 4 or min(image.shape) < 1 or image.dtype not in (torch.float32,torch.float64):
            raise ValueError('image must be FP32/FP64 [B,4,D,H,W]')
        if torch.is_autocast_enabled(image.device.type) or not torch.isfinite(image).all():
            raise ValueError('finite input with autocast disabled required')
        features = self.encoder(image)
        F = features[-1]
        coarse = self.coarse_head(F)
        nodes = self.space_to_node(F,coarse.probabilities)
        relation = self.relation.forward_with_last_edges(nodes.z0,nodes.centroid,nodes.size,nodes.confidence)
        content = self.node_to_space(F,relation.zK)
        fused = self.fusion(F,content)
        decoded = self.decoder(fused,features,relation.zK,nodes,relation.last_alpha,diagnostics=diagnostics)
        final = decoded[0] if diagnostics else decoded
        if not torch.isfinite(final).all():
            raise ValueError('nonfinite final logits')
        output = SegmentorOutput(coarse.logits,final)
        return BrainDiagnostics(output,features,nodes,relation.zK,relation.last_alpha,decoded[1]) if diagnostics else output

    def forward(self, image):
        return self._run(image,False)

    def forward_with_diagnostics(self, image):
        return self._run(image,True)
