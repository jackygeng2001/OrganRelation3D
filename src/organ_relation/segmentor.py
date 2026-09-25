"""Full METHOD_SPEC forward graph; image only, without loss or training logic."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .backbone import Decoder3D, Encoder3D, EncoderFeatures
from .coarse_head import CoarseHead, CoarsePrediction
from .dynamic_relation import DynamicRelation, RelationResult
from .node_to_space import NodeToSpace, NodeToSpaceResult
from .residual_fusion import ResidualFusion
from .segmentor_config import SegmentorConfig
from .space_to_node import OrganNodes, SpaceToNode


class SegmentorOutput(NamedTuple):
    coarse_logits: Tensor  # [B,16,Df,Hf,Wf]; not upsampled in this forward.
    final_logits: Tensor  # [B,16,D,H,W]; no final softmax here.


class SegmentorDiagnostics(NamedTuple):
    """Explicit opt-in only; retaining this result retains the intermediate graph."""

    output: SegmentorOutput
    encoder: EncoderFeatures
    coarse: CoarsePrediction
    nodes: OrganNodes
    relation: RelationResult
    writeback: NodeToSpaceResult
    fused: Tensor


class Segmentor(nn.Module):
    def __init__(self, config: SegmentorConfig):
        super().__init__()
        if not isinstance(config, SegmentorConfig):
            raise ValueError('config must be a SegmentorConfig')
        self.config = config
        channels = config.backbone.channels[-1]
        self.encoder = Encoder3D(config.backbone)
        self.coarse_head = CoarseHead(channels, bias=config.coarse_bias)
        self.space_to_node = SpaceToNode(epsilon=config.epsilon)
        self.relation = DynamicRelation(channels, relation_channels=config.relation_channels, rounds=config.rounds)
        self.node_to_space = NodeToSpace(channels, attention_channels=config.attention_channels,
                                         content_channels=config.content_channels, beta_init=config.beta_init)
        self.fusion = ResidualFusion(channels, content_channels=config.content_channels, bias=config.fusion_bias)
        self.decoder = Decoder3D(config.backbone)

    def _run(self, image: Tensor, *, diagnostics: bool) -> SegmentorOutput | SegmentorDiagnostics:
        if not isinstance(image, Tensor) or image.ndim != 5 or min(image.shape) < 1 or image.shape[1] != 1:
            raise ValueError('image must have positive shape [B,1,D,H,W]')
        if image.dtype not in (torch.float32, torch.float64):
            raise ValueError('Segmentor requires float32 or float64; AMP is not validated')
        if torch.is_autocast_enabled(image.device.type):
            raise ValueError('Segmentor requires autocast disabled until AMP is validated')
        if any(p.dtype != image.dtype or p.device != image.device for p in self.parameters()):
            raise ValueError('module parameters and image must share device and dtype')
        if not torch.isfinite(image).all():
            raise ValueError('image must contain only finite values')

        encoded = self.encoder(image)
        F = encoded.deepest
        coarse = self.coarse_head(F)
        nodes = self.space_to_node(F, coarse.probabilities)
        relation = self.relation(nodes.z0, nodes.centroid, nodes.size, nodes.confidence,
                                 return_diagnostics=diagnostics)
        zK = relation.zK if diagnostics else relation
        writeback = self.node_to_space(F, zK, return_diagnostics=diagnostics)
        G = writeback.G if diagnostics else writeback
        Fprime = self.fusion(F, G)
        final_logits = self.decoder(Fprime, encoded.skips)
        if not torch.isfinite(final_logits).all():
            raise ValueError('decoder produced non-finite logits')
        output = SegmentorOutput(coarse.logits, final_logits)
        if diagnostics:
            return SegmentorDiagnostics(output, encoded, coarse, nodes, relation, writeback, Fprime)
        return output

    def forward(self, image: Tensor) -> SegmentorOutput:
        """Normal forward has exactly one input; no diagnostics cached on modules."""
        return self._run(image, diagnostics=False)

    def forward_with_diagnostics(self, image: Tensor) -> SegmentorDiagnostics:
        """Same graph, with explicit read-only intermediate results; no detach."""
        return self._run(image, diagnostics=True)
