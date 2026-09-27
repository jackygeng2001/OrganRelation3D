"""Organ relation at the inspected deepest block of the official MONAI UNet."""
from torch import nn

from .monai_reference import MonaiReferenceUNet
from .coarse_head import CoarseHead
from .space_to_node import SpaceToNode
from .dynamic_relation import DynamicRelation
from .node_to_space import NodeToSpace
from .residual_fusion import ResidualFusion
from .segmentor import SegmentorOutput


BOTTLENECK_PATH = 'model.1.submodule.1.submodule.1.submodule.1.submodule'


class BottleneckAdapter(nn.Module):
    """Keep the original MONAI block and transform only its output."""
    def __init__(self, block, channels, relation_config, enabled):
        super().__init__()
        self.block = block
        self.channels = channels
        self.enabled = enabled
        # A call-scoped output collector, always cleared by the outer finally.
        # No hooks or activation cache survives forward (including failures).
        self._coarse_sink = None
        self._relation_diagnostics_sink = None
        if enabled:
            c = relation_config
            self.coarse_head = CoarseHead(channels, bias=c['coarse_bias'])
            self.space_to_node = SpaceToNode(epsilon=c['epsilon'])
            self.relation = DynamicRelation(channels, relation_channels=c['relation_channels'], rounds=c['rounds'])
            self.node_to_space = NodeToSpace(channels, attention_channels=c['attention_channels'],
                                             content_channels=c['content_channels'], beta_init=c['beta_init'])
            self.fusion = ResidualFusion(channels, content_channels=c['content_channels'], bias=c['fusion_bias'],
                learnable_relation_scale=c.get('learnable_relation_scale', False),
                relation_scale_init=c.get('relation_scale_init', 1.0))
            self.diagnostics_epsilon = c['epsilon']

    def forward(self, image):
        features = self.block(image)
        if not self.enabled:
            return features
        if self._coarse_sink is None:
            raise RuntimeError('enabled adapter must run inside MonaiRelationUNet.forward')
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError('unexpected MONAI deepest feature shape')
        coarse = self.coarse_head(features)
        nodes = self.space_to_node(features, coarse.probabilities)
        zK = self.relation(nodes.z0, nodes.centroid, nodes.size, nodes.confidence)
        content = self.node_to_space(features, zK)
        if self._relation_diagnostics_sink is None:
            fused = self.fusion(features, content)
        else:
            fused, stats = self.fusion.forward_with_diagnostics(features, content, epsilon=self.diagnostics_epsilon)
            self._relation_diagnostics_sink.update(stats)
        self._coarse_sink.append(coarse.logits)
        return fused


class MonaiRelationUNet(MonaiReferenceUNet):
    def __init__(self, constructor, input_processing, *, relation_enabled=False, relation_config=None):
        super().__init__(constructor, input_processing)
        if type(relation_enabled) is not bool:
            raise ValueError('relation_enabled must be boolean')
        if relation_enabled and (relation_config is None or relation_config['feature_channels'] != constructor['channels'][-1]):
            raise ValueError('explicit relation feature_channels must match MONAI bottleneck')
        self.relation_enabled = relation_enabled
        self.relation_config = dict(relation_config or {})
        from monai.networks.blocks import ResidualUnit
        from monai.networks.layers.simplelayers import SkipConnection
        if len(constructor['channels']) != 5 or list(constructor['strides']) != [2]*4:
            raise ValueError('adapter requires the inspected five-level MONAI hierarchy')
        parent_path, _, name = BOTTLENECK_PATH.rpartition('.')
        parent = self.network.get_submodule(parent_path)
        block = getattr(parent, name)
        if not isinstance(parent, SkipConnection) or parent.mode != 'cat' or not isinstance(block, ResidualUnit):
            raise ValueError('unsupported MONAI bottleneck structure; inspect before adapting')
        if block.out_channels != constructor['channels'][-1]:
            raise ValueError('MONAI bottleneck channel mismatch')
        setattr(parent, name, BottleneckAdapter(block, block.out_channels, self.relation_config, relation_enabled))

    @property
    def bottleneck(self):
        return self.network.get_submodule(BOTTLENECK_PATH)

    def forward(self, image):
        return self._forward(image)[0]

    @property
    def relation_scale_enabled(self):
        return self.relation_enabled and self.bottleneck.fusion.learnable_relation_scale

    def forward_with_relation_diagnostics(self, image):
        """Return (normal output, scalar diagnostics); no activation cache survives."""
        return self._forward(image, collect=True)

    def _forward(self, image, *, collect=False):
        if not self.relation_enabled:
            return super().forward(image), {}
        adapter = self.bottleneck
        if adapter._coarse_sink is not None:
            raise RuntimeError('concurrent/reentrant forwards on one adapter are unsupported')
        coarse = []
        stats = {}
        adapter._coarse_sink = coarse
        adapter._relation_diagnostics_sink = stats if collect and self.relation_scale_enabled else None
        try:
            final = super().forward(image).final_logits
            if len(coarse) != 1:
                raise RuntimeError('expected exactly one whole-volume bottleneck evaluation')
            return SegmentorOutput(coarse[0], final), stats
        finally:
            adapter._coarse_sink = None
            adapter._relation_diagnostics_sink = None
            coarse.clear()

    def diagnostic_parameter_groups(self):
        """Disjoint original encoder/decoder and added-module parameter groups."""
        decoder_ids = set()
        level = self.network.model
        while isinstance(level, nn.Sequential):
            decoder_ids.update(id(p) for p in level[2].parameters())
            level = level[1].submodule
        prefix = 'network.' + BOTTLENECK_PATH + '.'
        groups = {name: [] for name in ('encoder', 'decoder', 'coarse_head', 'relation', 'node_to_space', 'fusion')}
        for name, parameter in self.named_parameters():
            local = name.removeprefix(prefix).split('.')[0] if name.startswith(prefix) else None
            group = local if local in groups else 'decoder' if id(parameter) in decoder_ids else 'encoder'
            groups[group].append((name, parameter))
        return {name: params for name, params in groups.items() if params}
