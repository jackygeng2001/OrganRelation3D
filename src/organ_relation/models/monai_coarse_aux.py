"""Coarse-supervision ablation: the official MONAI decoder receives unchanged F."""
from torch import nn

from .coarse_head import CoarseHead
from .monai_reference import MonaiReferenceUNet
from .monai_relation import BOTTLENECK_PATH
from .segmentor import SegmentorOutput


class CoarseAuxAdapter(nn.Module):
    def __init__(self, block, *, channels, bias):
        super().__init__()
        self.block = block
        self.channels = channels
        self.coarse_head = CoarseHead(channels, bias=bias)
        self._coarse_sink = None

    def forward(self, image):
        features = self.block(image)
        if self._coarse_sink is None:
            raise RuntimeError('coarse adapter must run inside MonaiCoarseAuxUNet.forward')
        if features.ndim != 5 or features.shape[1] != self.channels:
            raise ValueError('unexpected MONAI deepest feature shape')
        self._coarse_sink.append(self.coarse_head(features).logits)
        return features  # SAME tensor object; no fusion, clone, detach, gate or layout conversion.


class MonaiCoarseAuxUNet(MonaiReferenceUNet):
    """Independent mode; does not reinterpret MonaiRelationUNet's disabled bypass."""
    def __init__(self, constructor, input_processing, *, coarse_head):
        # Initialize the complete official backbone first, exactly as in A and C.
        super().__init__(constructor, input_processing)
        from monai.networks.blocks import ResidualUnit
        from monai.networks.layers.simplelayers import SkipConnection
        self.coarse_config = dict(coarse_head)
        if (set(coarse_head) != {'feature_channels', 'bias'}
                or coarse_head['feature_channels'] != constructor['channels'][-1]):
            raise ValueError('explicit coarse feature_channels must match MONAI bottleneck')
        if len(constructor['channels']) != 5 or list(constructor['strides']) != [2]*4:
            raise ValueError('adapter requires the inspected five-level MONAI hierarchy')
        parent_path, _, name = BOTTLENECK_PATH.rpartition('.')
        parent = self.network.get_submodule(parent_path)
        block = getattr(parent, name)
        if (not isinstance(parent, SkipConnection) or parent.mode != 'cat'
                or not isinstance(block, ResidualUnit)
                or block.out_channels != coarse_head['feature_channels']):
            raise ValueError('unsupported MONAI bottleneck structure')
        setattr(parent, name, CoarseAuxAdapter(block, channels=block.out_channels, bias=coarse_head['bias']))

    @property
    def bottleneck(self):
        return self.network.get_submodule(BOTTLENECK_PATH)

    def forward(self, image):
        adapter = self.bottleneck
        if adapter._coarse_sink is not None:
            raise RuntimeError('concurrent/reentrant forwards on one adapter are unsupported')
        coarse = []
        adapter._coarse_sink = coarse
        try:
            final = super().forward(image).final_logits
            if len(coarse) != 1:
                raise RuntimeError('expected exactly one whole-volume bottleneck evaluation')
            return SegmentorOutput(coarse[0], final)
        finally:
            adapter._coarse_sink = None
            coarse.clear()

    def diagnostic_parameter_groups(self):
        decoder_ids = set()
        level = self.network.model
        while isinstance(level, nn.Sequential):
            decoder_ids.update(id(p) for p in level[2].parameters())
            level = level[1].submodule
        prefix = 'network.' + BOTTLENECK_PATH + '.coarse_head.'
        groups = {name: [] for name in ('encoder', 'decoder', 'coarse_head')}
        for name, parameter in self.named_parameters():
            group = 'coarse_head' if name.startswith(prefix) else 'decoder' if id(parameter) in decoder_ids else 'encoder'
            groups[group].append((name, parameter))
        return groups
