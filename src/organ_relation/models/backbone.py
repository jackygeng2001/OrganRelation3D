"""Configurable full-volume 3D U-Net backbone, without organ relation modules."""
from __future__ import annotations
from typing import NamedTuple
import torch
from torch import Tensor,nn
from torch.nn import functional as functional
from .backbone_config import BackboneConfig,spatial_shape


class EncoderFeatures(NamedTuple):
    deepest: Tensor
    skips: tuple[Tensor,...]  # High -> low resolution, excluding deepest F.


def _check_tensor(value,name,channels):
    if not isinstance(value,Tensor) or value.ndim!=5:
        raise ValueError(f'{name} must be a [B,C,D,H,W] tensor')
    if value.shape[0]<1 or value.shape[1]!=channels or min(value.shape[2:])<1:
        raise ValueError(f'{name} requires B>=1, C={channels}, and positive spatial sizes')
    if not value.is_floating_point():
        raise ValueError(f'{name} must use a floating dtype')


class ConvBlock3D(nn.Module):
    def __init__(self,in_channels,out_channels,config,stride=(1,1,1)):
        super().__init__()
        def norm():
            return nn.GroupNorm(config.norm_groups,out_channels,eps=config.norm_eps) if config.normalization=='group' else nn.Identity()
        self.layers=nn.Sequential(
            nn.Conv3d(in_channels,out_channels,3,stride=stride,padding=1,bias=config.conv_bias),
            norm(),nn.LeakyReLU(config.negative_slope,inplace=False),
            nn.Conv3d(out_channels,out_channels,3,padding=1,bias=config.conv_bias),
            norm(),nn.LeakyReLU(config.negative_slope,inplace=False))

    def forward(self,x):
        return self.layers(x)


def resize_to_skip(x:Tensor,skip_spatial,*,align_corners:bool)->Tensor:
    """Resize feature field to actual skip size, never crop or pad the input.

    This is explicit trilinear feature interpolation, not a physical inverse of
    a strided convolution. See backbone_stage1.md for sampling phase equations.
    """
    return functional.interpolate(x,size=spatial_shape(skip_spatial),mode='trilinear',align_corners=align_corners)


class Encoder3D(nn.Module):
    def __init__(self,config:BackboneConfig):
        super().__init__();self.config=config
        blocks=[]
        for level,width in enumerate(config.channels):
            blocks.append(ConvBlock3D(1 if level==0 else config.channels[level-1],width,config,
                                      (1,1,1) if level==0 else config.downsample_strides[level-1]))
        self.blocks=nn.ModuleList(blocks)

    def forward(self,image:Tensor)->EncoderFeatures:
        _check_tensor(image,'image',1)
        self.config.validate_spatial(tuple(image.shape[2:]))
        skips=[];x=image
        for level,block in enumerate(self.blocks):
            x=block(x)
            if level<len(self.blocks)-1:
                skips.append(x)
        return EncoderFeatures(x,tuple(skips))


class Decoder3D(nn.Module):
    def __init__(self,config:BackboneConfig):
        super().__init__();self.config=config
        self.blocks=nn.ModuleList(ConvBlock3D(config.channels[level+1]+config.channels[level],config.channels[level],config)
                                 for level in reversed(range(len(config.channels)-1)))
        self.logits=nn.Conv3d(config.channels[0],16,1,bias=config.head_bias)

    def _validate_features(self,deepest,skips):
        _check_tensor(deepest,'deepest',self.config.channels[-1])
        if not isinstance(skips,(tuple,list)) or len(skips)!=len(self.config.channels)-1:
            raise ValueError('skips must contain each encoder scale, high to low, excluding deepest')
        for level,skip in enumerate(skips):
            _check_tensor(skip,f'skips[{level}]',self.config.channels[level])
            if skip.shape[0]!=deepest.shape[0] or skip.device!=deepest.device or skip.dtype!=deepest.dtype:
                raise ValueError('deepest/skips must share batch, device and dtype')
        expected=self.config.validate_spatial(tuple(skips[0].shape[2:]))
        if tuple(deepest.shape[2:])!=expected[-1] or any(tuple(skip.shape[2:])!=shape for skip,shape in zip(skips,expected[:-1])):
            raise ValueError('feature pyramid does not match configured strides and full-resolution skip')

    def forward(self,deepest:Tensor,skips:tuple[Tensor,...])->Tensor:
        """Accept original F now, or same-shaped Fprime after future graph fusion.

        All features retain gradients and are unmodified; no label argument.
        The full-resolution first skip determines the exact output extent.
        """
        self._validate_features(deepest,skips)
        x=deepest
        for block,skip in zip(self.blocks,reversed(skips)):
            x=resize_to_skip(x,tuple(skip.shape[2:]),align_corners=self.config.align_corners)
            x=block(torch.cat((x,skip),dim=1))
        return self.logits(x)


class UNetBackbone3D(nn.Module):
    """Encoder->Decoder integration harness, NOT the complete proposed method.

    Future segmentor must explicitly call encoder, coarse/graph/fusion modules,
    then decoder(Fprime, skips); this convenience forward only tests backbone.
    """
    def __init__(self,config:BackboneConfig):
        super().__init__();self.config=config
        self.encoder=Encoder3D(config)
        self.decoder=Decoder3D(config)

    def forward(self,image:Tensor)->Tensor:
        features=self.encoder(image)
        return self.decoder(features.deepest,features.skips)
