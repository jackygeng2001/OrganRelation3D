"""Device-independent, explicit backbone configuration and shape algebra."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import math


def spatial_shape(value):
    shape=tuple(value)
    if len(shape)!=3 or any(type(n) is not int or n<1 for n in shape):
        raise ValueError('spatial shape must contain three positive integers (D,H,W)')
    return shape


@dataclass(frozen=True)
class BackboneConfig:
    """No implicit production widths/depth; callers must supply an experiment config.

    v1 block topology is two 3x3 convolutions, first optionally strided; there is
    no graph, coarse head, softmax, loss, input crop/padding, or validity mask.
    """
    channels: tuple[int,...]
    downsample_strides: tuple[tuple[int,int,int],...]
    normalization: str
    norm_groups: int
    norm_eps: float
    negative_slope: float
    conv_bias: bool
    head_bias: bool
    align_corners: bool

    def __post_init__(self):
        object.__setattr__(self,'channels',tuple(self.channels))
        object.__setattr__(self,'downsample_strides',tuple(tuple(s) for s in self.downsample_strides))
        if len(self.channels)<2 or any(type(c) is not int or c<1 for c in self.channels):
            raise ValueError('at least two positive integer channel widths are required')
        if len(self.downsample_strides)!=len(self.channels)-1:
            raise ValueError('one downsample stride is required between each adjacent scale')
        if any(len(s)!=3 or any(type(n) is not int or n not in (1,2) for n in s) or s==(1,1,1) for s in self.downsample_strides):
            raise ValueError('each stride must contain 1 or 2 and downsample at least one axis')
        if self.normalization not in ('group','none'):
            raise ValueError('normalization must be group or none')
        if type(self.norm_groups) is not int or self.norm_groups<1:
            raise ValueError('norm_groups must be a positive integer')
        if self.normalization=='group' and any(c%self.norm_groups for c in self.channels):
            raise ValueError('every channel width must be divisible by norm_groups')
        if not math.isfinite(self.norm_eps) or self.norm_eps<=0:
            raise ValueError('norm_eps must be finite and positive')
        if not math.isfinite(self.negative_slope) or not 0<self.negative_slope<1:
            raise ValueError('negative_slope must be in (0,1)')
        for field in ('conv_bias','head_bias','align_corners'):
            if type(getattr(self,field)) is not bool:
                raise ValueError(f'{field} must be bool')

    def to_dict(self):
        return asdict(self)

    def spatial_pyramid(self,input_spatial):
        """Conv3d(k=3,p=1,dilation=1): out=ceil(in/stride) per axis."""
        shapes=[spatial_shape(input_spatial)]
        for stride in self.downsample_strides:
            shapes.append(tuple((n+s-1)//s for n,s in zip(shapes[-1],stride)))
        return tuple(shapes)

    def validate_spatial(self,input_spatial):
        shapes=self.spatial_pyramid(input_spatial)
        if self.normalization=='group':
            for level,(c,shape) in enumerate(zip(self.channels,shapes)):
                # Each sample/group needs >1 value; increasing batch must not
                # silently permit a degenerate per-sample normalization group.
                if c//self.norm_groups*math.prod(shape)<=1:
                    raise ValueError(f'GroupNorm needs >1 value per sample/group at scale {level}; got {shape}')
        return shapes
