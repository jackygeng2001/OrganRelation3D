"""Exact dense Conv3d via bounded batches of native NCHW Conv2d.

This is a convolution execution backend, not a patch model. Every depth-kernel
plane contributes with the original stride/padding/dilation; all spatial output
positions are assembled before the next network layer sees the result. Only the
original input and weight are saved. No network forward is recomputed backward.
"""
import torch
from torch import nn
from torch.nn import functional as F

DEPTH_CHUNK = 4
# Optional reduced-precision arithmetic for the 3x3x3 kernels only. Inputs and
# weights are rounded per plane batch; every partial result, output and
# gradient is accumulated into the original FP32 buffers.
COMPUTE_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16}


def _depth_slice(depth, kernel_index, stride, padding, dilation, start, end):
    # Output j reads input j*stride-padding+kernel_index*dilation.
    # Out-of-image contributions are zero, exactly as zero-padded Conv3d.
    shift = kernel_index * dilation - padding
    lo = max(start, (-shift + stride - 1) // stride)
    hi = min(end, (depth - 1 - shift) // stride + 1)
    return lo, hi, slice(lo * stride + shift, (hi - 1) * stride + shift + 1, stride)


def _planes(x, depth_slice, dtype=None):
    part = x[:, :, depth_slice]
    batch, channels, depth, height, width = part.shape
    if dtype is None:
        # Deliberately NCHW: NHWC can choose the same slow naive MIOpen solvers.
        return part.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, height, width).contiguous()
    planes = torch.empty((batch, depth, channels, height, width), device=x.device, dtype=dtype)
    planes.copy_(part.permute(0, 2, 1, 3, 4))  # One pass: NCHW layout and rounding together.
    return planes.view(batch * depth, channels, height, width)


def _volume(x, batch, depth):
    return x.reshape(batch, depth, x.shape[1], *x.shape[2:]).permute(0, 2, 1, 3, 4)


def _input_planes(x, second, source, dtype=None):
    first = _planes(x, source, dtype)
    return first if second is None else torch.cat((first, _planes(second, source, dtype)), 1)


class _DepthSlicedConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, chunk, second=None, compute_dtype=None):
        batch, _, depth, _, _ = x.shape
        size = tuple((n + 2*p - di*(k-1) - 1)//s + 1
                     for n, p, di, k, s in zip(x.shape[2:], padding, dilation, weight.shape[2:], stride))
        if min(size) < 1:
            raise ValueError('convolution kernel exceeds padded input')
        output = torch.empty((batch, weight.shape[0], *size), device=x.device, dtype=x.dtype,
                             memory_format=torch.channels_last_3d).zero_()
        planes = [weight[:, :, k].contiguous() if compute_dtype is None
                  else weight[:, :, k].to(compute_dtype).contiguous() for k in range(weight.shape[2])]
        for start in range(0, size[0], chunk):
            for k, kernel in enumerate(planes):
                lo, hi, source = _depth_slice(depth, k, stride[0], padding[0], dilation[0],
                                             start, min(start + chunk, size[0]))
                if hi <= lo:
                    continue
                part = F.conv2d(_input_planes(x, second, source, compute_dtype), kernel, stride=stride[1:],
                                padding=padding[1:], dilation=dilation[1:])
                output[:, :, lo:hi].add_(_volume(part, batch, hi-lo))
        if bias is not None:
            output.add_(bias.view(1, -1, 1, 1, 1))
        ctx.save_for_backward(x, weight, second)
        ctx.options = stride, padding, dilation, chunk, compute_dtype
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, second = ctx.saved_tensors
        stride, padding, dilation, chunk, compute_dtype = ctx.options
        batch, _, depth, _, _ = x.shape
        dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
        dsecond = torch.zeros_like(second) if second is not None and ctx.needs_input_grad[7] else None
        dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        planes = [weight[:, :, k].contiguous() if compute_dtype is None
                  else weight[:, :, k].to(compute_dtype).contiguous() for k in range(weight.shape[2])]
        need_image = dx is not None or dsecond is not None
        if need_image or dw is not None:
            for start in range(0, grad_output.shape[2], chunk):
                for k, kernel in enumerate(planes):
                    lo, hi, source = _depth_slice(depth, k, stride[0], padding[0], dilation[0],
                                                 start, min(start + chunk, grad_output.shape[2]))
                    if hi <= lo:
                        continue
                    image = _input_planes(x, second, source, compute_dtype)
                    gradient = _planes(grad_output, slice(lo, hi), compute_dtype)
                    # The same ATen operator used by torch.nn.grad.conv2d_*;
                    # request both derivatives together, without recomputation.
                    gi, gw, _ = torch.ops.aten.convolution_backward.default(
                        gradient, image, kernel, None, list(stride[1:]), list(padding[1:]),
                        list(dilation[1:]), False, [0, 0], 1,
                        (need_image, dw is not None, False))
                    if dx is not None:
                        dx[:, :, source].add_(_volume(gi[:, :x.shape[1]], batch, hi-lo))
                    if dsecond is not None:
                        dsecond[:, :, source].add_(_volume(gi[:, x.shape[1]:], batch, hi-lo))
                    if dw is not None:
                        dw[:, :, k].add_(gw)
        db = (grad_output.sum((0, 2, 3, 4))
              if ctx.has_bias and ctx.needs_input_grad[2] else None)
        result = (dx, dw, db, None, None, None, None, dsecond, None)
        return result[:len(ctx.needs_input_grad)]


class DepthSlicedConv3d(nn.Conv3d):
    """Opt-in FP32/FP64, groups=1, zero-padding Conv3d with unchanged weights."""
    compute_dtype = None  # Optional bf16/fp16 kernel arithmetic; parameters stay FP32.

    def _apply_sliced(self, x, second=None):
        if self.compute_dtype is None:  # Exactly the original FP32 call.
            if second is None:
                return _DepthSlicedConv.apply(x, self.weight, self.bias, self.stride,
                                             self.padding, self.dilation, DEPTH_CHUNK)
            return _DepthSlicedConv.apply(x, self.weight, self.bias, self.stride,
                                         self.padding, self.dilation, DEPTH_CHUNK, second)
        if x.dtype != torch.float32:
            raise ValueError('reduced-precision convolution arithmetic requires FP32 volumes')
        return _DepthSlicedConv.apply(x, self.weight, self.bias, self.stride, self.padding,
                                     self.dilation, DEPTH_CHUNK, second, self.compute_dtype)

    @classmethod
    def from_native(cls, module):
        if (module.groups != 1 or module.padding_mode != 'zeros'
                or isinstance(module.padding, str)):
            raise ValueError('depth_sliced requires groups=1 and numeric zero padding')
        # Do not call Conv3d.__init__: retain Parameter objects, RNG, names and
        # initialization exactly, including an existing optimizer's references.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        for name in ('in_channels', 'out_channels', 'kernel_size', 'stride', 'padding',
                     'dilation', 'groups', 'padding_mode', '_reversed_padding_repeated_twice'):
            setattr(result, name, getattr(module, name))
        result.transposed, result.output_padding = False, (0, 0, 0)
        result.register_parameter('weight', module.weight)
        result.register_parameter('bias', module.bias)
        result.train(module.training)
        return result

    def forward(self, x, second=None):
        if second is not None:
            return self.forward_concatenated(x, second)
        if self.groups != 1 or self.padding_mode != 'zeros' or isinstance(self.padding, str):
            raise ValueError('depth_sliced requires groups=1 and numeric zero padding')
        if (x.ndim != 5 or x.dtype not in (torch.float32, torch.float64)
                or torch.is_autocast_enabled(x.device.type)):
            raise ValueError('depth_sliced requires batched FP32/FP64 input and autocast disabled')
        if self.kernel_size == (1, 1, 1) and self.stride == (1, 1, 1) and self.padding == (0, 0, 0):
            # A pointwise convolution is a channel-wise linear map.
            return F.linear(x.movedim(1, -1), self.weight[:, :, 0, 0, 0], self.bias).movedim(-1, 1)
        return self._apply_sliced(x)

    def forward_concatenated(self, first, second):
        """Conv3d(cat(first, second), weight) without a saved full-volume cat.

        Only the bounded NCHW plane batch is concatenated. All input channels
        enter the same Conv2d call, retaining its channel-reduction order.
        Both original inputs are saved and receive independent exact gradients.
        """
        if (self.groups != 1 or self.padding_mode != 'zeros' or isinstance(self.padding, str)
                or first.ndim != 5 or second.ndim != 5
                or first.shape[0] != second.shape[0] or first.shape[2:] != second.shape[2:]
                or first.shape[1] + second.shape[1] != self.in_channels
                or first.dtype not in (torch.float32, torch.float64)
                or second.dtype != first.dtype or second.device != first.device
                or torch.is_autocast_enabled(first.device.type)):
            raise ValueError('concatenated convolution requires compatible FP32/FP64 volumes and groups=1 zero padding')
        return self._apply_sliced(first, second)


def enable_depth_sliced_convolutions(module, compute_dtype=None):
    """Configure a newly built BraTS model, before adding any runtime hooks.

    compute_dtype (None, 'bfloat16' or 'float16') applies only to kernels larger
    than 1x1x1; pointwise projections keep FP32 linear maps.
    """
    if compute_dtype is not None and compute_dtype not in COMPUTE_DTYPES:
        raise ValueError('convolution compute dtype must be null, bfloat16 or float16')
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv3d):
            if child._forward_hooks or child._forward_pre_hooks or child._backward_hooks:
                raise ValueError('configure convolution execution before registering hooks')
            converted = DepthSlicedConv3d.from_native(child)
            if compute_dtype is not None:
                converted.compute_dtype = COMPUTE_DTYPES[compute_dtype]
            setattr(module, name, converted)
        else:
            enable_depth_sliced_convolutions(child, compute_dtype)
