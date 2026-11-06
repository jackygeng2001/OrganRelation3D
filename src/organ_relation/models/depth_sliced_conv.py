"""Exact dense Conv3d via bounded batches of native NCHW Conv2d.

This is a convolution execution backend, not a patch model. Every depth-kernel
plane contributes with the original stride/padding/dilation; all spatial output
positions are assembled before the next network layer sees the result. Only the
original input and weight are saved. No network forward is recomputed backward.
In-plane dilated 'same' kernels run as interleaved ordinary convolutions
(POLYPHASE_DILATION), the same sums in a different order.
"""
import torch
from torch import nn
from torch.nn import functional as F

DEPTH_CHUNK = 4
POLYPHASE_DILATION = 'interleaved_v1'  # Recorded in run identity: see _polyphase.
# Optional reduced-precision arithmetic for the 3x3x3 kernels only. Inputs and
# weights are rounded per plane batch; every partial result, output and
# gradient is accumulated into the original FP32 buffers.
COMPUTE_DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16}
# Whole-network mixed precision instead stores volumes in bfloat16. Kernel-plane
# partial sums are then accumulated per depth chunk in FP32 and rounded once.
VOLUME_DTYPES = (torch.float32, torch.float64, torch.bfloat16)


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


def _polyphase(kernel, stride, padding, dilation):
    """True for an in-plane 'same' dilated kernel: stride 1, padding dilation*(k-1)/2.

    Such a convolution only combines positions of equal residue modulo the
    dilation, so it splits exactly into dilation_h*dilation_w ordinary
    (dilation 1) convolutions on interleaved sub-images with the same weights.
    MIOpen has no fast solver for dilated kernels and falls back to a naive one.
    """
    return (any(d > 1 for d in dilation) and all(s == 1 for s in stride)
            and all(k % 2 == 1 and p == d * (k - 1) // 2
                    for k, p, d in zip(kernel.shape[2:], padding, dilation)))


def _phases(planes, dilation):
    for a in range(dilation[0]):
        for b in range(dilation[1]):
            if a < planes.shape[2] and b < planes.shape[3]:
                yield (slice(None), slice(None), slice(a, None, dilation[0]), slice(b, None, dilation[1]))


def _conv2d(planes, kernel, stride, padding, dilation):
    """F.conv2d over a plane batch; dilated 'same' kernels run as interleaved ordinary convolutions."""
    if not _polyphase(kernel, stride, padding, dilation):
        return F.conv2d(planes, kernel, stride=stride, padding=padding, dilation=dilation)
    plain = tuple((k - 1) // 2 for k in kernel.shape[2:])
    output = planes.new_empty((planes.shape[0], kernel.shape[0], *planes.shape[2:]))
    for phase in _phases(planes, dilation):
        output[phase] = F.conv2d(planes[phase].contiguous(), kernel, padding=plain)
    return output


def _conv2d_backward(gradient, planes, kernel, stride, padding, dilation, mask):
    """(input, weight) gradients of _conv2d; the weight gradient of the phases is summed in >= FP32."""
    if not _polyphase(kernel, stride, padding, dilation):
        gi, gw, _ = torch.ops.aten.convolution_backward.default(
            gradient, planes, kernel, None, list(stride), list(padding), list(dilation),
            False, [0, 0], 1, mask)
        return gi, gw
    plain = [(k - 1) // 2 for k in kernel.shape[2:]]
    gi = planes.new_empty(planes.shape) if mask[0] else None
    gw = None
    for phase in _phases(planes, dilation):
        part_i, part_w, _ = torch.ops.aten.convolution_backward.default(
            gradient[phase].contiguous(), planes[phase].contiguous(), kernel, None, [1, 1], plain, [1, 1],
            False, [0, 0], 1, mask)
        if mask[0]:
            gi[phase] = part_i
        if mask[1]:
            part_w = part_w.to(torch.promote_types(part_w.dtype, torch.float32))
            gw = part_w if gw is None else gw.add_(part_w)
    return gi, gw


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
        ctx.save_for_backward(x, weight, second)
        ctx.options = stride, padding, dilation, chunk, compute_dtype
        ctx.has_bias = bias is not None
        if x.dtype == torch.bfloat16:
            return _reduced_forward(x, weight, bias, stride, padding, dilation, chunk, second, size)
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
                part = _conv2d(_input_planes(x, second, source, compute_dtype), kernel,
                               stride[1:], padding[1:], dilation[1:])
                output[:, :, lo:hi].add_(_volume(part, batch, hi-lo))
        if bias is not None:
            output.add_(bias.view(1, -1, 1, 1, 1))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, second = ctx.saved_tensors
        stride, padding, dilation, chunk, compute_dtype = ctx.options
        if x.dtype == torch.bfloat16:
            dx, dw, db, dsecond = _reduced_backward(ctx, x, weight, second, grad_output,
                                                    stride, padding, dilation, chunk)
            result = (dx, dw, db, None, None, None, None, dsecond, None)
            return result[:len(ctx.needs_input_grad)]
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
                    gi, gw = _conv2d_backward(gradient, image, kernel, stride[1:], padding[1:],
                                              dilation[1:], (need_image, dw is not None, False))
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


def _reduced_forward(x, weight, bias, stride, padding, dilation, chunk, second, size):
    """bf16 volumes: bf16 kernel arithmetic, FP32 accumulation, one rounding per output."""
    batch = x.shape[0]
    planes = [weight[:, :, k].to(x.dtype).contiguous() for k in range(weight.shape[2])]
    output = torch.empty((batch, weight.shape[0], *size), device=x.device, dtype=x.dtype,
                         memory_format=torch.channels_last_3d)
    for start in range(0, size[0], chunk):
        end = min(start + chunk, size[0])
        total = torch.empty((batch, weight.shape[0], end - start, *size[1:]), device=x.device,
                            dtype=torch.float32, memory_format=torch.channels_last_3d).zero_()
        for k, kernel in enumerate(planes):
            lo, hi, source = _depth_slice(x.shape[2], k, stride[0], padding[0], dilation[0], start, end)
            if hi <= lo:
                continue
            part = _conv2d(_input_planes(x, second, source), kernel, stride[1:], padding[1:], dilation[1:])
            total[:, :, lo-start:hi-start].add_(_volume(part, batch, hi-lo))
        if bias is not None:
            total.add_(bias.view(1, -1, 1, 1, 1))
        output[:, :, start:end].copy_(total)
    return output


def _reduced_backward(ctx, x, weight, second, grad_output, stride, padding, dilation, chunk):
    """Input gradients of one output chunk are summed in FP32 before touching the bf16 buffer.

    An input plane is then rounded once per output chunk that reads it (at most
    twice for these 3x3x3 kernels). Weight/bias gradients accumulate in FP32.
    """
    batch, channels, depth = x.shape[:3]
    grad_output = grad_output.to(x.dtype)
    dx = torch.zeros_like(x) if ctx.needs_input_grad[0] else None
    dsecond = torch.zeros_like(second) if second is not None and ctx.needs_input_grad[7] else None
    dw = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
    need_image = dx is not None or dsecond is not None
    planes = [weight[:, :, k].to(x.dtype).contiguous() for k in range(weight.shape[2])]
    if need_image or dw is not None:
        inputs = channels + (0 if second is None else second.shape[1])
        for start in range(0, grad_output.shape[2], chunk):
            end = min(start + chunk, grad_output.shape[2])
            work = []
            for k in range(len(planes)):
                lo, hi, source = _depth_slice(depth, k, stride[0], padding[0], dilation[0], start, end)
                if hi > lo:
                    work.append((k, lo, hi, source))
            if not work:
                continue
            first = min(source.start for _, _, _, source in work)
            last = max(source.stop for _, _, _, source in work)
            total = (torch.empty((batch, inputs, last - first, *x.shape[3:]), device=x.device,
                                 dtype=torch.float32, memory_format=torch.channels_last_3d).zero_()
                     if need_image else None)
            for k, lo, hi, source in work:
                image = _input_planes(x, second, source)
                gradient = _planes(grad_output, slice(lo, hi))
                gi, gw = _conv2d_backward(gradient, image, planes[k], stride[1:], padding[1:],
                                          dilation[1:], (need_image, dw is not None, False))
                if need_image:
                    shifted = slice(source.start - first, source.stop - first, source.step)
                    total[:, :, shifted].add_(_volume(gi, batch, hi-lo))
                if dw is not None:
                    dw[:, :, k].add_(gw)
            if dx is not None:
                dx[:, :, first:last].add_(total[:, :channels])
            if dsecond is not None:
                dsecond[:, :, first:last].add_(total[:, channels:])
    db = (grad_output.sum((0, 2, 3, 4), dtype=torch.float32)
          if ctx.has_bias and ctx.needs_input_grad[2] else None)
    return dx, dw, db, dsecond


class DepthSlicedConv3d(nn.Conv3d):
    """Opt-in groups=1, zero-padding Conv3d with unchanged FP32/FP64 weights.

    FP32/FP64 volumes keep the original arithmetic. bfloat16 volumes (whole-network
    mixed precision) round weights per call and return bfloat16 outputs.
    """
    compute_dtype = None  # Optional bf16/fp16 kernel arithmetic; parameters stay FP32.

    def _apply_sliced(self, x, second=None):
        if x.dtype == torch.bfloat16:
            if self.compute_dtype is not None:
                raise ValueError('choose either convolution-only or whole-network reduced precision')
            return _DepthSlicedConv.apply(x, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, DEPTH_CHUNK, second)
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
        if (x.ndim != 5 or x.dtype not in VOLUME_DTYPES
                or torch.is_autocast_enabled(x.device.type)):
            raise ValueError('depth_sliced requires batched FP32/FP64/BF16 input and autocast disabled')
        if self.kernel_size == (1, 1, 1) and self.stride == (1, 1, 1) and self.padding == (0, 0, 0):
            # A pointwise convolution is a channel-wise linear map. For bf16 volumes
            # the FP32 weights are rounded per call; FP32/FP64 calls are unchanged.
            weight, bias = self.weight[:, :, 0, 0, 0], self.bias
            if x.dtype == torch.bfloat16:
                weight, bias = weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
            return F.linear(x.movedim(1, -1), weight, bias).movedim(-1, 1)
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
                or first.dtype not in VOLUME_DTYPES
                or second.dtype != first.dtype or second.device != first.device
                or torch.is_autocast_enabled(first.device.type)):
            raise ValueError('concatenated convolution requires compatible FP32/FP64/BF16 volumes and groups=1 zero padding')
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
