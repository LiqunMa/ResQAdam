import copy
import math

import torch
import torch.nn as nn
from torch.autograd import Function

SCALE_MIN = 1e-4
SCALE_MAX = 1e4
ZERO_POINT_MIN = -1e4
ZERO_POINT_MAX = 1e4


class ScaleGradient(Function):
    """Identity in the forward pass, scales the gradient in the backward pass (LSQ)."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, output_grad):
        return output_grad * ctx.scale, None


class STEClamp(Function):
    @staticmethod
    def forward(ctx, x, clip_min, clip_max):
        return x.clamp(clip_min, clip_max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


class STERound(Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class BaseLearnableQuantizer(nn.Module):
    """Group-wise weight quantizer with learnable scale / zero point (LSQ-style).

    For n_bits > 2 an asymmetric quantizer with a learnable zero point is used.
    For n_bits == 2 a symmetric 4-level quantizer {-1.5, -0.5, 0.5, 1.5} * scale / 2 is used.
    """

    def __init__(self, n_bits, group_size, x_shape, q_type):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.x_shape = x_shape
        self.group_size = group_size
        scale_shape = (math.prod(x_shape) // group_size, 1)
        self.scale = nn.Parameter(torch.empty(scale_shape), requires_grad=True)
        if self.n_bits != 2:
            self.zero_point = nn.Parameter(torch.empty(scale_shape), requires_grad=True)
        self.grad_scale = (self.group_size * (2 ** (n_bits - 1) - 1)) ** (-0.5)
        self.q_type = q_type

    def init_scale_and_zero_point(self, x: torch.Tensor):
        x = x.reshape(-1, self.group_size)
        with torch.no_grad():
            if self.n_bits != 2:
                xmin = x.amin([-1], keepdim=True)
                xmax = x.amax([-1], keepdim=True)
                scale = (xmax - xmin) / (2 ** self.n_bits - 1)
                scale = scale.clamp(min=SCALE_MIN, max=SCALE_MAX)
                zero_point = -(xmin / scale).clamp(min=ZERO_POINT_MIN, max=ZERO_POINT_MAX)
                self.zero_point.data.copy_(zero_point)
            else:
                scale = x.abs().amax([-1], keepdim=True)
            self.scale.data.copy_(scale)

    def forward(self, x: torch.Tensor):
        dims = x.shape
        x = x.reshape(-1, self.group_size)

        if self.n_bits != 2:
            scale = STEClamp.apply(self.scale, SCALE_MIN, SCALE_MAX)
            scale = ScaleGradient.apply(scale, self.grad_scale)
            round_zero_point = STERound.apply(self.zero_point)
            round_zero_point = STEClamp.apply(round_zero_point, ZERO_POINT_MIN, ZERO_POINT_MAX)

            x_int = STERound.apply(x / scale)
            x_int = x_int.add(round_zero_point)
            x_int = x_int.clamp(self.qmin, self.qmax)

            x_dequant = x_int.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
        else:
            scale = ScaleGradient.apply(self.scale, self.grad_scale)
            clip_val = 1 - 1e-2
            x_int = (
                STERound.apply(torch.clamp(x / scale, -clip_val, clip_val) * 2 - 0.5) + 0.5
            ) / 2
            x_dequant = x_int.mul(scale)

        return x_dequant.reshape(dims)

    @torch.no_grad()
    def infer_fake_quantize(self, x: torch.Tensor):
        dims = x.shape
        x = x.reshape(-1, self.group_size)

        if self.n_bits != 2:
            scale = self.scale.clamp(min=SCALE_MIN, max=SCALE_MAX)
            round_zero_point = self.zero_point.round()
            round_zero_point = round_zero_point.clamp(min=ZERO_POINT_MIN, max=ZERO_POINT_MAX)

            x_int = (x / scale).round()
            x_int = x_int.add(round_zero_point)
            x_int = x_int.clamp(self.qmin, self.qmax)

            x_dequant = x_int.sub(round_zero_point)
            x_dequant = x_dequant.mul(scale)
        else:
            scale = self.scale
            clip_val = 1 - 1e-2
            x_int = (
                torch.round(torch.clamp(x / scale, -clip_val, clip_val) * 2 - 0.5) + 0.5
            ) / 2
            x_dequant = x_int.mul(scale)

        return x_dequant.reshape(dims)


class StaticQuantizer(nn.Module):
    """Dynamic group-wise asymmetric MinMax quantizer (used for activations)."""

    def __init__(self, n_bits, group_size):
        super().__init__()
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits
        self.qmin = 0
        self.qmax = 2 ** (n_bits) - 1
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        dims = x.shape
        x = x.reshape(-1, self.group_size)

        with torch.no_grad():
            xmin = x.amin([-1], keepdim=True)
            xmax = x.amax([-1], keepdim=True)
            scale = (xmax - xmin) / (2 ** self.n_bits - 1)
            scale = scale.clamp(min=SCALE_MIN, max=SCALE_MAX).detach()
            round_zero_point = -(xmin / scale).round()
            round_zero_point = round_zero_point.clamp(min=ZERO_POINT_MIN, max=ZERO_POINT_MAX).detach()

        x_int = STERound.apply(x / scale)
        x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)

        x_dequant = x_int.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        return x_dequant.reshape(dims)


class QLinear(nn.Module):
    """Fake-quantized linear layer. Bit widths >= 16 disable the corresponding quantizer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        qtype: str,
        w_bits: int,
        a_bits: int,
        group_size: int,
        round_type: str = "ste",
        round_param: str = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = has_bias
        self.w_bits = w_bits
        self.a_bits = a_bits
        # group_size == -1 means per-channel quantization
        self.group_size = group_size if group_size != -1 else self.in_features
        assert self.in_features % self.group_size == 0
        self.qtype = qtype

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(out_features), requires_grad=True)

        if self.w_bits < 16:
            if self.qtype == "static":
                self.w_quantizer = StaticQuantizer(n_bits=self.w_bits, group_size=self.group_size)
            else:
                self.w_quantizer = BaseLearnableQuantizer(
                    n_bits=self.w_bits,
                    group_size=self.group_size,
                    x_shape=self.weight.shape,
                    q_type=self.qtype,
                )

        if self.a_bits < 16:
            self.a_quantizer = StaticQuantizer(n_bits=self.a_bits, group_size=self.group_size)

    def forward(self, x: torch.Tensor):
        dequant_weight = self.w_quantizer(self.weight) if self.w_bits < 16 else self.weight
        dequant_act = self.a_quantizer(x) if self.a_bits < 16 else x

        out = nn.functional.linear(dequant_act, dequant_weight)
        if self.has_bias:
            out += self.bias.view(1, -1).expand_as(out)
        return out


class DequantizedQLinear(nn.Module):
    """Inference-time layer that stores the fake-quantized weight directly."""

    def __init__(self, q_linear: QLinear):
        super().__init__()
        self.in_features = q_linear.in_features
        self.out_features = q_linear.out_features
        self.has_bias = q_linear.has_bias
        self.w_bits = q_linear.w_bits
        self.a_bits = q_linear.a_bits
        self.group_size = q_linear.group_size

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features), requires_grad=True)
        if self.has_bias:
            self.bias = nn.Parameter(torch.empty(self.out_features), requires_grad=True)

        assert self.w_bits < 16
        self.w_quantizer = copy.deepcopy(q_linear.w_quantizer)
        self.weight.data.copy_(self.w_quantizer.infer_fake_quantize(q_linear.weight.data))
        if self.has_bias:
            self.bias.data.copy_(q_linear.bias.data)

        if self.a_bits < 16:
            self.a_quantizer = copy.deepcopy(q_linear.a_quantizer)

    def forward(self, x: torch.Tensor):
        dequant_act = self.a_quantizer(x) if self.a_bits < 16 else x
        out = nn.functional.linear(dequant_act, self.weight)
        if self.has_bias:
            out += self.bias.view(1, -1).expand_as(out)
        return out


def init_quant_model_with_hp_model(hp_model, q_model):
    """Copy weights from a full-precision model and initialize quantizer scales from them."""
    q_model.load_state_dict(hp_model.state_dict(), strict=False)
    for _, module in q_model.named_modules():
        if isinstance(module, QLinear) and module.qtype in ["learnable", "lsq"] and module.w_bits < 16:
            module.w_quantizer.init_scale_and_zero_point(module.weight.data)


def replace_qlinear_with_dequantized_linear(q_model):
    for name, module in q_model.named_children():
        if isinstance(module, QLinear):
            setattr(q_model, name, DequantizedQLinear(module))
        else:
            replace_qlinear_with_dequantized_linear(module)
    return q_model
