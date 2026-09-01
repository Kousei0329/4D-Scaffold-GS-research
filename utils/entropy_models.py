import torch
import torch.nn as nn
from utils.encodings import use_clamp


class Low_bound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        x = torch.clamp(x, min=1e-6)
        return x

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        grad1 = g.clone()
        grad1[x < 1e-6] = 0
        pass_through_if = (x >= 1e-6) | (g < 0.0)
        return grad1 * pass_through_if.float()


class Entropy_gaussian(nn.Module):
    """Differentiable bit-cost of quantized values under a per-element Gaussian prior.

    Ported from HAC's utils/entropy_models.py (single-Gaussian variant, no GMM).
    """

    def __init__(self, Q=1):
        super(Entropy_gaussian, self).__init__()
        self.Q = Q

    def forward(self, x, mean, scale, Q=None, x_mean=None):
        if Q is None:
            Q = self.Q

        if use_clamp:
            if x_mean is None:
                x_mean = x.mean()
            x_min = x_mean - 15_000 * Q
            x_max = x_mean + 15_000 * Q
            x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())

        scale = torch.clamp(scale, min=1e-9)
        m = torch.distributions.normal.Normal(mean, scale)
        lower = m.cdf(x - 0.5 * Q)
        upper = m.cdf(x + 0.5 * Q)
        likelihood = torch.abs(upper - lower)
        likelihood = Low_bound.apply(likelihood)
        bits = -torch.log2(likelihood)
        return bits


def get_binary_vxl_size(binary_vxl):
    """Theoretical Bernoulli-source bit cost of a {0,1} tensor, ported from HAC's
    utils/encodings.py. Used both as a differentiable rate-loss term (pushes a binarized
    tensor's values to be skewed toward 0 or 1, which is what makes it compressible below
    1 bit/element) and, at encode time, as the entropy estimate a real Bernoulli arithmetic
    coder (utils/arithmetic_coding.py: encode_binary/decode_binary) can actually achieve."""
    ttl_num = binary_vxl.numel()
    pos_num = torch.sum(binary_vxl)
    neg_num = ttl_num - pos_num

    Pg = torch.clamp(pos_num / ttl_num, min=1e-6, max=1 - 1e-6)
    pos_bit = pos_num * (-torch.log2(Pg))
    neg_bit = neg_num * (-torch.log2(1 - Pg))
    ttl_bit = pos_bit + neg_bit
    ttl_bit = ttl_bit + 32  # cost of storing Pg itself
    return Pg, ttl_bit, ttl_bit.item() / 8.0 / 1024 / 1024, ttl_num
