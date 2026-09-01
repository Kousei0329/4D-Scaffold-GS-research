import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np

import _gridencoder as _backend

anchor_round_digits = 16
Q_anchor = 1 / (2 ** anchor_round_digits - 1)
use_clamp = True


class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = torch.clamp(input, min=-1, max=1)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q, input_mean=None):
        if use_clamp:
            if input_mean is None:
                input_mean = input.mean()
            input_min = input_mean - 15_000 * Q
            input_max = input_mean + 15_000 * Q
            input = torch.clamp(input, min=input_min.detach(), max=input_max.detach())

        Q_round = torch.round(input / Q)
        Q_q = Q_round * Q
        return Q_q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, min_level_id=None, n_levels_calc=1, binary_vxl=None, PV=0):
        # inputs: [N, num_dim], float in [0, 1]
        # embeddings: [sO, n_features], float
        inputs = inputs.contiguous()

        Rb = 128
        if binary_vxl is not None:
            binary_vxl = binary_vxl.contiguous()
            Rb = binary_vxl.shape[-1]
            assert len(binary_vxl.shape) == inputs.shape[-1]

        N, num_dim = inputs.shape
        n_levels = offsets_list.shape[0] - 1
        n_features = embeddings.shape[1]

        max_level_id = min_level_id + n_levels_calc

        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings.to(torch.half)

        outputs = torch.empty(n_levels_calc, N, n_features, device=inputs.device, dtype=embeddings.dtype)

        if calc_grad_inputs:
            dy_dx = torch.empty(N, n_levels_calc * num_dim * n_features, device=inputs.device, dtype=embeddings.dtype)
        else:
            dy_dx = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_forward(
                inputs, embeddings,
                offsets_list[min_level_id:max_level_id + 1],
                resolutions_list[min_level_id:max_level_id],
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx, binary_vxl, None
            )
        else:
            _backend.grid_encode_forward(
                inputs, embeddings,
                offsets_list, resolutions_list,
                outputs,
                N, num_dim, n_features, n_levels_calc, 0, Rb, PV,
                dy_dx, binary_vxl, min_level_id
            )

        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels_calc * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl)
        ctx.dims = [N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV]

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        inputs, embeddings, offsets_list, resolutions_list, dy_dx, binary_vxl = ctx.saved_tensors
        N, num_dim, n_features, n_levels_calc, min_level_id, max_level_id, Rb, PV = ctx.dims

        grad = grad.view(N, n_levels_calc, n_features).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        if isinstance(min_level_id, int):
            _backend.grid_encode_backward(
                grad, inputs, embeddings,
                offsets_list[min_level_id:max_level_id + 1],
                resolutions_list[min_level_id:max_level_id],
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx, grad_inputs, binary_vxl, None
            )
        else:
            _backend.grid_encode_backward(
                grad, inputs, embeddings,
                offsets_list, resolutions_list,
                grad_embeddings,
                N, num_dim, n_features, n_levels_calc, 0, Rb,
                dy_dx, grad_inputs, binary_vxl, min_level_id
            )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    """3D multi-resolution hash grid encoder (ported from HAC's utils/encodings.py).

    Used as a per-axis-triplet encoder: a 4D anchor (x, y, z, t) is split into
    several 3D coordinate triplets, and one GridEncoder is applied per triplet.
    """

    def __init__(self,
                 num_dim=3,
                 n_features=2,
                 resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736),
                 log2_hashmap_size=19,
                 ste_binary=True,
                 ste_multistep=False,
                 add_noise=False,
                 Q=1
                 ):
        super().__init__()

        resolutions_list = torch.tensor(resolutions_list).to(torch.int)
        n_levels = resolutions_list.numel()

        self.num_dim = num_dim
        self.n_levels = n_levels
        self.n_features = n_features
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary
        self.ste_multistep = ste_multistep
        self.add_noise = add_noise
        self.Q = Q

        offsets_list = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim)
            params_in_level = int(np.ceil(params_in_level / 8) * 8)
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)

        self.n_params = offsets_list[-1] * n_features

        self.params = nn.Parameter(torch.empty(offset, n_features))

        self.reset_parameters()

        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 1e-4
        self.params.data.uniform_(-std, std)

    def forward(self, inputs, min_level_id=None, max_level_id=None, test_phase=False, binary_vxl=None, PV=0):
        # inputs: [..., num_dim], normalized to [0, 1]
        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        params = self.params

        if self.ste_binary:
            embeddings = STE_binary.apply(params)
        elif self.add_noise and not test_phase:
            embeddings = params + (torch.rand_like(params) - 0.5) * (1 / self.Q)
        elif self.ste_multistep or (self.add_noise and test_phase):
            embeddings = STE_multistep.apply(params, self.Q)
        else:
            embeddings = params

        min_level_id = 0 if min_level_id is None else max(min_level_id, 0)
        max_level_id = self.n_levels if max_level_id is None else min(max_level_id, self.n_levels)
        n_levels_calc = max_level_id - min_level_id

        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list,
                               inputs.requires_grad, min_level_id, n_levels_calc, binary_vxl, PV)

        outputs = outputs.view(prefix_shape + [n_levels_calc * self.n_features])

        return outputs
