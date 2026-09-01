#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from utils.encodings import GridEncoder, STE_binary, STE_multistep, anchor_round_digits
from utils.entropy_models import Entropy_gaussian, get_binary_vxl_size
from utils.arithmetic_coding import encode_gaussian_chunk as encode_gaussian, decode_gaussian_chunk as decode_gaussian
from utils.arithmetic_coding import encode_binary, decode_binary
from utils.octree_coding import encode_positions, decode_positions

# 4D anchor axes are (x, y, z, t). Since the existing 3D hash-grid CUDA kernel
# (ported from HAC) can't be fed 4D coordinates directly, we decompose the 4D
# anchor position into every 3-axis combination and run one 3D GridEncoder per
# combination, then concatenate their outputs as the context feature.
HASH_TRIPLETS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]  # xyz, xyt, xzt, yzt

# Initial logit for the learnable per-anchor rate-distortion mask (sigmoid(2.0) ~= 0.88):
# anchors start "mostly open" so useful ones aren't killed before they prove useful.
MASK_INIT_LOGIT = 2.0


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, 
                 feat_dim: int=32, 
                 n_offsets: int=5, 
                 voxel_size: float=0.01,
                 t_grid_size: float=0.0333,
                 update_depth: int=3, 
                 update_init_factor: int=100,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 ratio : int = 1,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 temporal_opacity : bool = False,
                 use_flow : bool = False,
                 sigma_denom_weight : bool = False,
                 disable_denom_weight : bool = False,
                 hparam_beta : float=4.0,
                 max_init_t : float=0.0,
                 use_entropy_coding : bool = False,
                 hash_n_features : int = 2,
                 hash_log2_size : int = 19,
                 noise_start_iter : int = 3_000,
                 entropy_start_iter : int = 10_000,
                 ):

        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.t_grid_size = t_grid_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist

        self.temporal_opacity = temporal_opacity
        self.use_flow = use_flow
        self.sigma_denom_weight = sigma_denom_weight
        self.disable_denom_weight = disable_denom_weight
        self.hparam_beta = hparam_beta
        self.max_init_t = max_init_t

        self.use_entropy_coding = use_entropy_coding
        self.noise_start_iter = noise_start_iter
        self.entropy_start_iter = entropy_start_iter

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        
        self.offset_gradient_accum = torch.empty(0)
        self.offset_time_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.offset_time_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)
                
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3+1, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim+self.opacity_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, n_offsets),
            nn.Tanh()
        ).cuda()

        self.add_cov_dist = add_cov_dist
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim+self.cov_dist_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 8*self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim+3+self.color_dist_dim+self.appearance_dim, feat_dim),
            nn.ReLU(True),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        self.mlp_flow = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(True),
            # nn.Linear(feat_dim, 12*self.n_offsets),
            nn.Linear(feat_dim, 3*self.n_offsets),
            nn.Tanh()
        ).cuda()

        if self.use_entropy_coding:
            self.x_bound_min = torch.zeros((1, 4), device="cuda")
            self.x_bound_max = torch.ones((1, 4), device="cuda")
            self.bounds_ready = False

            self.encoding_xyz = nn.ModuleList([
                GridEncoder(num_dim=3, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                            ste_binary=True, ste_multistep=False, add_noise=False, Q=1)
                for _ in HASH_TRIPLETS
            ]).cuda()

            hash_feat_dim = self.encoding_xyz[0].output_dim * len(HASH_TRIPLETS)
            # mlp_grid predicts a per-anchor Gaussian (mean, scale) for each entropy-coded
            # attribute: anchor_feat (feat_dim), scaling (8-dim), offsets (4*n_offsets-dim),
            # plus one adaptive-quantization-step adjustment logit per attribute group.
            # anchor position (x,y,z,t) itself and mlp_flow / temporal sigma are NOT coded here.
            grid_out_dim = 2 * feat_dim + 2 * 8 + 2 * (4 * self.n_offsets) + 3
            self.mlp_grid = nn.Sequential(
                nn.Linear(hash_feat_dim, feat_dim * 2),
                nn.ReLU(True),
                nn.Linear(feat_dim * 2, grid_out_dim),
            ).cuda()

            self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

    def calc_interp_feat(self, x):
        # x: [N, 4] anchor positions (x, y, z, t)
        assert len(x.shape) == 2 and x.shape[1] == 4
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # -> [0, 1]
        feats = [encoder(x[:, triplet]) for encoder, triplet in zip(self.encoding_xyz, HASH_TRIPLETS)]
        return torch.cat(feats, dim=-1)

    def forward_grid(self, feat_context):
        y = self.mlp_grid(feat_context)
        return torch.split(
            y,
            [self.feat_dim, self.feat_dim, 8, 8, 4 * self.n_offsets, 4 * self.n_offsets, 1, 1, 1],
            dim=-1,
        )

    @staticmethod
    def adaptive_Q(Q_base, Q_adj):
        # HAC-style adaptive quantization step: Q in [0.1*Q_base, 2*Q_base].
        # The floor matters more than it looks: encode_gaussian's arithmetic coder builds
        # a CDF table sized by round(x/Q)'s range, shared across every element in one
        # call. A Q allowed to collapse toward 0 for even a single element can blow that
        # range up by orders of magnitude and OOM the whole batch, so Q is kept within a
        # bounded ratio of Q_base rather than letting tanh saturate all the way to -1.
        return torch.clamp(Q_base * (1 + torch.tanh(Q_adj)), min=Q_base * 0.1)

    def get_encoding_params(self):
        """All 4 hash-grid encoders' raw weights, concatenated and STE-binarized to +-1 --
        i.e. exactly the embeddings actually used at inference (see STE_binary in
        utils/encodings.py). Used both for the hash-grid rate loss (train.py: pushes this
        distribution to skew toward 0 or 1, which is what makes it compressible below 1
        bit/weight) and, at encode time, as the payload for save_hash_grid's real entropy
        coding of that same skew."""
        params = torch.cat([encoder.params.view(-1) for encoder in self.encoding_xyz], dim=0)
        return STE_binary.apply(params)

    def save_hash_grid(self, path):
        """Entropy-code the hash grid weights with a single-probability Bernoulli arithmetic
        coder (utils/arithmetic_coding.py: encode_binary), matching HAC++'s hash-grid rate
        loss (get_binary_vxl_size in train.py) with an actual codec: since training pushes
        get_encoding_params() to skew toward 0 or 1, this achieves the binary entropy of that
        learned skew (< 1 bit/weight) rather than a flat 1 bit/weight from naive packing."""
        shapes = [tuple(encoder.params.shape) for encoder in self.encoding_xyz]
        binary_vals = (self.get_encoding_params() + 1) / 2  # {-1,+1} -> {0,1}
        with open(path, 'wb') as f:
            f.write(np.array([len(shapes)], dtype=np.int64).tobytes())
            for s in shapes:
                f.write(np.array(s, dtype=np.int64).tobytes())
            encode_binary(binary_vals, f)

    def load_hash_grid(self, path):
        with open(path, 'rb') as f:
            n_encoders = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            shapes = [tuple(int(v) for v in np.frombuffer(f.read(16), dtype=np.int64)) for _ in range(n_encoders)]
            binary_vals = decode_binary(f)
        values = binary_vals * 2 - 1  # {0,1} -> {-1,+1}
        offset = 0
        for encoder, shape in zip(self.encoding_xyz, shapes):
            n = shape[0] * shape[1]
            encoder.params.data.copy_(values[offset:offset + n].view(shape))
            offset += n

    @property
    def get_mask_anchor(self):
        # Per-anchor rate-distortion mask: anchors whose learned probability of being
        # "kept" falls below mask_prune_threshold are candidates for physical pruning.
        return torch.sigmoid(self._mask_anchor)

    def update_anchor_bound(self):
        x_bound_min = (torch.min(self._anchor, dim=0, keepdim=True)[0]).detach()
        x_bound_max = (torch.max(self._anchor, dim=0, keepdim=True)[0]).detach()
        for c in range(x_bound_min.shape[-1]):
            x_bound_min[0, c] = x_bound_min[0, c] * 1.2 if x_bound_min[0, c] < 0 else x_bound_min[0, c] * 0.8
        for c in range(x_bound_max.shape[-1]):
            x_bound_max[0, c] = x_bound_max[0, c] * 1.2 if x_bound_max[0, c] > 0 else x_bound_max[0, c] * 0.8
        self.x_bound_min = x_bound_min
        self.x_bound_max = x_bound_max
        self.bounds_ready = True

    def estimate_bits(self, mask_prune_threshold=0.01):
        """Rough bit-size estimate for logging, restricted to anchors that survive the
        rate-distortion mask (get_mask_anchor > mask_prune_threshold) -- anchors below
        that threshold are dropped entirely rather than coded. Anchor geometry (x,y,z,t)
        is not yet entropy-coded (V-PCC integration is a separate follow-up); it is
        counted here as a fixed-length placeholder (anchor_round_digits bits per axis)."""
        bit2MB_scale = 8 * 1024 * 1024
        with torch.no_grad():
            keep = self.get_mask_anchor.squeeze(1) > mask_prune_threshold if self.use_entropy_coding \
                else torch.ones(self.get_anchor.shape[0], dtype=torch.bool, device="cuda")

            anchor = self.get_anchor[keep]
            N_total = self.get_anchor.shape[0]
            N = anchor.shape[0]
            feat = self._anchor_feat[keep]
            scaling = self.get_scaling[keep]
            offsets = self._offset[keep].reshape(N, -1)

            feat_context = self.calc_interp_feat(anchor)
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj)
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj)
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj)

            feat_q = STE_multistep.apply(feat, Q_feat, feat.mean())
            scaling_q = STE_multistep.apply(scaling, Q_scaling, scaling.mean())
            offsets_q = STE_multistep.apply(offsets, Q_offsets, offsets.mean())

            bit_feat = torch.sum(self.entropy_gaussian(feat_q, mean_feat, scale_feat, Q_feat)).item()
            bit_scaling = torch.sum(self.entropy_gaussian(scaling_q, mean_scaling, scale_scaling, Q_scaling)).item()
            bit_offsets = torch.sum(self.entropy_gaussian(offsets_q, mean_offsets, scale_offsets, Q_offsets)).item()
            bit_anchor_placeholder = N * 4 * anchor_round_digits

            return {
                'anchor_MB': bit_anchor_placeholder / bit2MB_scale,
                'feat_MB': bit_feat / bit2MB_scale,
                'scaling_MB': bit_scaling / bit2MB_scale,
                'offsets_MB': bit_offsets / bit2MB_scale,
                'anchor_num': N,
                'anchor_num_total': N_total,
            }

    def _anchor_to_int(self, anchor):
        # anchor: [N, 4] float (x,y,z,t), already on the create_from_pcd/anchor_growing grid.
        # Returns non-negative int64 grid coordinates plus the per-axis minimum that was
        # subtracted (needed to shift back to world space in _int_to_anchor).
        ix = torch.round(anchor[:, 0] / self.voxel_size).long()
        iy = torch.round(anchor[:, 1] / self.voxel_size).long()
        iz = torch.round(anchor[:, 2] / self.voxel_size).long()
        it = torch.round(anchor[:, 3] / self.t_grid_size).long()
        raw = torch.stack([ix, iy, iz, it], dim=1)
        mins = raw.min(dim=0).values
        return raw - mins, mins

    def _int_to_anchor(self, anchor_int, mins):
        raw = (anchor_int + mins).float()
        xyz = raw[:, :3] * self.voxel_size
        t = raw[:, 3:4] * self.t_grid_size
        return torch.cat([xyz, t], dim=1)

    def conduct_encoding(self, path, mask_prune_threshold=0.01):
        """Actually arithmetic-encode feat/scaling/offsets to disk (utils/arithmetic_coding.py,
        ported from HAC's submodules/arithmetic), using the per-anchor Gaussian mean/scale/Q
        predicted by mlp_grid. Anchors below mask_prune_threshold are dropped entirely rather
        than coded. Anchor position (x,y,z,t) is losslessly entropy-coded via an anisotropic
        4D hyperoctree (utils/octree_coding.py); rotation and opacity are not per-anchor data
        at all (see below). Returns a bit-size report dict."""
        os.makedirs(path, exist_ok=True)
        bit2MB_scale = 8 * 1024 * 1024
        byte2MB_scale = 1024 * 1024
        with torch.no_grad():
            keep = self.get_mask_anchor.squeeze(1) > mask_prune_threshold
            anchor = self.get_anchor[keep]
            feat = self._anchor_feat[keep]
            scaling = self.get_scaling[keep]
            offsets = self._offset[keep].reshape(anchor.shape[0], -1)

            # Entropy-code anchor geometry first, then reorder every other per-anchor
            # attribute into that same canonical order -- decode_positions always returns
            # its N leaves in this order, so aligning everything to it here means
            # conduct_decoding needs no separate un-permutation step at all.
            # encode_positions returns perm such that decoded[perm[i]] == anchor_int[i], i.e.
            # perm maps FROM this original order TO the decoded order -- so to reorder INTO
            # decoded order we need its inverse (argsort(perm)), not perm itself.
            anchor_int, anchor_mins = self._anchor_to_int(anchor)
            perm = encode_positions(anchor_int, os.path.join(path, 'anchor_geom.b'))
            inv_perm = torch.argsort(perm)
            anchor, feat, scaling, offsets = anchor[inv_perm], feat[inv_perm], scaling[inv_perm], offsets[inv_perm]
            N = anchor.shape[0]

            feat_context = self.calc_interp_feat(anchor)
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj).expand(N, self.feat_dim).contiguous()
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj).expand(N, 8).contiguous()
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj).expand(N, 4 * self.n_offsets).contiguous()

            feat_q = STE_multistep.apply(feat, Q_feat, feat.mean())
            scaling_q = STE_multistep.apply(scaling, Q_scaling, scaling.mean())
            offsets_q = STE_multistep.apply(offsets, Q_offsets, offsets.mean())

            bits_feat = encode_gaussian(feat_q.reshape(-1), mean_feat.reshape(-1), scale_feat.reshape(-1),
                                         Q_feat.reshape(-1), os.path.join(path, 'feat.b'))
            bits_scaling = encode_gaussian(scaling_q.reshape(-1), mean_scaling.reshape(-1), scale_scaling.reshape(-1),
                                            Q_scaling.reshape(-1), os.path.join(path, 'scaling.b'))
            bits_offsets = encode_gaussian(offsets_q.reshape(-1), mean_offsets.reshape(-1), scale_offsets.reshape(-1),
                                            Q_offsets.reshape(-1), os.path.join(path, 'offsets.b'))

            # rotation/opacity are NOT stored per-anchor at all: create_from_pcd and
            # anchor_growing always initialize every anchor to the same frozen
            # (requires_grad=False) identity rotation / constant opacity, and nothing in the
            # model ever trains or otherwise varies them, so every kept anchor's value is
            # identical -- one copy is stored and broadcast back on decode. If that assumption
            # is ever violated (e.g. a future change starts training them), the full
            # per-anchor tensor is stored instead so decode still reconstructs correctly.
            rotation = self._rotation[keep][inv_perm].detach()
            opacity = self._opacity[keep][inv_perm].detach()
            rotation_is_constant = bool(torch.all(rotation == rotation[0:1]))
            opacity_is_constant = bool(torch.all(opacity == opacity[0:1]))

            # The rate-distortion mask is CONTINUOUS (sigmoid(_mask_anchor) in (0,1)), not
            # binary -- keep_mask above only decides which anchors survive at all, but most
            # survivors still have a small-but-nonzero mask (e.g. 0.05) that meaningfully
            # dampens their rendered opacity. Losing that and treating every kept anchor as
            # "fully open" on decode reintroduces exactly the anchors training learned to
            # mostly (not entirely) suppress, at full strength -- a real, measured source of
            # a ~10dB PSNR drop between the uncompressed and decoded models. 1 byte/anchor is
            # cheap enough to store raw rather than build a dedicated entropy coder for it.
            mask_soft = self.get_mask_anchor[keep][inv_perm].detach()
            mask_q = torch.round(mask_soft * 255).clamp(0, 255).to(torch.uint8)

            torch.save({
                'anchor_mins': anchor_mins.cpu(),
                'mask_q': mask_q.cpu(),
                'rotation': (rotation[0:1] if rotation_is_constant else rotation).cpu(),
                'rotation_is_constant': rotation_is_constant,
                'opacity': (opacity[0:1] if opacity_is_constant else opacity).cpu(),
                'opacity_is_constant': opacity_is_constant,
            }, os.path.join(path, 'uncoded_attrs.pt'))

            return {
                'anchor_num': N,
                'anchor_geom_MB': os.path.getsize(os.path.join(path, 'anchor_geom.b')) / byte2MB_scale,
                'mask_MB': mask_q.numel() / byte2MB_scale,
                'feat_MB': bits_feat / bit2MB_scale,
                'scaling_MB': bits_scaling / bit2MB_scale,
                'offsets_MB': bits_offsets / bit2MB_scale,
            }

    def conduct_decoding(self, path):
        """Inverse of conduct_encoding. Requires mlp_grid/encoding_xyz to already be loaded
        (e.g. via load_mlp_checkpoints) with the SAME weights used at encode time, since the
        per-anchor mean/scale/Q are recomputed from anchor position, not stored in the bitstream."""
        with torch.no_grad():
            uncoded = torch.load(os.path.join(path, 'uncoded_attrs.pt'))
            anchor_mins = uncoded['anchor_mins'].cuda()
            anchor_int = decode_positions(os.path.join(path, 'anchor_geom.b'))
            anchor = self._int_to_anchor(anchor_int, anchor_mins)
            N = anchor.shape[0]

            feat_context = self.calc_interp_feat(anchor)
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj).expand(N, self.feat_dim).contiguous()
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj).expand(N, 8).contiguous()
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj).expand(N, 4 * self.n_offsets).contiguous()

            feat = decode_gaussian(mean_feat.reshape(-1), scale_feat.reshape(-1),
                                    Q_feat.reshape(-1), os.path.join(path, 'feat.b')).view(N, self.feat_dim)
            scaling_act = decode_gaussian(mean_scaling.reshape(-1), scale_scaling.reshape(-1),
                                          Q_scaling.reshape(-1), os.path.join(path, 'scaling.b')).view(N, 8)
            offsets = decode_gaussian(mean_offsets.reshape(-1), scale_offsets.reshape(-1),
                                       Q_offsets.reshape(-1), os.path.join(path, 'offsets.b')).view(N, self.n_offsets, 4)

            self._anchor = nn.Parameter(anchor)
            self._anchor_feat = nn.Parameter(feat)
            self._offset = nn.Parameter(offsets)
            # get_scaling = exp(_scaling), so invert with log to store back in _scaling's space
            self._scaling = nn.Parameter(self.scaling_inverse_activation(scaling_act.clamp(min=1e-9)))
            rotation = uncoded['rotation'].cuda()
            if uncoded.get('rotation_is_constant', True):
                rotation = rotation.expand(N, -1).contiguous()
            opacity = uncoded['opacity'].cuda()
            if uncoded.get('opacity_is_constant', True):
                opacity = opacity.expand(N, -1).contiguous()
            self._rotation = nn.Parameter(rotation)
            self._opacity = nn.Parameter(opacity)
            # Reconstruct the real (8-bit quantized) continuous mask value rather than
            # treating every surviving anchor as fully open -- see conduct_encoding's
            # comment on mask_q for why that matters.
            mask_soft = uncoded['mask_q'].cuda().float() / 255
            self._mask_anchor = nn.Parameter(inverse_sigmoid(mask_soft.clamp(1e-6, 1 - 1e-6)).view(N, 1))

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.use_entropy_coding:
            self.mlp_grid.eval()
            self.encoding_xyz.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.use_entropy_coding:
            self.mlp_grid.train()
            self.encoding_xyz.train()

    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_flow_mlp(self):
        return self.mlp_flow

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    
    def voxelize_sample(self, data=None, times=None, voxel_size=0.01, t_grid_size=0.0333):
        if times is None:
            np.random.shuffle(data)
            data = np.unique(np.round(data/voxel_size), axis=0) * voxel_size
            
            return data, None
        else:
            points = np.round(np.concatenate([data/voxel_size, times/t_grid_size], axis=1))
            np.random.shuffle(points)
            points = np.unique(points, axis=0)

            data = points[:, :3] * voxel_size
            times = points[:, 3] * t_grid_size

            return data, times


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float, init_voxel_scale : float = 1.0):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.ratio]
        if pcd.times is not None:
            times = pcd.times[::self.ratio]
        else:
            times = None

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        
        points, times = self.voxelize_sample(points, times, voxel_size=self.voxel_size * init_voxel_scale, t_grid_size=self.t_grid_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 4)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 8)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        anchors = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        anchors[:, :3] = fused_point_cloud
        if times is not None:
            anchors[:, 3] = torch.tensor(np.asarray(times)).float().cuda()
        else:
            anchors[:, 3] = torch.rand_like(anchors[:, 3]).float().cuda() * self.max_init_t


        self._anchor = nn.Parameter(anchors.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

        if self.use_entropy_coding:
            mask_anchor = torch.full((fused_point_cloud.shape[0], 1), MASK_INIT_LOGIT, device="cuda")
            self._mask_anchor = nn.Parameter(mask_anchor.requires_grad_(True))


    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_time_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_time_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        
        
        if self.use_feat_bank:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                
                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.mlp_flow.parameters(), 'lr': training_args.mlp_flow_lr_init, "name": "mlp_flow"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        elif self.appearance_dim > 0:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
                {'params': self.mlp_flow.parameters(), 'lr': training_args.mlp_flow_lr_init, "name": "mlp_flow"},
                {'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"},
            ]
        else:
            l = [
                {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
                {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
                {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},

                {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
                {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
                {'params': self.mlp_flow.parameters(), 'lr': training_args.mlp_flow_lr_init, "name": "mlp_flow"},
                {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
            ]

        if self.use_entropy_coding:
            l.append({'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"})
            l.append({'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"})
            l.append({'params': [self._mask_anchor], 'lr': training_args.mask_lr, "name": "mask_anchor"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        
        self.mlp_flow_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_flow_lr_init,
                                                    lr_final=training_args.mlp_flow_lr_final,
                                                    lr_delay_mult=training_args.mlp_flow_lr_delay_mult,
                                                    max_steps=training_args.mlp_flow_lr_max_steps)
        
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)
        if self.use_entropy_coding:
            self.mlp_grid_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_grid_lr_init,
                                                        lr_final=training_args.mlp_grid_lr_final,
                                                        lr_delay_mult=training_args.mlp_grid_lr_delay_mult,
                                                        max_steps=training_args.mlp_grid_lr_max_steps)
            self.encoding_xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.encoding_xyz_lr_init,
                                                        lr_final=training_args.encoding_xyz_lr_final,
                                                        lr_delay_mult=training_args.encoding_xyz_lr_delay_mult,
                                                        max_steps=training_args.encoding_xyz_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_flow":
                lr = self.mlp_flow_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_entropy_coding and param_group["name"] == "mlp_grid":
                lr = self.mlp_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_entropy_coding and param_group["name"] == "encoding_xyz":
                lr = self.encoding_xyz_scheduler_args(iteration)
                param_group['lr'] = lr
            
            
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 't']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.use_entropy_coding:
            l.append('mask_anchor')
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        to_cat = [anchor, offset, anchor_feat, opacities, scale, rotation]
        if self.use_entropy_coding:
            to_cat.append(self._mask_anchor.detach().cpu().numpy())
        attributes = np.concatenate(to_cat, axis=1)
        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def construct_list_of_attributes_compressed(self):
        l = ['x', 'y', 'z', 't']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        return l

    def save_ply_compressed(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()

        # prunning
        neural_opacity = self.get_opacity_mlp(self._anchor_feat)
        mask = ((neural_opacity > 0.00).sum(dim=1) > 0).detach().cpu().numpy()
        anchor = anchor[mask]
        anchor_feat = anchor_feat[mask]
        scale = scale[mask]
        offset = self._offset.detach().cpu()[mask].transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes_compressed()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, offset, anchor_feat, scale), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"]),
                        np.asarray(plydata.elements[0]["t"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 4, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_entropy_coding:
            if "mask_anchor" in [p.name for p in plydata.elements[0].properties]:
                mask_anchor = np.asarray(plydata.elements[0]["mask_anchor"])[..., np.newaxis].astype(np.float32)
            else:
                mask_anchor = np.full((anchor.shape[0], 1), MASK_INIT_LOGIT, dtype=np.float32)
            self._mask_anchor = nn.Parameter(torch.tensor(mask_anchor, dtype=torch.float, device="cuda").requires_grad_(True))

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'encoding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, opacity_t, sigma, lambda_temporal_sigma, timestamp, opt, activate_DA):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        if opt.opacity_accum_method == "median":
            self.opacity_accum[anchor_visible_mask] += temp_opacity.median(dim=1, keepdim=True)[0]
        elif opt.opacity_accum_method == "mean":
            self.opacity_accum[anchor_visible_mask] += temp_opacity.mean(dim=1, keepdim=True)[0]
        else:
            self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        temp_opacity_t = torch.zeros_like(temp_mask, dtype=torch.float)
        temp_opacity_t[temp_mask] = opacity_t.squeeze(-1)

        temp_sigma = torch.zeros_like(temp_mask, dtype=torch.float)
        temp_sigma[temp_mask] = sigma.squeeze(-1)

        if self.disable_denom_weight:
            weight = 1
        elif opt.const_denom_weight > 0:
            weight = opt.const_denom_weight
        elif self.sigma_denom_weight and activate_DA:
            weight = temp_opacity_t[combined_mask].unsqueeze(-1) * (temp_sigma[combined_mask].unsqueeze(-1) * opt.lambda_sigma_multiply)**lambda_temporal_sigma + opt.weight_base
        elif activate_DA:
            weight = temp_opacity_t[combined_mask].unsqueeze(-1) + opt.weight_base
        else:
            weight = 1

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm * weight
        self.offset_time_accum[combined_mask] += timestamp * grad_norm
        self.offset_denom[combined_mask] += 1 * weight
        self.offset_time_denom[combined_mask] += grad_norm

        

        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name'] or \
                'encoding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,4:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,4:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,4:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,4:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            
            
        return optimizable_tensors

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        if self.use_entropy_coding:
            self._mask_anchor = optimizable_tensors["mask_anchor"]

    
    def anchor_growing(self, opt, grads, threshold, offset_mask, timestamp):
        ## 
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # print(f"grad th {i}: {candidate_mask.sum().item()} / {candidate_mask.shape[0]}")

            if not opt.disable_partial_densify:
                # random pick
                rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
                rand_mask = rand_mask.cuda()
                candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            # # replace t with timestamp
            # selected_t = timestamp[candidate_mask].unsqueeze(-1)

            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:4].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            t_size = self.t_grid_size
            # t_size = self.t_grid_size*size_factor
            
            # grid_coords = torch.round(self.get_anchor / cur_size).int()
            grid_coords = torch.cat([
                torch.round(self.get_anchor[:,:3] / cur_size).int(),
                torch.round(self.get_anchor[:,3:4] / t_size).int()
            ], dim=1)

            selected_xyz = all_xyz.view([-1, 4])[candidate_mask]

            if opt.densify_t == 'randn':
                selected_t = torch.randn_like(selected_xyz[:,3:4]) + selected_xyz[:,3:4]
            elif opt.densify_t == 'rand':
                selected_t = torch.rand_like(selected_xyz[:,3:4])
            else:
                selected_t = selected_xyz[:,3:4]

            # selected_grid_coords = torch.round(selected_xyz / cur_size).int()
            selected_grid_coords = torch.cat([
                torch.round(selected_xyz[:,:3] / cur_size).int(),
                torch.round(selected_t / t_size).int()
            ], dim=1)

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            # candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size
            candidate_anchor = torch.cat([
                selected_grid_coords_unique[remove_duplicates][:,:3]*cur_size,
                selected_grid_coords_unique[remove_duplicates][:,3:4]*t_size
            ], dim=1)
            
            # print(f"remove dup: {candidate_anchor.shape[0]} / {selected_grid_coords.shape[0]}")

            if candidate_anchor.shape[0] > 0:
                # new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.cat([
                    torch.ones_like(candidate_anchor[:,:3]).repeat([1,2]).float().cuda()*cur_size,
                    torch.ones_like(candidate_anchor[:,3:4]).repeat([1,2]).float().cuda()*t_size
                ], dim=1)
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                if self.use_entropy_coding:
                    d["mask_anchor"] = torch.full((candidate_anchor.shape[0], 1), MASK_INIT_LOGIT, device="cuda")


                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                if self.use_entropy_coding:
                    self._mask_anchor = optimizable_tensors["mask_anchor"]
                


    def adjust_anchor(self, opt, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        # print(f"denom th: {offset_mask.sum()} / {offset_mask.shape[0]}")

        timestamp = (self.offset_time_accum / self.offset_time_denom).squeeze()
        
        self.anchor_growing(opt, grads_norm, grad_threshold, offset_mask, timestamp)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_time_denom[offset_mask] = 0
        padding_offset_time_denom = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_time_denom.shape[0], 1],
                                             dtype=torch.int32, 
                                             device=self.offset_time_denom.device)
        self.offset_time_denom = torch.cat([self.offset_time_denom, padding_offset_time_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        padding_offset_time_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_time_accum.shape[0], 1],
                                             dtype=torch.int32, 
                                             device=self.offset_time_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        self.offset_time_accum = torch.cat([self.offset_time_accum, padding_offset_time_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]
        if self.use_entropy_coding:
            # rate-distortion mask: anchors the model has learned are not worth their coding cost
            rate_prune_mask = (self.get_mask_anchor.squeeze(dim=1) < opt.mask_prune_threshold)
            prune_mask = torch.logical_or(prune_mask, rate_prune_mask)
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_time_denom = self.offset_time_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_time_denom = offset_time_denom.view([-1, 1])
        del self.offset_time_denom
        self.offset_time_denom = offset_time_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        offset_time_accum = self.offset_time_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_time_accum = offset_time_accum.view([-1, 1])
        del self.offset_time_accum
        self.offset_time_accum = offset_time_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            self.mlp_flow.eval()
            flow_mlp = torch.jit.trace(self.mlp_flow, (torch.rand(1, self.feat_dim).cuda()))
            flow_mlp.save(os.path.join(path, 'flow_mlp.pt'))
            self.mlp_flow.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

            if self.use_entropy_coding:
                self.mlp_grid.eval()
                hash_feat_dim = self.encoding_xyz[0].output_dim * len(HASH_TRIPLETS)
                grid_mlp = torch.jit.trace(self.mlp_grid, (torch.rand(1, hash_feat_dim).cuda()))
                grid_mlp.save(os.path.join(path, 'grid_mlp.pt'))
                self.mlp_grid.train()

                self.save_hash_grid(os.path.join(path, 'encoding_xyz.bin'))
                torch.save({'x_bound_min': self.x_bound_min, 'x_bound_max': self.x_bound_max},
                           os.path.join(path, 'anchor_bound.pt'))

        elif mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'flow_mlp': self.mlp_flow.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'flow_mlp': self.mlp_flow.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'flow_mlp': self.mlp_flow.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            self.mlp_flow = torch.jit.load(os.path.join(path, 'flow_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
            if self.use_entropy_coding:
                self.mlp_grid = torch.jit.load(os.path.join(path, 'grid_mlp.pt')).cuda()
                self.load_hash_grid(os.path.join(path, 'encoding_xyz.bin'))
                bounds = torch.load(os.path.join(path, 'anchor_bound.pt'))
                self.x_bound_min = bounds['x_bound_min']
                self.x_bound_max = bounds['x_bound_max']
                self.bounds_ready = True
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            self.mlp_flow.load_state_dict(checkpoint['flow_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError
