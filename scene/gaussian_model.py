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
import math
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
from utils.entropy_models import Entropy_gaussian, Entropy_gaussian_mix_prob_2, get_binary_vxl_size
from utils.arithmetic_coding import encode_gaussian_chunk as encode_gaussian, decode_gaussian_chunk as decode_gaussian
from utils.arithmetic_coding import encode_gaussian_mixed_chunk, decode_gaussian_mixed_chunk
from utils.arithmetic_coding import encode_gaussian_mixed, decode_gaussian_mixed
from utils.arithmetic_coding import fit_gmm_1d_2comp
from utils.arithmetic_coding import encode_binary, decode_binary
from utils.octree_coding import encode_positions, decode_positions

# 4D anchor axes are (x, y, z, t). Default design: decompose the 4D anchor position into
# every 3-axis combination and run one 3D GridEncoder per combination, then concatenate
# their outputs as the context feature (K-Planes/HexPlane-style factorization; the existing
# 3D hash-grid CUDA kernel ported from HAC only supported num_dim<=3 when this was written).
HASH_TRIPLETS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]  # xyz, xyt, xzt, yzt

# Ablation: use_joint_4d_hash=True replaces the 4 factored 3D encoders above with a single
# GridEncoder(num_dim=4) hashing (x,y,z,t) jointly -- a true 4D hash grid rather than a
# triplet decomposition. num_dim=4 support was added to submodules/gridencoder's CUDA kernel
# (previously commented out, num_dim<=3 only) specifically for this. Same hash_n_features /
# hash_log2_size per table as the triplet mode, so this compares "1 joint 4D table" against
# "4 factored 3D tables" at equal per-table capacity, not equal total capacity.

# Ablation: use_hybrid_4d_hash=True combines a joint 4D table (xyzt) with a dedicated xyz-only
# 3D table -- the 4D table supplies spatiotemporal coupling, the xyz table supplies spatial
# detail the 4D table can't afford at equal hash_log2_size (see HYBRID_4D_RESOLUTIONS below).
#
# A GridEncoder's per-level hash table saturates its log2_hashmap_size cap once
# resolution**num_dim exceeds 2**log2_hashmap_size -- resolution ~27 for num_dim=4 vs ~81 for
# num_dim=3 at the default log2_hashmap_size=19. Past that point every extra level of the
# default 12-level schedule (16..736) is oversubscribed by an ever-growing factor (2x at R=32,
# up to ~560,000x at R=736) -- almost pure hash collision with negligible new information, while
# still costing mlp_grid an extra n_features-wide input slice. The joint 4D table therefore uses
# a shortened schedule that stops before that regime (through R=128, ~512x oversubscribed) --
# the xyz-only table keeps the original 12-level schedule since num_dim=3 doesn't saturate until
# much higher up.
HYBRID_4D_RESOLUTIONS = (16, 23, 32, 46, 64, 92, 128)

# Initial logit for the learnable per-anchor rate-distortion mask (sigmoid(2.0) ~= 0.88):
# anchors start "mostly open" so useful ones aren't killed before they prove useful.
MASK_INIT_LOGIT = 2.0


class ChannelCtxFeat(nn.Module):
    """Intra-anchor context model for anchor_feat: splits feat_dim into n_groups equal
    channel groups and predicts each group's (mean_adj, scale_adj, prob_adj) from the
    already-known groups of the SAME anchor (channels before it) plus the hash-grid's own
    per-channel (mean, scale, prob) for the whole attribute. Ported from HAC++'s
    scene/gaussian_model.py Channel_CTX_fea, generalized from its fixed 5x10 split (for
    HAC++'s feat_dim=50) to n_groups equal groups so it works with this fork's feat_dim=32
    default (4 groups of 8).

    forward() with to_dec=-1 computes all n_groups from the TRUE (already quantized)
    feat_q -- used during training, where the whole rate loss must be differentiable and
    every group's ground truth is already available as context (teacher forcing). With
    to_dec=cc it returns only group cc's (mean_adj, scale_adj, prob_adj), computed from
    groups [0, cc) of feat_q only -- used during real encode/decode, which must reproduce
    this exact ordering since the decoder doesn't have groups >= cc yet (the caller passes
    a feat_q whose groups >= cc are unused placeholders).

    Tried the same scheme for `offsets` (one group per child-Gaussian's (x,y,z,t) slot,
    conditioning later slots on earlier ones within the same anchor) and reverted it:
    offsets are ~87-90% inactive per anchor (see conduct_encoding's active-slot
    filtering), so most groups carry near-zero context to pass on, and the extra
    per-group MLP + bitstream-file overhead measured a net LARGER compressed size on
    average across a 5-lambda sweep (worst case +17% at the sparsest anchor counts) --
    not worth it for this attribute, unlike feat which is dense and always populated."""

    def __init__(self, feat_dim, n_groups=4):
        super().__init__()
        assert feat_dim % n_groups == 0, \
            f"ChannelCtxFeat needs feat_dim ({feat_dim}) divisible by n_groups ({n_groups})"
        self.feat_dim = feat_dim
        self.n_groups = n_groups
        self.group_size = feat_dim // n_groups
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * feat_dim + self.group_size * i, 2 * self.group_size),
                nn.LeakyReLU(inplace=True),
                nn.Linear(2 * self.group_size, 3 * self.group_size),
            )
            for i in range(n_groups)
        ])

    def forward(self, feat_q, mean_scale_prob, to_dec=-1):
        groups = torch.split(feat_q, self.group_size, dim=-1)
        prior_groups = []
        means, scales, probs = [], [], []
        for i, mlp in enumerate(self.mlps):
            inp = torch.cat(prior_groups + [mean_scale_prob], dim=-1)
            mean_i, scale_i, prob_i = torch.chunk(mlp(inp), chunks=3, dim=-1)
            # Same NaN/Inf firewall as forward_grid (see its comment) -- this MLP's
            # output feeds entropy_gaussian_mix's second component independently of
            # forward_grid's, so it needs its own guard against the same failure mode
            # (confirmed in practice: a late-training, high-lambda run crashed here with
            # Normal's scale constraint violated, from an unsanitized scale_i).
            mean_i = torch.nan_to_num(mean_i, nan=0.0, posinf=1e4, neginf=-1e4)
            scale_i = torch.nan_to_num(scale_i, nan=0.0, posinf=1e4, neginf=-1e4)
            prob_i = torch.nan_to_num(prob_i, nan=0.0, posinf=1e4, neginf=-1e4)
            if to_dec == i:
                return mean_i, scale_i, prob_i
            means.append(mean_i)
            scales.append(scale_i)
            probs.append(prob_i)
            prior_groups.append(groups[i])
        if to_dec != -1:
            raise ValueError(f"to_dec={to_dec} out of range for n_groups={self.n_groups}")
        return torch.cat(means, dim=-1), torch.cat(scales, dim=-1), torch.cat(probs, dim=-1)


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
                 use_intra_anchor_context : bool = False,
                 use_joint_4d_hash : bool = False,
                 use_hybrid_4d_hash : bool = False,
                 use_full_hybrid_4d_hash : bool = False,
                 ablation_no_hash_context : bool = False,
                 ablation_fixed_quant : bool = False,
                 ablation_no_time_agg_mask : bool = False,
                 motion_model : str = 'linear',
                 motion_basis_dim : int = 16,
                 ):
        assert sum([use_joint_4d_hash, use_hybrid_4d_hash, use_full_hybrid_4d_hash, ablation_no_hash_context]) <= 1, \
            "use_joint_4d_hash, use_hybrid_4d_hash, use_full_hybrid_4d_hash and ablation_no_hash_context are mutually exclusive hash-grid configurations"

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
        self.use_intra_anchor_context = use_intra_anchor_context
        self.use_joint_4d_hash = use_joint_4d_hash
        self.use_hybrid_4d_hash = use_hybrid_4d_hash
        self.use_full_hybrid_4d_hash = use_full_hybrid_4d_hash
        self.ablation_no_hash_context = ablation_no_hash_context
        self.ablation_fixed_quant = ablation_fixed_quant
        self.ablation_no_time_agg_mask = ablation_no_time_agg_mask
        self.motion_model = motion_model
        self.motion_basis_dim = motion_basis_dim

        self._anchor = torch.empty(0)
        # Mip-Splatting-style 3D smoothing filter: per-anchor variance floor so a Gaussian
        # can't render thinner than the finest world-space pixel footprint any TRAINING
        # camera could actually resolve for it (see compute_3D_filter). Not a learned
        # parameter and not entropy-coded -- deterministically recomputed from anchor
        # position + the (fixed) training camera set, same category as rotation/flow.
        self.filter_3D = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        
        self.opacity_accum = torch.empty(0)
        self.mask_val_accum = torch.empty(0)
        # Ablation companion to mask_val_accum: overwritten (not accumulated) on every
        # visit, so it always holds only the CURRENT iteration's instantaneous mask
        # value rather than a running average -- see ablation_no_time_agg_mask's use at
        # the prune decision.
        self.mask_val_last = torch.empty(0)

        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        # Offset-level (sub-anchor) rate-distortion mask, gating opacity and bit_offsets
        # per individual offset slot rather than per whole anchor like _mask_anchor.
        # Only allocated once anchor growing has finished (iteration == update_until in
        # train.py), since its shape [N, n_offsets] must track the final, frozen anchor
        # count -- unlike _mask_anchor it does NOT participate in the grow/prune lifecycle.
        self._mask_offset = torch.empty(0)
        self.use_offset_mask = False

        # Ex4DGS-inspired per-anchor motion gate: multiplies the anchor's raw flow output
        # before motion displacement is computed, letting a static anchor learn to fully
        # suppress its own motion. Allocated at update_until like _mask_offset (shape [N,
        # 1] must track the final, frozen anchor count), so it does NOT participate in
        # the grow/prune lifecycle.
        self._motion_gate = torch.empty(0)
        self.use_motion_gate = False
        
        self.offset_gradient_accum = torch.empty(0)
        # Parallel accumulator for the photometric-error densification signal (see
        # anchor_growing): per-visit rendered-vs-GT L1 error sampled at this offset's
        # projected screen location, averaged the same way as offset_gradient_accum.
        self.photo_error_accum = torch.empty(0)
        # Parallel accumulator for the flow-magnitude densification signal (see
        # anchor_growing's flow_mask): per-visit motion speed (flow[:, :3].norm()) at this
        # offset, averaged the same way as offset_gradient_accum/photo_error_accum. Lets
        # anchor_growing grow offsets in fast-moving regions even when their screen-space
        # gradient doesn't cross densify_grad_threshold -- the motion-blur problem this
        # targets showed up specifically as under-resolved detail in fast-moving regions,
        # not slow/static ones, so gradient alone (which responds to appearance error, not
        # motion per se) can miss it.
        self.flow_speed_accum = torch.empty(0)
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

        # Motion model ablation (see arguments/__init__.py's motion_model docstring for the
        # full rationale of each option). In every case mlp_flow stays a feat-only ->
        # per-offset-coefficients network computed once per anchor (same as the original
        # velocity-only model), traceable/saveable exactly like before; only its output
        # width changes. 'basis' and 'taylor_residual' additionally need a dt-dependent
        # shared time-basis Phi(dt), computed separately at render time (see
        # generate_neural_gaussians / render()) since dt is only known after per-camera,
        # per-timestamp masking -- mlp_time_basis holds that piece. 'fourier' needs no
        # learned time module: Phi(dt) is a fixed sin/cos harmonic basis computed in the
        # renderer directly from motion_fourier_freqs.
        B = self.motion_basis_dim
        if self.motion_model == 'linear':
            self.mlp_flow = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*self.n_offsets),
                nn.Tanh()
            ).cuda()
        elif self.motion_model == 'basis':
            self.mlp_flow = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*B*self.n_offsets),
                nn.Tanh()
            ).cuda()
            self.mlp_time_basis = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(True),
                nn.Linear(64, B),
                nn.Tanh()
            ).cuda()
        elif self.motion_model == 'fourier':
            self.mlp_flow = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, 3*B*self.n_offsets),
                nn.Tanh()
            ).cuda()
            n_harmonics = B // 2
            # Scene clips span ~10s (time_duration default [0, 10]); harmonics span that
            # fundamental period so the fixed basis covers the observed dt range.
            self.motion_fourier_freqs = (2 * torch.pi * torch.arange(1, n_harmonics + 1, dtype=torch.float32) / 10.0).cuda()
        elif self.motion_model == 'taylor_residual':
            B_res = B // 2
            self.mlp_flow = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(True),
                nn.Linear(feat_dim, (9 + 3*B_res)*self.n_offsets),
                nn.Tanh()
            ).cuda()
            self.mlp_time_basis = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(True),
                nn.Linear(64, B_res),
                nn.Tanh()
            ).cuda()
        else:
            raise ValueError(f"unknown motion_model {self.motion_model!r}")

        if self.use_entropy_coding:
            self.x_bound_min = torch.zeros((1, 4), device="cuda")
            self.x_bound_max = torch.ones((1, 4), device="cuda")
            self.bounds_ready = False

            if self.ablation_no_hash_context:
                # Ablation: same single 4D joint table as use_joint_4d_hash, but
                # calc_interp_feat below queries it at a FIXED constant coordinate for
                # every anchor instead of the anchor's real (x,y,z,t) -- so the looked-up
                # feature (and everything mlp_grid predicts from it) no longer depends on
                # anchor position, i.e. no inter-anchor spatiotemporal context, while
                # still being a learned (not hardcoded-zero) value. Reusing the real
                # GridEncoder this way (rather than a bare nn.Parameter) means every
                # existing save/load/rate-loss code path for encoding_xyz works
                # unchanged; only calc_interp_feat's query point differs.
                self.encoding_xyz = nn.ModuleList([
                    GridEncoder(num_dim=4, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1)
                ]).cuda()
            elif self.use_full_hybrid_4d_hash:
                # 4D joint table (shortened schedule) + all 4 original 3D triplet tables --
                # the "everything" ablation point: does stacking the factored triplets ON TOP
                # of the joint table help, or is the joint table's own spatiotemporal coupling
                # already redundant with them (as use_hybrid_4d_hash's single xyz table
                # suggested)?
                self.encoding_xyz = nn.ModuleList([
                    GridEncoder(num_dim=4, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                resolutions_list=HYBRID_4D_RESOLUTIONS,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1),
                ] + [
                    GridEncoder(num_dim=3, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1)
                    for _ in HASH_TRIPLETS
                ]).cuda()
            elif self.use_hybrid_4d_hash:
                self.encoding_xyz = nn.ModuleList([
                    GridEncoder(num_dim=4, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                resolutions_list=HYBRID_4D_RESOLUTIONS,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1),
                    GridEncoder(num_dim=3, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1),
                ]).cuda()
            elif self.use_joint_4d_hash:
                self.encoding_xyz = nn.ModuleList([
                    GridEncoder(num_dim=4, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1)
                ]).cuda()
            else:
                self.encoding_xyz = nn.ModuleList([
                    GridEncoder(num_dim=3, n_features=hash_n_features, log2_hashmap_size=hash_log2_size,
                                ste_binary=True, ste_multistep=False, add_noise=False, Q=1)
                    for _ in HASH_TRIPLETS
                ]).cuda()

            # NOTE: sum(), not output_dim * len(...) -- the hybrid mode's two tables have
            # different level counts (HYBRID_4D_RESOLUTIONS has 7 levels vs the xyz table's
            # 12), so their output_dim differs and can't be assumed uniform across tables.
            hash_feat_dim = sum(encoder.output_dim for encoder in self.encoding_xyz)
            # mlp_grid predicts a per-anchor Gaussian (mean, scale) for each entropy-coded
            # attribute: anchor_feat (feat_dim), scaling (8-dim), offsets (4*n_offsets-dim),
            # plus one adaptive-quantization-step adjustment logit per attribute group.
            # anchor position (x,y,z,t) itself and mlp_flow / temporal sigma are NOT coded here.
            # With intra-anchor context enabled, anchor_feat additionally gets a per-channel
            # prob_feat output: the mixing weight (pre-softmax logit) between this hash-grid
            # (inter-anchor) Gaussian and the intra-anchor MLP's adjustment -- see
            # ChannelCtxFeat and Entropy_gaussian_mix_prob_2.
            grid_out_dim = 2 * feat_dim + 2 * 7 + 2 * (4 * self.n_offsets) + 3
            if self.use_intra_anchor_context:
                grid_out_dim += feat_dim
            self.mlp_grid = nn.Sequential(
                nn.Linear(hash_feat_dim, feat_dim * 2),
                nn.ReLU(True),
                nn.Linear(feat_dim * 2, grid_out_dim),
            ).cuda()

            self.entropy_gaussian = Entropy_gaussian(Q=1).cuda()

            if self.use_intra_anchor_context:
                self.mlp_channel_ctx = ChannelCtxFeat(feat_dim, n_groups=4).cuda()
                self.entropy_gaussian_mix = Entropy_gaussian_mix_prob_2(Q=1).cuda()

    def calc_interp_feat(self, x):
        # x: [N, 4] anchor positions (x, y, z, t)
        assert len(x.shape) == 2 and x.shape[1] == 4
        x = (x - self.x_bound_min) / (self.x_bound_max - self.x_bound_min)  # -> [0, 1]
        if self.ablation_no_hash_context:
            # Query the single table at the same fixed point (grid center) for every
            # anchor, regardless of x -- see this flag's comment in __init__.
            x_const = torch.full_like(x, 0.5)
            feats = [self.encoding_xyz[0](x_const)]
        elif self.use_full_hybrid_4d_hash:
            feats = [self.encoding_xyz[0](x)] + \
                [encoder(x[:, triplet]) for encoder, triplet in zip(self.encoding_xyz[1:], HASH_TRIPLETS)]
        elif self.use_hybrid_4d_hash:
            feats = [self.encoding_xyz[0](x), self.encoding_xyz[1](x[:, :3])]
        elif self.use_joint_4d_hash:
            feats = [self.encoding_xyz[0](x)]
        else:
            feats = [encoder(x[:, triplet]) for encoder, triplet in zip(self.encoding_xyz, HASH_TRIPLETS)]
        return torch.cat(feats, dim=-1)

    def forward_grid(self, feat_context):
        y = self.mlp_grid(feat_context)
        # Firewall against NaN/Inf propagating into the entropy-coding mean/scale/Q
        # predictions (seen in practice at low lambda, where weaker rate regularization
        # lets some intermediate value drift until a log/division blows up) -- a no-op
        # on already-finite values, so this changes nothing for any already-validated run.
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        sizes = [self.feat_dim, self.feat_dim, 7, 7, 4 * self.n_offsets, 4 * self.n_offsets, 1, 1, 1]
        if self.use_intra_anchor_context:
            sizes.append(self.feat_dim)
        return torch.split(y, sizes, dim=-1)

    def adaptive_Q(self, Q_base, Q_adj):
        # Ablation: skip the adaptive step entirely and always quantize at the fixed
        # base width. Q_adj is still predicted by mlp_grid upstream (unused dead weight
        # under this flag) rather than conditionally removed from grid_out_dim -- keeping
        # the network architecture identical to the non-ablated run isolates this
        # ablation to exactly the quantization step, not a confounded smaller model.
        if self.ablation_fixed_quant:
            return torch.as_tensor(Q_base, dtype=Q_adj.dtype, device=Q_adj.device).expand_as(Q_adj)
        # HAC-style adaptive quantization step: Q in [0.1*Q_base, 2*Q_base].
        # The floor matters more than it looks: encode_gaussian's arithmetic coder builds
        # a CDF table sized by round(x/Q)'s range, shared across every element in one
        # call. A Q allowed to collapse toward 0 for even a single element can blow that
        # range up by orders of magnitude and OOM the whole batch, so Q is kept within a
        # bounded ratio of Q_base rather than letting tanh saturate all the way to -1.
        #
        # The ceiling (tanh saturating at +1) looks like an unintentional artifact rather
        # than a deliberate choice -- and on a real checkpoint, Q does sit pinned at
        # exactly this ceiling for ~100% of elements across feat/scaling/offsets alike.
        # But removing it is NOT a free win: both tried fixes (raising Q_base 4x with this
        # same tanh formula, and switching to a non-saturating softplus formula at the
        # ORIGINAL Q_base) collapsed to the same catastrophic ~16.7dB PSNR (vs ~28dB
        # normal) on an otherwise-identical smoke test, with feat's coded size shrinking
        # to near-nothing (~0.03-0.04MB vs ~0.65-0.9MB) -- i.e. feat gets quantized away
        # almost entirely. That rules out "Q_base was just too large" as the explanation:
        # the failure reproduced at the original Q_base too, under a different
        # non-saturating formula. The real cause looks like a one-sided imbalance between
        # the rate loss's gradient (pushes Q up, direct and strong) and the distortion
        # loss's gradient on Q (indirect, via STE noise, and apparently too weak to push
        # back at this scale) -- meaning tanh's vanishing gradient at saturation was
        # doing useful, unintentional work as a brake, not merely limiting an otherwise-
        # fine equilibrium. Left as tanh/[0.1x, 2x] rather than revisiting this without a
        # fix for that underlying rate/distortion gradient imbalance (e.g. a separate,
        # slower learning rate for Q_adj, or a warm-up schedule for the allowed range).
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

    @property
    def get_mask_offset(self):
        # Per-offset (sub-anchor) rate-distortion mask, shape [N, n_offsets]. Gates
        # opacity and bit_offsets at finer granularity than get_mask_anchor, so an anchor
        # that is worth keeping can still shed individual unused offset slots. Only valid
        # once self._mask_offset has been allocated (see use_offset_mask in train.py).
        return torch.sigmoid(self._mask_offset)

    @property
    def get_motion_gate(self):
        # Per-anchor motion gate, shape [N, 1]. Only valid once self._motion_gate has
        # been allocated (see use_motion_gate in train.py).
        return torch.sigmoid(self._motion_gate)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        # Mip-Splatting-style 3D smoothing filter: a per-anchor variance floor set from
        # the finest world-space pixel footprint any TRAINING camera could actually resolve
        # at that anchor's location. Prevents a Gaussian from shrinking thinner than every
        # camera's sampling rate can support (the source of aliasing/erosion under
        # out-of-distribution test views/zooms). Deterministic given anchor position + the
        # fixed training camera set -- recompute whenever the anchor set changes (same
        # cadence as adjust_anchor's growing/pruning) to stay shape-consistent.
        xyz = self.get_anchor[:, :3]
        min_sampling = torch.full((xyz.shape[0],), float('inf'), device="cuda")
        for cam in cameras:
            dist = (xyz - cam.camera_center.to(xyz.device)).norm(dim=-1).clamp(min=1e-6)
            focal_x = cam.image_width / (2 * math.tan(cam.FoVx * 0.5))
            focal_y = cam.image_height / (2 * math.tan(cam.FoVy * 0.5))
            focal = min(focal_x, focal_y)
            sampling = dist / focal
            min_sampling = torch.min(min_sampling, sampling)
        # variance-equivalent floor, in world units, mirroring the de facto
        # "+0.3 pixel dilation" convention used across 3DGS forks.
        self.filter_3D = (min_sampling * 0.2) ** 2

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
            if self.use_intra_anchor_context:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj, prob_feat = self.forward_grid(feat_context)
            else:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj)
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj)
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj)

            feat_q = STE_multistep.apply(feat, Q_feat, feat.mean())
            scaling_q = STE_multistep.apply(scaling, Q_scaling, scaling.mean())
            offsets_q = STE_multistep.apply(offsets, Q_offsets, offsets.mean())

            if self.use_intra_anchor_context:
                mean_adj, scale_adj, prob_adj = self.mlp_channel_ctx(
                    feat_q, torch.cat([mean_feat, scale_feat, prob_feat], dim=-1))
                probs = torch.softmax(torch.stack([prob_feat, prob_adj], dim=-1), dim=-1)
                bit_feat = torch.sum(self.entropy_gaussian_mix(
                    feat_q, mean_feat, mean_adj, scale_feat, scale_adj,
                    probs[..., 0], probs[..., 1], Q_feat)).item()
            else:
                bit_feat = torch.sum(self.entropy_gaussian(feat_q, mean_feat, scale_feat, Q_feat)).item()
            bit_scaling = torch.sum(self.entropy_gaussian(scaling_q, mean_scaling, scale_scaling, Q_scaling)).item()
            # NOTE: offsets rate here is still the single-Gaussian estimate (not the
            # channel-context mixture actually used by conduct_encoding) -- this function
            # is a logging-only approximation, not the real bit_per_param training loss or
            # the real bitstream, so the mismatch doesn't affect correctness anywhere.
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
        # Clear any bitstream files from a PREVIOUS call on this same path first. The
        # chunked coders (encode_gaussian_chunk etc.) name files offsets_0.b, offsets_1.b,
        # ... up to however many chunks THIS call needs; re-encoding in place (e.g. after
        # a code change that shrinks the data, as happened when offsets filtering was
        # added) writes fewer/smaller chunks than before, but never deletes the old
        # chunks that are no longer needed -- they silently survive as stale leftovers,
        # inflating the directory's real size far beyond what this call's own byte counts
        # (bits_feat/bits_scaling/bits_offsets) report. Caught in practice: re-running
        # this on an already-encoded checkpoint left ~2200 stale offsets_*.b chunk files
        # from before offsets filtering shrank the chunk count to ~34.
        for fname in os.listdir(path):
            if fname == 'uncoded_attrs.pt' or fname == 'anchor_geom.b' or \
               fname.startswith(('feat', 'scaling', 'offsets', 'mask')):
                os.remove(os.path.join(path, fname))
        bit2MB_scale = 8 * 1024 * 1024
        byte2MB_scale = 1024 * 1024
        with torch.no_grad():
            keep = self.get_mask_anchor.squeeze(1) > mask_prune_threshold
            anchor = self.get_anchor[keep]
            feat = self._anchor_feat[keep]
            scaling = self.get_scaling[keep]
            offsets = self._offset[keep].reshape(anchor.shape[0], -1)

            # Drop anchors whose every child-offset slot is inactive (neural_opacity<=0
            # for all n_offsets) -- training's accumulated opacity_accum pruning (see
            # adjust_anchor) already removes most such anchors, but that's an AVERAGE
            # over many past visits/timesteps, so a handful can still have zero active
            # slots at this EXACT final feat value (measured ~0.04% of anchors on a real
            # checkpoint) and slip through to the final checkpoint. These contribute
            # feat/scaling/position bits for literally nothing rendered, and -- same
            # argument as the keep-mask above -- dropping them needs no side info: the
            # decoder recomputes the identical "nothing here" fact from feat alone.
            # Checked here, before geometry gets encoded, so these anchors are excluded
            # from anchor_geom.b too, not just from feat/scaling/offsets.
            assert not self.add_opacity_dist, \
                "dead-anchor pruning assumes neural_opacity depends only on feat, not view direction/distance"
            has_active_offset = (self.get_opacity_mlp(feat) > 0.0).any(dim=1)
            anchor, feat, scaling, offsets = (
                anchor[has_active_offset], feat[has_active_offset],
                scaling[has_active_offset], offsets[has_active_offset])
            # rotation/opacity/mask below are re-sliced from self.* directly (not derived
            # from the already-filtered anchor/feat/scaling/offsets above), so they need
            # this SAME combined filter -- scatter has_active_offset (defined in the
            # keep-subspace) back into the full anchor-index space so `keep` alone can't
            # be reused by accident. Using plain `keep` for rotation/opacity/mask_soft
            # while anchor/feat/scaling/offsets used `keep & has_active_offset` was a real
            # bug: it gave those three tensors ~0.04% MORE rows than inv_perm's length,
            # so indexing with inv_perm silently pulled values from the wrong anchors
            # instead of erroring -- confirmed by direct measurement to corrupt
            # mask_anchor (which gates render-time opacity for every anchor, not just the
            # dropped ones) enough to cost ~7dB of PSNR on a real checkpoint.
            keep_prev = keep
            keep = keep.clone()
            keep[keep_prev] = has_active_offset

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
            if self.use_intra_anchor_context:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj, prob_feat = self.forward_grid(feat_context)
            else:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj).expand(N, self.feat_dim).contiguous()
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj).expand(N, 7).contiguous()
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj).expand(N, 4 * self.n_offsets).contiguous()

            feat_q = STE_multistep.apply(feat, Q_feat, feat.mean())
            scaling_q = STE_multistep.apply(scaling, Q_scaling, scaling.mean())
            offsets_q = STE_multistep.apply(offsets, Q_offsets, offsets.mean())

            # HAC/HAC++ filter offsets down to only the per-offset slots that actually
            # render something (neural_opacity > 0 -- the SAME criterion
            # gaussian_renderer.generate_neural_gaussians uses to mask out unrendered
            # child Gaussians) before arithmetic-coding them. Most slots are inactive
            # (measured ~89% on a real checkpoint), so coding the full dense tensor wastes
            # most of offsets.b's bits on values nothing ever renders. This needs NO side
            # information: which slots are active is a deterministic function of feat_q
            # alone (via mlp_opacity), which conduct_decoding reconstructs byte-identically
            # from the losslessly-decoded feat -- valid only because add_opacity_dist=False
            # (neural_opacity doesn't depend on view direction/distance, which the decoder
            # has no camera to recompute anyway); if that ever changes, this filtering
            # breaks and must be reverted to the dense scheme.
            assert not self.add_opacity_dist, \
                "offsets filtering assumes neural_opacity (and which offset slots are " \
                "active) depends only on feat, not on view direction/distance"
            offset_active = (self.get_opacity_mlp(feat_q) > 0.0)  # [N, n_offsets]
            offset_active_flat = offset_active.unsqueeze(-1).expand(N, self.n_offsets, 4).reshape(N, -1)

            if self.use_intra_anchor_context:
                # Sequential per-channel-group encode: group cc's context (mean_adj,
                # scale_adj, prob_adj) is predicted from feat_q's already-known groups
                # [0, cc) of this SAME anchor -- conduct_decoding must reproduce this
                # exact group order since it won't have feat_q's later groups yet.
                gs = self.mlp_channel_ctx.group_size
                mean_scale_prob = torch.cat([mean_feat, scale_feat, prob_feat], dim=-1)
                bits_feat = 0
                feat_b_name = os.path.join(path, 'feat.b')
                for cc in range(self.mlp_channel_ctx.n_groups):
                    sl = slice(cc * gs, (cc + 1) * gs)
                    mean_adj, scale_adj, prob_adj = self.mlp_channel_ctx(feat_q, mean_scale_prob, to_dec=cc)
                    probs = torch.softmax(torch.stack([prob_feat[:, sl], prob_adj], dim=-1), dim=-1)
                    bits_feat += encode_gaussian_mixed_chunk(
                        feat_q[:, sl].reshape(-1), mean_feat[:, sl].reshape(-1), mean_adj.reshape(-1),
                        scale_feat[:, sl].reshape(-1), scale_adj.reshape(-1),
                        probs[..., 0].reshape(-1), probs[..., 1].reshape(-1), Q_feat[:, sl].reshape(-1),
                        feat_b_name.replace('.b', f'_g{cc}.b'))
            else:
                bits_feat = encode_gaussian(feat_q.reshape(-1), mean_feat.reshape(-1), scale_feat.reshape(-1),
                                             Q_feat.reshape(-1), os.path.join(path, 'feat.b'))
            bits_scaling = encode_gaussian(scaling_q.reshape(-1), mean_scaling.reshape(-1), scale_scaling.reshape(-1),
                                            Q_scaling.reshape(-1), os.path.join(path, 'scaling.b'))
            bits_offsets = encode_gaussian(offsets_q[offset_active_flat], mean_offsets[offset_active_flat],
                                            scale_offsets[offset_active_flat], Q_offsets[offset_active_flat],
                                            os.path.join(path, 'offsets.b'))

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
            # a ~10dB PSNR drop between the uncompressed and decoded models.
            mask_soft = self.get_mask_anchor[keep][inv_perm].detach()
            mask_q = torch.round(mask_soft * 255).clamp(0, 255).to(torch.uint8)
            mask_q_f = mask_q.float().view(-1)

            # mask_offset/motion_gate (experimental, not yet entropy-coded) must be
            # re-sliced into this SAME keep+inv_perm canonical order and stored alongside
            # rotation/opacity so conduct_decoding's reconstructed anchor count/order
            # lines up -- otherwise they silently keep their pre-encoding N and crash (or
            # worse, silently misalign) the next time they're indexed by visible_mask.
            mask_offset_enc = None
            if self.use_offset_mask and self._mask_offset.numel() > 0:
                mask_offset_enc = self._mask_offset[keep][inv_perm].detach()
            motion_gate_enc = None
            if self.use_motion_gate and self._motion_gate.numel() > 0:
                motion_gate_enc = self._motion_gate[keep][inv_perm].detach()

            # Entropy-code mask_q itself instead of storing it raw (flat 8 bits/anchor).
            # Its empirical distribution is heavily right-skewed -- most survivors sit just
            # above mask_prune_threshold, with a long tail toward 255 -- which a 2-component
            # Gaussian mixture fits far better than a single Gaussian (measured ~27% smaller
            # than the flat 8 bits/anchor on a real checkpoint, within a few percentage
            # points of the empirical Shannon-entropy bound for the same data; a single
            # Gaussian only gets ~16%). The mixture is fit fresh at encode time -- one
            # GLOBAL model for the whole checkpoint, not a per-anchor hyper-network
            # prediction -- so the only side info the decoder needs is its 6 scalars
            # (2 means, 2 scales, 2 weights), saved below.
            mask_means, mask_scales, mask_weights = fit_gmm_1d_2comp(mask_q_f)
            mask_mean1 = mask_means[0].expand(N).contiguous()
            mask_mean2 = mask_means[1].expand(N).contiguous()
            mask_scale1 = mask_scales[0].expand(N).contiguous()
            mask_scale2 = mask_scales[1].expand(N).contiguous()
            mask_prob1 = mask_weights[0].expand(N).contiguous()
            mask_prob2 = mask_weights[1].expand(N).contiguous()
            # mean/scale are GLOBAL constants here (one fit for the whole checkpoint, not
            # per-anchor), so x's quantized-symbol range is trivially bounded to [0,255] --
            # no chunking needed for the int16/SAFE_RANGE safety the _chunk variants exist
            # for (see utils/arithmetic_coding.py's chunking comment); one file is simpler
            # and avoids the small per-chunk fixed-header overhead (32*3 bits/chunk).
            bits_mask = encode_gaussian_mixed(
                mask_q_f, mask_mean1, mask_mean2, mask_scale1, mask_scale2, mask_prob1, mask_prob2,
                1.0, os.path.join(path, 'mask.b'))

            torch.save({
                'anchor_mins': anchor_mins.cpu(),
                'mask_gmm_params': torch.stack([mask_means, mask_scales, mask_weights]).cpu(),  # [3, 2]
                'rotation': (rotation[0:1] if rotation_is_constant else rotation).cpu(),
                'rotation_is_constant': rotation_is_constant,
                'opacity': (opacity[0:1] if opacity_is_constant else opacity).cpu(),
                'opacity_is_constant': opacity_is_constant,
                # Marks offsets.b as the filtered (active-slots-only) format -- lets
                # conduct_decoding tell this apart from bitstreams written before offsets
                # filtering existed, which coded the full dense tensor instead.
                'offsets_filtered': True,
                'mask_offset': mask_offset_enc.cpu() if mask_offset_enc is not None else None,
                'motion_gate': motion_gate_enc.cpu() if motion_gate_enc is not None else None,
            }, os.path.join(path, 'uncoded_attrs.pt'))

            return {
                'anchor_num': N,
                'anchor_geom_MB': os.path.getsize(os.path.join(path, 'anchor_geom.b')) / byte2MB_scale,
                'mask_MB': bits_mask / bit2MB_scale,
                'offsets_active_frac': offset_active_flat.float().mean().item(),
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
            if self.use_intra_anchor_context:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj, prob_feat = self.forward_grid(feat_context)
            else:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = self.forward_grid(feat_context)

            Q_feat = self.adaptive_Q(1.0, Q_feat_adj).expand(N, self.feat_dim).contiguous()
            Q_scaling = self.adaptive_Q(0.001, Q_scaling_adj).expand(N, 7).contiguous()
            Q_offsets = self.adaptive_Q(0.2, Q_offsets_adj).expand(N, 4 * self.n_offsets).contiguous()

            if self.use_intra_anchor_context:
                # Sequential per-channel-group decode, mirroring conduct_encoding: group
                # cc is decoded using groups [0, cc) already written into `feat` (group 0
                # has no dependency, so the still-zero placeholder for groups >= cc is
                # never read by ChannelCtxFeat's forward for to_dec=cc).
                gs = self.mlp_channel_ctx.group_size
                mean_scale_prob = torch.cat([mean_feat, scale_feat, prob_feat], dim=-1)
                feat = torch.zeros(N, self.feat_dim, device=anchor.device)
                feat_b_name = os.path.join(path, 'feat.b')
                for cc in range(self.mlp_channel_ctx.n_groups):
                    sl = slice(cc * gs, (cc + 1) * gs)
                    mean_adj, scale_adj, prob_adj = self.mlp_channel_ctx(feat, mean_scale_prob, to_dec=cc)
                    probs = torch.softmax(torch.stack([prob_feat[:, sl], prob_adj], dim=-1), dim=-1)
                    feat[:, sl] = decode_gaussian_mixed_chunk(
                        mean_feat[:, sl].reshape(-1), mean_adj.reshape(-1),
                        scale_feat[:, sl].reshape(-1), scale_adj.reshape(-1),
                        probs[..., 0].reshape(-1), probs[..., 1].reshape(-1), Q_feat[:, sl].reshape(-1),
                        feat_b_name.replace('.b', f'_g{cc}.b')).view(N, gs)
            else:
                feat = decode_gaussian(mean_feat.reshape(-1), scale_feat.reshape(-1),
                                        Q_feat.reshape(-1), os.path.join(path, 'feat.b')).view(N, self.feat_dim)
            scaling_act = decode_gaussian(mean_scaling.reshape(-1), scale_scaling.reshape(-1),
                                          Q_scaling.reshape(-1), os.path.join(path, 'scaling.b')).view(N, 7)
            if uncoded.get('offsets_filtered', False):
                # Mirrors conduct_encoding's filtering exactly: which offset slots are
                # active is recomputed from the already-decoded `feat` via the same
                # mlp_opacity call, so decode needs no extra side info for this -- see
                # conduct_encoding's comment on offset_active for why that's valid.
                assert not self.add_opacity_dist, \
                    "offsets filtering assumes neural_opacity depends only on feat, not view direction/distance"
                offset_active = (self.get_opacity_mlp(feat) > 0.0)  # [N, n_offsets]
                offset_active_flat = offset_active.unsqueeze(-1).expand(N, self.n_offsets, 4).reshape(N, -1)
                offsets_active_vals = decode_gaussian(
                    mean_offsets[offset_active_flat], scale_offsets[offset_active_flat],
                    Q_offsets[offset_active_flat], os.path.join(path, 'offsets.b'))
                offsets_flat = torch.zeros(N, 4 * self.n_offsets, device=anchor.device)
                offsets_flat[offset_active_flat] = offsets_active_vals
                offsets = offsets_flat.view(N, self.n_offsets, 4)
            else:
                # Fallback for bitstreams written before offsets filtering existed (full
                # dense tensor, no mask needed to reconstruct it).
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
            # comment on mask_q for why that matters. mask_q is entropy-coded (2-component
            # Gaussian mixture, global params stored in uncoded_attrs.pt) rather than
            # stored raw; 'mask_q' is kept as a fallback to still decode bitstreams written
            # before that change.
            if 'mask_gmm_params' in uncoded:
                gmm_params = uncoded['mask_gmm_params'].cuda()
                mask_means, mask_scales, mask_weights = gmm_params[0], gmm_params[1], gmm_params[2]
                mask_mean1 = mask_means[0].expand(N).contiguous()
                mask_mean2 = mask_means[1].expand(N).contiguous()
                mask_scale1 = mask_scales[0].expand(N).contiguous()
                mask_scale2 = mask_scales[1].expand(N).contiguous()
                mask_prob1 = mask_weights[0].expand(N).contiguous()
                mask_prob2 = mask_weights[1].expand(N).contiguous()
                mask_q_f = decode_gaussian_mixed(
                    mask_mean1, mask_mean2, mask_scale1, mask_scale2, mask_prob1, mask_prob2,
                    1.0, os.path.join(path, 'mask.b'))
                mask_soft = mask_q_f.clamp(0, 255) / 255
            else:
                mask_soft = uncoded['mask_q'].cuda().float() / 255
            self._mask_anchor = nn.Parameter(inverse_sigmoid(mask_soft.clamp(1e-6, 1 - 1e-6)).view(N, 1))

            # Restore mask_offset/motion_gate (raw, uncoded -- see conduct_encoding) in
            # the same N and canonical order as every other just-reconstructed per-anchor
            # tensor above, so a stale pre-decode size never reaches a later indexing by
            # visible_mask.
            if uncoded.get('mask_offset') is not None:
                self._mask_offset = nn.Parameter(uncoded['mask_offset'].cuda())
                self.use_offset_mask = True
            if uncoded.get('motion_gate') is not None:
                self._motion_gate = nn.Parameter(uncoded['motion_gate'].cuda())
                self.use_motion_gate = True

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
            if self.use_intra_anchor_context:
                self.mlp_channel_ctx.eval()

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
            if self.use_intra_anchor_context:
                self.mlp_channel_ctx.train()

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

    def set_appearance(self, physical_camera_ids):
        # Indexed by PHYSICAL camera identity (e.g. N3DV's "camNN" rig index, shared across
        # every frame of that view), not by Camera.uid (an arbitrary per-(view,frame)
        # enumerate index with no semantic meaning -- see gaussian_renderer/__init__.py's
        # appearance lookup for why that distinction matters: N3DV holds out one whole
        # physical camera for test, which then has NO trained embedding row at all).
        if self.appearance_dim > 0:
            self.appearance_id_map = {pid: i for i, pid in enumerate(sorted(set(physical_camera_ids)))}
            self.embedding_appearance = Embedding(len(self.appearance_id_map), self.appearance_dim).cuda()

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
    def get_time_basis_mlp(self):
        return self.mlp_time_basis

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
        # 7, not 8: scaling's 8th channel was dead (never read by anything that affects
        # the rendered image -- see conduct_encoding's comment), so it's been dropped
        # entirely rather than kept around and excluded only at encode time.
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 7)
        
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
        if self.use_entropy_coding:
            # Running sum of get_mask_anchor's value each time an anchor is visible, alongside
            # anchor_demon's visit count -- averaging the two (in adjust_anchor) gives the
            # rate-distortion mask's value AGGREGATED OVER EVERY (view, time) sample seen in
            # this check_interval, instead of the single most-recent training iteration's
            # instantaneous value. In a dynamic 4D scene each iteration renders a different
            # (camera, timestamp) pair, so an anchor only relevant at one moment can look
            # spuriously prunable (or keepable) at any single instant; a 3D static scene's
            # anchors don't have this failure mode since only the camera varies iteration to
            # iteration, never "when". See adjust_anchor's rate_prune_mask.
            self.mask_val_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
            self.mask_val_last = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.photo_error_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.flow_speed_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
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

        if self.motion_model in ('basis', 'taylor_residual'):
            # NOT mlp_flow's own LR: mlp_time_basis is a tiny dt-only network that gets a
            # gradient contribution from EVERY rendered offset in EVERY iteration (it's
            # multiplied elementwise against per-offset coefficients and summed), so its
            # effective gradient is far larger/more concentrated than any per-anchor-feat
            # MLP's. At the full mlp_flow LR this diverges to NaN around the first
            # densification events (~iter 2000-2500) in practice; 100x smaller keeps the
            # same bilinear structure but stable.
            l.append({'params': self.mlp_time_basis.parameters(), 'lr': training_args.mlp_flow_lr_init * 0.01, "name": "mlp_time_basis"})

        if self.use_entropy_coding:
            l.append({'params': self.mlp_grid.parameters(), 'lr': training_args.mlp_grid_lr_init, "name": "mlp_grid"})
            l.append({'params': self.encoding_xyz.parameters(), 'lr': training_args.encoding_xyz_lr_init, "name": "encoding_xyz"})
            l.append({'params': [self._mask_anchor], 'lr': training_args.mask_lr, "name": "mask_anchor"})
            if self.use_intra_anchor_context:
                # Reuses mlp_grid's LR schedule rather than a dedicated one: it's a
                # similarly-sized MLP predicting the same kind of (mean, scale, prob)
                # entropy-model parameters, just from intra- instead of inter-anchor
                # context, so no reason to expect it needs different LR dynamics.
                l.append({'params': self.mlp_channel_ctx.parameters(), 'lr': training_args.mlp_grid_lr_init,
                          "name": "mlp_channel_ctx"})

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
            if param_group["name"] == "mlp_time_basis":
                lr = self.mlp_flow_scheduler_args(iteration) * 0.01
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
            if self.use_entropy_coding and self.use_intra_anchor_context and param_group["name"] == "mlp_channel_ctx":
                lr = self.mlp_grid_scheduler_args(iteration)
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
        if self.use_offset_mask and self._mask_offset.numel() > 0:
            for i in range(self._mask_offset.shape[1]):
                l.append('mask_offset_{}'.format(i))
        if self.use_motion_gate and self._motion_gate.numel() > 0:
            l.append('motion_gate')
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
        if self.use_offset_mask and self._mask_offset.numel() > 0:
            to_cat.append(self._mask_offset.detach().cpu().numpy())
        if self.use_motion_gate and self._motion_gate.numel() > 0:
            to_cat.append(self._motion_gate.detach().cpu().numpy())
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

        # mask_offset/motion_gate are allocated mid-training (see update_until in
        # train.py) rather than through a constructor flag, so whether to restore them
        # is inferred from the ply's own properties rather than self.use_offset_mask /
        # self.use_motion_gate (which default to False on a freshly constructed model,
        # e.g. the decode-time GaussianModel built for compressed-render evaluation).
        ply_prop_names = [p.name for p in plydata.elements[0].properties]
        mask_offset_names = [n for n in ply_prop_names if n.startswith("mask_offset_")]
        if mask_offset_names:
            mask_offset_names = sorted(mask_offset_names, key=lambda x: int(x.split('_')[-1]))
            mask_offset = np.zeros((anchor.shape[0], len(mask_offset_names)), dtype=np.float32)
            for idx, attr_name in enumerate(mask_offset_names):
                mask_offset[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
            self._mask_offset = nn.Parameter(torch.tensor(mask_offset, dtype=torch.float, device="cuda").requires_grad_(True))
            self.use_offset_mask = True
        if "motion_gate" in ply_prop_names:
            motion_gate = np.asarray(plydata.elements[0]["motion_gate"])[..., np.newaxis].astype(np.float32)
            self._motion_gate = nn.Parameter(torch.tensor(motion_gate, dtype=torch.float, device="cuda").requires_grad_(True))
            self.use_motion_gate = True

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
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask, opacity_t, sigma, lambda_temporal_sigma, timestamp, opt, activate_DA, neural_xyz=None, pixel_error_map=None, viewpoint_camera=None, flow_speed=None):
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

        if self.use_entropy_coding:
            # accumulate this iteration's (view, time) sample's mask value -- averaged against
            # anchor_demon's matching visit count in adjust_anchor's rate_prune_mask
            cur_mask_val = self.get_mask_anchor[anchor_visible_mask].detach()
            self.mask_val_accum[anchor_visible_mask] += cur_mask_val
            # ablation_no_time_agg_mask companion: overwrite (not +=) so this always
            # holds just the latest visit's value, never an average.
            self.mask_val_last[anchor_visible_mask] = cur_mask_val

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

        if neural_xyz is not None and pixel_error_map is not None:
            # Photometric-error densification signal: sample this step's rendered-vs-GT
            # per-pixel L1 error at each visible neural gaussian's own screen location,
            # and accumulate it the same way as grad_norm above (same combined_mask,
            # same per-visit averaging via offset_denom). Used by anchor_growing as an
            # ADDITIONAL (OR'd, never subtracted) growth trigger alongside the existing
            # gradient-norm criterion -- see anchor_growing's photo_mask.
            xyz_vis = neural_xyz[update_filter]
            ones = torch.ones_like(xyz_vis[:, :1])
            clip = torch.cat([xyz_vis, ones], dim=-1) @ viewpoint_camera.full_proj_transform
            w = clip[:, 3:4].clamp(min=1e-6)
            ndc = (clip[:, :2] / w).clamp(min=-1.5, max=1.5)
            grid = ndc.view(1, -1, 1, 2)
            sampled_error = torch.nn.functional.grid_sample(
                pixel_error_map.unsqueeze(0), grid, mode='bilinear',
                padding_mode='border', align_corners=False
            ).view(-1, 1)
            self.photo_error_accum[combined_mask] += sampled_error * weight

        if flow_speed is not None:
            # Same masked-space layout as opacity_t/sigma above (both come from the same
            # mask-selected `masked` split in generate_neural_gaussians), unlike
            # photo_error's projection -- flow_speed is already per-offset, so it only
            # needs the same temp_mask scatter opacity_t/sigma use, not grid_sample.
            temp_flow_speed = torch.zeros_like(temp_mask, dtype=torch.float)
            temp_flow_speed[temp_mask] = flow_speed.squeeze(-1)
            self.flow_speed_accum[combined_mask] += temp_flow_speed[combined_mask].unsqueeze(-1) * weight




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

    
    def anchor_growing(self, opt, grads, threshold, offset_mask, timestamp, photo_error=None, flow_speed=None):
        ##
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            if i == 0 and photo_error is not None and getattr(opt, 'photo_densify_threshold', 0.0) > 0:
                # Additional, purely additive growth trigger (see training_statis):
                # applied only at the finest level (i==0), so it never interacts with
                # the hierarchical grad-threshold schedule used for i>0 -- it can only
                # ever ADD candidates the gradient signal missed, never remove any.
                photo_mask = (photo_error >= opt.photo_densify_threshold)
                candidate_mask = torch.logical_or(candidate_mask, photo_mask)
            if i == 0 and flow_speed is not None and getattr(opt, 'flow_densify_threshold', 0.0) > 0:
                # Same purely-additive pattern as photo_mask above, driven by average
                # per-visit motion speed instead of appearance error -- grows offsets in
                # fast-moving regions even when their screen-space gradient (which
                # responds to appearance error, not motion per se) doesn't cross
                # densify_grad_threshold. Targets the under-resolved-detail-in-fast-
                # motion-regions failure mode directly, at the density-growth level
                # rather than indirectly via the temporal-opacity window (see
                # motion_sigma_reg_weight).
                flow_mask = (flow_speed >= opt.flow_densify_threshold)
                candidate_mask = torch.logical_or(candidate_mask, flow_mask)
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
                # 7 cols total, not 8: the second block used to be .repeat([1,2]) (2
                # copies of t_size, landing in the now-removed dead channel 7 -- see
                # conduct_encoding's comment on it), so this just drops that duplicate,
                # leaving every surviving column's value identical to before.
                new_scaling = torch.cat([
                    torch.ones_like(candidate_anchor[:,:3]).repeat([1,2]).float().cuda()*cur_size,
                    torch.ones_like(candidate_anchor[:,3:4]).float().cuda()*t_size
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

                if self.use_entropy_coding:
                    temp_mask_val_accum = torch.cat([self.mask_val_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                    del self.mask_val_accum
                    self.mask_val_accum = temp_mask_val_accum

                    temp_mask_val_last = torch.cat([self.mask_val_last, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                    del self.mask_val_last
                    self.mask_val_last = temp_mask_val_last

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

        # Photometric-error densification signal (see training_statis): average per-visit
        # rendered-vs-GT error at this offset's own screen location, same denom/visit-count
        # gate as the gradient signal above. Purely additive in anchor_growing (OR'd with
        # the existing grad_threshold test, never replacing or weakening it), so this can
        # only ever add growth candidates the gradient signal missed, not remove any.
        photo_error = self.photo_error_accum / self.offset_denom
        photo_error[photo_error.isnan()] = 0.0
        photo_error_norm = photo_error.squeeze(-1)

        # Flow-magnitude densification signal (see training_statis): same additive,
        # OR'd-in pattern as photo_error above -- see anchor_growing's flow_mask.
        flow_speed_avg = self.flow_speed_accum / self.offset_denom
        flow_speed_avg[flow_speed_avg.isnan()] = 0.0
        flow_speed_norm = flow_speed_avg.squeeze(-1)

        timestamp = (self.offset_time_accum / self.offset_time_denom).squeeze()

        self.anchor_growing(opt, grads_norm, grad_threshold, offset_mask, timestamp, photo_error_norm, flow_speed_norm)

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

        self.photo_error_accum[offset_mask] = 0
        padding_photo_error_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.photo_error_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.photo_error_accum.device)
        self.photo_error_accum = torch.cat([self.photo_error_accum, padding_photo_error_accum], dim=0)

        self.flow_speed_accum[offset_mask] = 0
        padding_flow_speed_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.flow_speed_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.flow_speed_accum.device)
        self.flow_speed_accum = torch.cat([self.flow_speed_accum, padding_flow_speed_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N]
        if self.use_entropy_coding:
            # rate-distortion mask: anchors the model has learned are not worth their coding cost.
            # Averaged over every (view, time) sample seen since the last check (mask_val_accum /
            # anchor_demon), not the single most-recent iteration's instantaneous value -- a 4D
            # anchor's relevance can vary a lot across the time range, and each iteration only
            # renders one timestamp, so the instantaneous value is a noisy single sample of that.
            # Anchors with zero visits this interval (anchor_demon==0, e.g. just grown) are
            # excluded rather than defaulting to "prunable": mask_val_accum/anchor_demon would
            # otherwise read as 0 (below any positive threshold) for every one of them.
            visited = self.anchor_demon.squeeze(dim=1) > 0
            if self.ablation_no_time_agg_mask:
                # Ablation: use only the latest visit's instantaneous mask value
                # (mask_val_last) instead of the time-aggregated average -- reintroduces
                # the premature-pruning failure mode this feature exists to fix (see
                # comment above).
                mask_avg = self.mask_val_last.squeeze(dim=1)
            else:
                mask_avg = (self.mask_val_accum / self.anchor_demon.clamp(min=1)).squeeze(dim=1)
            rate_prune_mask = visited & (mask_avg < opt.mask_prune_threshold)
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

        photo_error_accum = self.photo_error_accum.view([-1, self.n_offsets])[~prune_mask]
        photo_error_accum = photo_error_accum.view([-1, 1])
        del self.photo_error_accum
        self.photo_error_accum = photo_error_accum

        flow_speed_accum = self.flow_speed_accum.view([-1, self.n_offsets])[~prune_mask]
        flow_speed_accum = flow_speed_accum.view([-1, 1])
        del self.flow_speed_accum
        self.flow_speed_accum = flow_speed_accum

        offset_time_accum = self.offset_time_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_time_accum = offset_time_accum.view([-1, 1])
        del self.offset_time_accum
        self.offset_time_accum = offset_time_accum
        
        # update opacity accum
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            if self.use_entropy_coding:
                self.mask_val_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        if self.use_entropy_coding:
            temp_mask_val_accum = self.mask_val_accum[~prune_mask]
            del self.mask_val_accum
            self.mask_val_accum = temp_mask_val_accum

            # mask_val_last is intentionally NOT zeroed alongside mask_val_accum above
            # (anchors_mask branch) -- it must keep holding each anchor's real last-seen
            # value between visits, not reset to 0 (which would read as "below any
            # positive threshold" and cause spurious pruning before the anchor is next
            # visited). Only shape-filtered here to track pruning, same as every other
            # per-anchor buffer.
            temp_mask_val_last = self.mask_val_last[~prune_mask]
            del self.mask_val_last
            self.mask_val_last = temp_mask_val_last

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

            if self.motion_model in ('basis', 'taylor_residual'):
                # Plain state_dict (not jit.trace) -- matches mlp_channel_ctx's pattern just
                # below; a 1-input MLP traces fine either way, but state_dict avoids any
                # risk of tracing baking in example-input-specific behavior.
                self.mlp_time_basis.eval()
                torch.save(self.mlp_time_basis.state_dict(), os.path.join(path, 'time_basis_mlp.pt'))
                self.mlp_time_basis.train()

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
                hash_feat_dim = sum(encoder.output_dim for encoder in self.encoding_xyz)
                grid_mlp = torch.jit.trace(self.mlp_grid, (torch.rand(1, hash_feat_dim).cuda()))
                grid_mlp.save(os.path.join(path, 'grid_mlp.pt'))
                self.mlp_grid.train()

                self.save_hash_grid(os.path.join(path, 'encoding_xyz.bin'))
                torch.save({'x_bound_min': self.x_bound_min, 'x_bound_max': self.x_bound_max},
                           os.path.join(path, 'anchor_bound.pt'))

                if self.use_intra_anchor_context:
                    # Not torch.jit.trace'd like the other MLPs above: ChannelCtxFeat's
                    # forward() branches on to_dec (which group to return, and how many
                    # of the preceding groups to actually run), and tracing only captures
                    # the ONE branch taken with whatever example input is given -- it
                    # would silently bake in a single to_dec value and break every other
                    # group's encode/decode call. Plain state_dict preserves the real
                    # Python control flow.
                    self.mlp_channel_ctx.eval()
                    torch.save(self.mlp_channel_ctx.state_dict(), os.path.join(path, 'channel_ctx_mlp.pt'))
                    self.mlp_channel_ctx.train()

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
            if self.motion_model in ('basis', 'taylor_residual'):
                self.mlp_time_basis.load_state_dict(torch.load(os.path.join(path, 'time_basis_mlp.pt')))
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
                if self.use_intra_anchor_context:
                    self.mlp_channel_ctx.load_state_dict(torch.load(os.path.join(path, 'channel_ctx_mlp.pt')))
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

    @staticmethod
    def _quantize_int8(tensor):
        """Per-tensor symmetric int8 quantization. Returns (int8_tensor_cpu, scale)."""
        scale = (tensor.abs().max().clamp(min=1e-12) / 127.0).item()
        q = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        return q.cpu(), scale

    def save_mlp_checkpoints_int8(self, path):
        """Export-only, lossy-but-cheap alternative to save_mlp_checkpoints(mode='split'):
        quantizes every parameter of the 6 small MLPs (opacity/cov/color/flow/grid/
        channel_ctx) to int8 + a per-tensor float scale, bypassing torch.jit entirely
        (jit has no native int8 linear-layer path, and its fixed per-module container
        overhead would eat most of the theoretical saving anyway -- confirmed in
        practice: fp16 via jit only achieved ~34% size reduction, not the ~50% the
        bit-width alone implies). Does NOT replace save_mlp_checkpoints -- call this
        separately at final export time once a checkpoint is otherwise finished;
        training/encoding keeps using the normal fp32 jit-traced path throughout.
        Confirmed on a real checkpoint: -0.26dB PSNR cost, ~72% smaller mlps component
        (see load_mlp_checkpoints_int8's docstring for the matching load path)."""
        mkdir_p(path)
        modules = {
            'opacity_mlp': self.mlp_opacity, 'cov_mlp': self.mlp_cov,
            'color_mlp': self.mlp_color, 'flow_mlp': self.mlp_flow,
        }
        if self.use_entropy_coding:
            modules['grid_mlp'] = self.mlp_grid
            if self.use_intra_anchor_context:
                modules['channel_ctx_mlp'] = self.mlp_channel_ctx
        out = {}
        for name, m in modules.items():
            out[name] = {pname: self._quantize_int8(p.data) for pname, p in m.named_parameters()}
        torch.save(out, os.path.join(path, 'mlps_int8.pt'))

    def load_mlp_checkpoints_int8(self, path):
        """Inverse of save_mlp_checkpoints_int8: dequantizes back to fp32 and
        load_state_dict's into the already-constructed self.mlp_* modules (built fresh
        in __init__, same ones load_mlp_checkpoints(mode='split') would otherwise
        jit.load over) -- no need to reconstruct the Sequential architectures here."""
        state = torch.load(os.path.join(path, 'mlps_int8.pt'))
        modules = {
            'opacity_mlp': self.mlp_opacity, 'cov_mlp': self.mlp_cov,
            'color_mlp': self.mlp_color, 'flow_mlp': self.mlp_flow,
        }
        if self.use_entropy_coding:
            modules['grid_mlp'] = self.mlp_grid
            if self.use_intra_anchor_context:
                modules['channel_ctx_mlp'] = self.mlp_channel_ctx
        for name, m in modules.items():
            sd = {pname: (q.float() * scale).cuda() for pname, (q, scale) in state[name].items()}
            m.load_state_dict(sd)
            m.cuda()
