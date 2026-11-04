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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs='+', type=float)
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self.feat_dim = 32
        self.n_offsets = 10
        self.voxel_size =  0.001 # if voxel_size<=0, using 1nn dist
        self.t_grid_size = 0.0333
        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.lod = 0

        self.appearance_dim = 32
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        self.dataloader = False
        # Perf: cache each (lazily-loaded, --dataloader) training image's decoded+resized
        # pixels in RAM after its first access, keyed by dataset index, instead of
        # re-decoding from disk via PIL every time it's resampled. Stored as uint8 (the
        # same integer values PILtoTorch's np.array() reads, before its /255.0) to keep
        # the cache's RAM footprint to 1x the dataset's decoded size rather than 4x
        # (float32). Only meaningful when `dataloader` is set (the eager, no-dataloader
        # path already decodes once via Camera.image and needs no help). Meant to be
        # paired with --num_workers 0: with workers > 0, each worker process gets its OWN
        # private copy of this dict (multiprocessing forks don't share a mutable dict
        # across processes), multiplying the cache's RAM cost by num_workers instead of
        # keeping it at 1x the dataset's decoded size.
        self.cache_images = False
        self.time_duration = [0.0, 10.0]
        self.frame_ratio = 1.0
        self.max_init_t = 0.0
        self.init_voxel_scale = 1.0
        self.ply_path = ''

        self.hparam_beta = 4.0 # Beta in the paper

        # Ablation options
        self.temporal_opacity = '4dgs' # ['4dgs', 'ours']
        self.use_flow = False
        # Ablation flag for validating the Mip-Splatting 3D smoothing filter (see
        # compute_3D_filter in gaussian_model.py): True reproduces pre-filter behavior
        # exactly (filter_3D is never computed, so generate_neural_gaussians' fallback
        # adds zero variance), for apples-to-apples baseline comparisons. Default False
        # (filter ON) is the actual intended behavior going forward.
        self.disable_mip_filter = False
        # Ablation: how per-offset xyz(t) motion beyond the base anchor is modeled.
        # 'linear'          -- current default: xyz(t) = base + velocity(feat)*dt
        # 'basis'           -- DynMF/TED-4DGS-style: xyz(t) = base + sum_b coeff_b(feat)*Phi_b(dt),
        #                      Phi a shared *learned* small MLP over dt (motion_basis_dim bases);
        #                      replaces the velocity term entirely but stays bitstream-neutral
        #                      (Phi is a fixed shared network, coeff comes from already-coded feat).
        # 'taylor_residual' -- Taylor-Gaussian-style hybrid: explicit 3rd-order Taylor
        #                      (velocity+acceleration+jerk) PLUS a small learned-basis residual
        #                      correction (half motion_basis_dim bases) added on top.
        # 'fourier'         -- RD4DGS/DynMF-Fourier-ablation-style: same combination as 'basis'
        #                      but Phi is a FIXED sin/cos harmonic basis (not learned) -- a
        #                      low-frequency-only control expected to underperform 'basis'.
        self.motion_model = 'linear'
        self.motion_basis_dim = 16
        self.sigma_denom_weight = False
        self.disable_denom_weight = False
        self.filter_opacity = False

        # entropy coding (HAC-style hash-grid context model)
        self.use_entropy_coding = False
        self.hash_n_features = 2
        self.hash_log2_size = 19
        self.noise_start_iter = 3_000
        self.entropy_start_iter = 10_000
        # HAC++-style intra-anchor channel context: splits anchor_feat into groups coded
        # sequentially (each conditioned on the same anchor's earlier groups), mixed with
        # the hash-grid's inter-anchor prediction via a 2-component Gaussian mixture. Only
        # takes effect when use_entropy_coding is also set.
        self.use_intra_anchor_context = False
        # Ablation: single true 4D (x,y,z,t) joint hash grid instead of the 4 factored 3D
        # triplet grids (HASH_TRIPLETS). Only takes effect when use_entropy_coding is set.
        self.use_joint_4d_hash = False
        # Ablation: joint 4D hash grid (shortened resolution schedule) PLUS a dedicated
        # xyz-only 3D hash grid. Mutually exclusive with use_joint_4d_hash. Only takes
        # effect when use_entropy_coding is set.
        self.use_hybrid_4d_hash = False
        # Ablation: joint 4D hash grid (shortened schedule) PLUS all 4 original 3D triplet
        # grids (5 tables total). Mutually exclusive with use_joint_4d_hash/use_hybrid_4d_hash.
        # Only takes effect when use_entropy_coding is set.
        self.use_full_hybrid_4d_hash = False
        # Ablation: drop the hash-grid context entirely. mlp_grid is fed a constant
        # (zero) input instead of calc_interp_feat's hash features, so every anchor gets
        # the same predicted mean/scale/Q regardless of its (x,y,z,t) -- i.e. no
        # inter-anchor spatiotemporal context at all. use_intra_anchor_context (if set)
        # still works normally on top of this. Only takes effect when use_entropy_coding
        # is set; encoding_xyz is not constructed at all in this mode.
        self.ablation_no_hash_context = False
        # Ablation: skip the per-element adaptive quantization step (adaptive_Q's Q_adj
        # MLP output is ignored) and always quantize at the fixed base step Q_base
        # (1.0 / 0.001 / 0.2 for feat/scaling/offsets). Only takes effect when
        # use_entropy_coding is set.
        self.ablation_fixed_quant = False
        # Ablation: prune anchors using only the CURRENT iteration's rate-distortion mask
        # value (self.get_mask_anchor at the moment of the periodic prune check) instead
        # of the running average over every (view, time) sample since the last check
        # (mask_val_accum / anchor_demon). Only takes effect when use_entropy_coding is
        # set.
        self.ablation_no_time_agg_mask = False
        # Ablation: replace the 4D anisotropic hyperoctree anchor-coordinate coding
        # (utils/octree_coding.py's encode_positions/decode_positions) with a naive
        # fixed-width-per-axis packing (encode_positions_naive/decode_positions_naive) --
        # no occupancy tree, no context modeling, no arithmetic coding. Only affects the
        # anchor_geom.b bitstream (anchor position); feat/scaling/offsets coding is
        # unchanged. Only takes effect when use_entropy_coding is set.
        self.ablation_no_anchor_coord_coding = False

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.vis = False
        self.vis_interval = 100
        self.vis_idx = 0
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0
        self.position_lr_final = 0.0
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        
        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = 30_000

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        
        
        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002  
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = 30_000

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = 30_000
        
        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = 30_000

        self.mlp_flow_lr_init = 0.001
        self.mlp_flow_lr_final = 0.001
        self.mlp_flow_lr_delay_mult = 0.01
        self.mlp_flow_lr_max_steps = 30_000
        
        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = 30_000

        self.appearance_lr_init = 0.05
        self.appearance_lr_final = 0.0005
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = 30_000

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        
        # for anchor densification
        self.start_stat = 500
        self.update_from = 1500
        self.update_interval = 100
        self.update_until = 15_000
        
        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        # Additional densification trigger (see anchor_growing's photo_mask): grow an
        # offset whose average per-visit rendered-vs-GT L1 error (sampled at its own
        # screen location) exceeds this, even if its gradient norm doesn't cross
        # densify_grad_threshold. Purely additive -- never removes a gradient-based
        # growth candidate. 0 disables it entirely (falls back to the original
        # gradient-only behavior). Default picked conservatively above typical average
        # L1 (~0.025-0.03 on N3DV scenes) so it only fires on persistently bad regions.
        self.photo_densify_threshold = 0.1
        # Additional densification trigger (see anchor_growing's flow_mask): grow an
        # offset whose average per-visit motion speed (flow[:, :3].norm(), range [0,
        # sqrt(3)] for 'linear'/'taylor_residual' motion models since flow's first 3
        # channels are Tanh-bounded velocity) exceeds this, even if its gradient norm
        # doesn't cross densify_grad_threshold. Same purely-additive pattern as
        # photo_densify_threshold. 0.0 (default) disables it entirely -- backward
        # compatible. Needs train.py to actually pass flow_speed into training_statis
        # (gated the same way as motion_sigma_reg_weight: only when use_flow and
        # motion_model in ('linear', 'taylor_residual')).
        self.flow_densify_threshold = 0.0
        # Debug/stability flag: force global grad-norm clipping on even for
        # motion_model='linear' (normally gated off there -- see the optimizer-step
        # comment in train.py). Use when a specific (scene, lambda) combination shows
        # gradient-blowup-style NaNs/CUDA illegal-memory-access that a non-'linear'
        # motion model would already have been protected against. Default False.
        self.force_grad_clip = False

        self.lambda_temporal_sigma = 1.0 # Gamma in the paper

        # Encourages a narrower temporal opacity window (larger sigma) specifically where
        # the learned motion velocity is large, via loss += weight * (flow_speed.detach()
        # * exp(-sigma)).mean() (see train.py). sigma is already per-offset/anchor-adaptive
        # (learned from get_cov_mlp's output, not a global constant), but nothing currently
        # gives it an explicit incentive to track motion speed -- it only gets photometric
        # gradient, which fast-moving regions (fewer sharp frames to anchor against) may
        # supply too weakly. 0.0 (default) disables this entirely -- backward compatible,
        # zero behavior change for any existing run. Only meaningful for motion_model in
        # ('linear', 'taylor_residual'), where flow's first 3 channels are a literal
        # velocity; silently skipped (no-op) for 'basis'/'fourier' where they aren't.
        self.motion_sigma_reg_weight = 0.0

        # E-D3DGS-style local motion-smoothness regularization: penalizes squared
        # difference between a Gaussian's learned flow and its nearest neighbor's (within
        # a random subsample), encouraging spatially close Gaussians to move similarly.
        # See the sample computation in train.py's loss section.
        self.motion_smooth_weight = 0.0
        self.motion_smooth_sample = 4096

        # Ex4DGS-inspired static/dynamic split: a learned per-anchor gate (sigmoid(logit),
        # allocated once at update_until like use_offset_mask, since its shape [N, 1] must
        # track the final frozen anchor count) multiplies the anchor's raw flow output
        # before the motion displacement is computed. motion_gate_weight regularizes the
        # gate toward 0 (no motion) unless the photometric loss needs it on, so truly
        # static anchors can learn to fully suppress their own motion computation's
        # contribution rather than relying on velocity/basis coefficients converging to
        # exactly zero on their own. motion_gate_weight > 0 is both the switch (whether
        # the gate is allocated at update_until) and its regularization strength, same
        # convention as motion_sigma_reg_weight/flow_densify_threshold.
        self.motion_gate_weight = 0.0

        # entropy coding (HAC-style hash-grid context model)
        self.lmbda = 0.001
        # Iterations over which lmbda ramps linearly from 0 to its target value after
        # entropy_start_iter, instead of switching on at full strength immediately. 0 (default)
        # keeps the old abrupt on/off behavior. See train.py's rate-loss comment for why this
        # exists.
        self.lmbda_ramp_iters = 0
        self.encoding_xyz_lr_init = 0.005
        self.encoding_xyz_lr_final = 0.00001
        self.encoding_xyz_lr_delay_mult = 0.33
        self.encoding_xyz_lr_max_steps = 30_000

        self.mlp_grid_lr_init = 0.005
        self.mlp_grid_lr_final = 0.00001
        self.mlp_grid_lr_delay_mult = 0.01
        self.mlp_grid_lr_max_steps = 30_000

        self.mask_lr = 0.01
        self.mask_prune_threshold = 0.01

        # Offset-level (sub-anchor) rate-distortion mask, gating opacity/bit_offsets per
        # individual offset slot instead of per whole anchor (see get_mask_offset in
        # gaussian_model.py). Only takes effect once anchor growing has finished
        # (iteration == update_until in train.py), so its shape can track the final,
        # frozen anchor count without needing to participate in the grow/prune lifecycle.
        # Reuses mask_lr for its own optimizer param group.
        self.use_offset_mask = False

        # Online best-checkpoint tracking: every best_ckpt_check_interval iterations (once
        # past entropy_start_iter), evaluate the test-set RD loss and, if it improves on the
        # best seen so far, save+encode the current weights over a single "best" checkpoint
        # slot (not one per check -- that would blow up disk and encode-time cost). This is
        # a strictly finer-grained version of the --save_iterations-based end-of-training
        # checkpoint selection (see train.py): instead of only comparing a handful of fixed
        # checkpoints, it continuously tracks the true minimum. 0 disables it (falls back to
        # --save_iterations-only selection).
        self.best_ckpt_check_interval = 2000

        # Some experimental options
        self.lambda_sigma_multiply = 1.0
        self.opacity_accum_method = "sum"
        self.da_start_iter = 0
        self.const_denom_weight = -1.0
        self.disable_partial_densify = False
        self.weight_base = 0.0
        self.densify_t = ''

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
