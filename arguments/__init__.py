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
        self.time_duration = [0.0, 10.0]
        self.frame_ratio = 1.0
        self.max_init_t = 0.0
        self.init_voxel_scale = 1.0
        self.ply_path = ''

        self.hparam_beta = 4.0 # Beta in the paper

        # Ablation options
        self.temporal_opacity = '4dgs' # ['4dgs', 'ours']
        self.use_flow = False
        self.sigma_denom_weight = False
        self.disable_denom_weight = False
        self.filter_opacity = False

        # entropy coding (HAC-style hash-grid context model)
        self.use_entropy_coding = False
        self.hash_n_features = 2
        self.hash_log2_size = 19
        self.noise_start_iter = 3_000
        self.entropy_start_iter = 10_000
        
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

        self.lambda_temporal_sigma = 1.0 # Gamma in the paper

        # entropy coding (HAC-style hash-grid context model)
        self.lmbda = 0.001
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
