#
# Standalone verification: exercise the NEW reload mechanism added for the online
# best-checkpoint tracker (load_ply_sparse_gaussian + load_mlp_checkpoints directly,
# bypassing Scene(load_iteration=X)) against an already-saved point_cloud/best_online/
# checkpoint, and confirm it decodes+renders without error and matches the anchor count
# logged when it was saved during training.
#
import os
import sys
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel
from render import render_set
from utils.general_utils import safe_state

if __name__ == "__main__":
    parser = ArgumentParser()
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    safe_state(args.quiet)

    best_online_path = os.path.join(dataset.model_path, "point_cloud", "best_online")
    assert os.path.exists(os.path.join(best_online_path, "point_cloud.ply")), f"no checkpoint at {best_online_path}"
    assert os.path.exists(os.path.join(best_online_path, "bitstreams", "uncoded_attrs.pt")), "no bitstream"

    with torch.no_grad():
        # Need a Scene once, just to get test cameras (dataset reading) -- NOT for loading
        # weights (that's the mechanism under test: direct load, no Scene(load_iteration=)).
        placeholder = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
        scene = Scene(dataset, placeholder, load_iteration=6000, shuffle=False)  # just for cameras
        test_dataset = scene.getTestCameras()
        print(f"Loaded {len(test_dataset)} test cameras via Scene (for comparison/camera access only)")

        # ---- THE MECHANISM UNDER TEST ----
        best_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
        best_gaussians.load_ply_sparse_gaussian(os.path.join(best_online_path, "point_cloud.ply"))
        best_gaussians.load_mlp_checkpoints(best_online_path)
        if dataset.appearance_dim > 0:
            # This direct-load path (the mechanism under test) bypasses Scene()'s
            # set_appearance() call -- copy appearance_id_map from `placeholder`, which
            # built it from the same train camera set via the ordinary Scene() path above.
            best_gaussians.appearance_id_map = placeholder.appearance_id_map
        print(f"Direct-loaded point_cloud/best_online/: anchor count = {best_gaussians.get_anchor.shape[0]}")
        best_gaussians.eval()
        best_gaussians.conduct_decoding(os.path.join(best_online_path, "bitstreams"))
        best_gaussians.eval()
        print(f"After conduct_decoding: anchor count = {best_gaussians.get_anchor.shape[0]}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        render_set(dataset.model_path, "test", "best_online_verify", test_dataset, best_gaussians, pipe, background, skip_images=False, skip_fps=True)
        print("SUCCESS: direct reload + decode + render_set completed without error")
