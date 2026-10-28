#
# Standalone diagnostic script: decode an ALREADY-SAVED bitstream (from any
# --save_iterations checkpoint, not just the final one) and render+save the
# actually-decoded (compressed) model's test views, mirroring exactly what
# train.py's end-of-training block does for the final iteration only. This
# lets us compare the true compressed-model quality at an earlier checkpoint
# against the final one, without retraining -- used to check whether an
# earlier checkpoint gives similar size with better generalization (see the
# "lambda=0.01->0.03 cliff" discussion: late-training overfitting).
#
# Usage mirrors render.py:
#   python decode_eval_checkpoint.py -m <model_path> --iteration <ckpt_iter> <same dataset flags as training>
#
# After this, run metrics.py -m <model_path> to compute SSIM/PSNR/LPIPS/ALEX
# for the new "ours_<iteration>_compressed" folder (and any other folders
# already under <model_path>/test/).
#

import os
import torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from gaussian_renderer import GaussianModel
from render import render_set
from utils.general_utils import safe_state

if __name__ == "__main__":
    parser = ArgumentParser(description="Decode an existing bitstream checkpoint and render its test views")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", required=True, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    iteration = args.iteration

    print("Decoding + rendering " + dataset.model_path + f" @ iteration {iteration}")
    safe_state(args.quiet)

    bitstream_path = os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "bitstreams")
    assert os.path.exists(os.path.join(bitstream_path, "uncoded_attrs.pt")), \
        f"No bitstream found at {bitstream_path} -- was conduct_encoding run for this iteration (i.e. was it in --save_iterations)?"

    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        gaussians.conduct_decoding(bitstream_path)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "test", f"{iteration}_compressed", scene.getTestCameras(), gaussians, pipe, background, skip_images=False, skip_fps=True)
        print(f"\n[ITER {iteration}] Rendered decoded (compressed) model outputs to test/ours_{iteration}_compressed/")
