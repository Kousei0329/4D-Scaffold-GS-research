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

import os
import sys
import numpy as np

# Must happen before torch/torchvision/wandb are imported below -- CUDA_VISIBLE_DEVICES
# is only read once the CUDA runtime first initializes, which these imports can trigger
# implicitly (e.g. via extension modules probing device count). Setting it again later,
# after args.gpu is parsed (see the `if args.gpu != '-1'` block further down), can
# silently no-op if that happens. Checking sys.argv directly here -- before argparse has
# even run -- is what makes an explicit --gpu reliably win over the auto-pick below.
# Without this, two processes launched close together each auto-pick "whichever GPU
# currently looks least busy" independently; if both snapshot nvidia-smi before either
# has allocated anything, they can BOTH pick the same physical GPU despite being given
# different --gpu values, and then collide on it (confirmed: this was the actual cause
# of several "random GPU1 instability" crashes diagnosed during development, not faulty
# hardware -- the two parallel jobs were silently sharing one physical GPU).
_explicit_gpu = None
if '--gpu' in sys.argv:
    _explicit_gpu = sys.argv[sys.argv.index('--gpu') + 1]

if _explicit_gpu is not None and _explicit_gpu != '-1':
    os.environ['CUDA_VISIBLE_DEVICES'] = _explicit_gpu
else:
    import subprocess
    cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
    os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')


import torch
import torchvision
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, render_cached, network_gui
import sys
from scene import Scene, GaussianModel
from scene.gaussian_model import MASK_INIT_LOGIT
import torch.nn as nn
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.entropy_models import get_binary_vxl_size
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import DataLoader
import cv2

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')
alex_fn = lpips.LPIPS(net='alex').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore', 'submodules', 'assets']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


bit2MB_scale = 8 * 1024 * 1024   # for converting a BIT count (e.g. tensor.numel()*32) to MB
byte2MB_scale = 1024 * 1024      # for converting a BYTE count (e.g. os.path.getsize()) to MB


def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, num_workers, debug_from, wandb=None, logger=None, ply_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight, dataset.disable_denom_weight, dataset.hparam_beta, dataset.max_init_t,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                              ablation_no_hash_context=dataset.ablation_no_hash_context,
                              ablation_fixed_quant=dataset.ablation_fixed_quant,
                              ablation_no_time_agg_mask=dataset.ablation_no_time_agg_mask,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # restore() (if any) can replace _anchor wholesale, so compute the filter after it,
    # not before -- it must match whatever anchor set training actually starts from.
    if not dataset.disable_mip_filter:
        gaussians.compute_3D_filter(cameras=scene.train_cameras[1.0])

    final_size_report = None
    compressed_visible_count = None
    # Per-iteration test-set RD loss (distortion + rate, mirroring the training loss) and
    # per-iteration compressed-size report, both keyed by iteration -- used at the end of
    # training to ship whichever --save_iterations checkpoint generalizes best instead of
    # always the literal final iteration (see the end-of-training block below). A long
    # post-densification fine-tune phase under a strong rate penalty can overfit to the
    # training views in its final stretch (train loss keeps falling while test loss rises);
    # shipping that checkpoint trades quality for nothing, since an earlier, better-
    # generalizing checkpoint of the SAME run compresses to about the same size (the rate
    # term barely differs between two checkpoints of one lmbda run -- size differences come
    # mostly from anchor_num, which is fairly stable by the later checkpoints).
    test_rd_loss_by_iter = {}
    size_report_by_iter = {}
    # Online best-checkpoint tracking (opt.best_ckpt_check_interval): a strictly finer-
    # grained version of the above -- checked every best_ckpt_check_interval iterations
    # (not just at the handful of --save_iterations) so the true minimum of the test RD
    # loss curve is found, not just the best among 4 widely-spaced samples. Each time it
    # improves, the current weights are saved+encoded over this one fixed "best" slot
    # (never one slot per check -- that would be --save_iterations all over again, just
    # finer and far more expensive). See the end-of-training block for how this is
    # reconciled with the --save_iterations candidates.
    best_online_score = None
    best_online_iter = None
    best_online_path = None
    best_online_size_report = None

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    torch.manual_seed(42)
    training_dataset = scene.getTrainCameras()
    training_dataloader = DataLoader(training_dataset, batch_size=1, shuffle=True, num_workers=num_workers if dataset.dataloader else 0, collate_fn=lambda x: x, drop_last=True)
    test_dataset = scene.getTestCameras()
    test_gt, test_view = test_dataset[pipe.vis_idx]
    test_gt = test_gt.cuda()
    test_view = test_view.cuda()

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    while iteration < opt.iterations + 1:
        for batch_data in training_dataloader:
            iteration += 1
            if iteration > opt.iterations:
                break  

            # network gui not available in scaffold-gs yet
            # if network_gui.conn == None:
            #     network_gui.try_connect()
            # while network_gui.conn != None:
            #     try:
            #         net_image_bytes = None
            #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
            #         if custom_cam != None:
            #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
            #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
            #         network_gui.send(net_image_bytes, dataset.source_path)
            #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
            #             break
            #     except Exception as e:
            #         network_gui.conn = None

            iter_start.record()

            gaussians.update_learning_rate(iteration)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            viewpoint_image, viewpoint_cam = batch_data[0]
            viewpoint_image = viewpoint_image.cuda()
            viewpoint_cam = viewpoint_cam.cuda()
            
            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True
            
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
            retain_grad = (iteration < opt.update_until and iteration >= 0)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)

            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity, opacity_t, sigma, bit_per_param, neural_xyz, flow = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["opacity_t"], render_pkg["sigma"], render_pkg["bit_per_param"], render_pkg["xyz"], render_pkg["flow"]

            gt_image = viewpoint_image
            Ll1 = l1_loss(image, gt_image)
            # Per-pixel error map for the photometric-error densification signal (see
            # training_statis/anchor_growing) -- detached, diagnostic only, not part of
            # the backprop graph.
            pixel_error_map = (image.detach() - gt_image).abs().mean(dim=0, keepdim=True)

            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg

            # flow is None whenever use_flow=False; first 3 channels are a literal
            # velocity only for 'linear'/'taylor_residual' motion models -- shared by both
            # motion_sigma_reg_weight (below) and flow_densify_threshold (passed into
            # training_statis further down), so computed once here regardless of which
            # (if either) is actually enabled.
            flow_speed = None
            if flow is not None and gaussians.motion_model in ('linear', 'taylor_residual'):
                flow_speed = flow[:, :3].norm(dim=-1, keepdim=True).detach()

            # Motion-magnitude temporal-sharpness regularization (see
            # opt.motion_sigma_reg_weight's docstring in arguments/__init__.py).
            if opt.motion_sigma_reg_weight > 0 and flow_speed is not None:
                # sigma = ELU(x)+1 is unbounded above, and this loss's gradient always
                # pushes sigma UP wherever flow_speed>0 -- nothing in it saturates, so
                # with no strong enough opposing photometric gradient (e.g. right after
                # densification creates fresh anchors with poorly-calibrated features),
                # Adam can drive sigma to run away over thousands of steps. Once sigma
                # overflows, opacity_t = exp(-|dt*sigma|^beta) can hit an exact 0 * inf
                # for any anchor whose dt is exactly 0 (its own designated timestamp) =
                # NaN. Confirmed via a real crash (NaN starting iteration 1610, right
                # after update_from=1500's first densification). Clamping sigma here
                # removes the gradient once it's already far past "fully narrow" (exp(-20)
                # ~= 2e-9), so the loss can't keep pushing it toward overflow.
                sigma_reg_loss = (flow_speed * torch.exp(-sigma.clamp(max=20.0))).mean()
                loss = loss + opt.motion_sigma_reg_weight * sigma_reg_loss

            # E-D3DGS-style local motion-smoothness regularization: encourages spatially
            # close Gaussians to carry similar learned motion (flow), on top of whatever
            # consistency the photometric loss alone induces. Computed on a random
            # subsample (motion_smooth_sample Gaussians) since a full NxN distance matrix
            # over every visible Gaussian this iteration would be too large; nearest
            # neighbor within just that subsample is a cheap, unbiased-in-expectation
            # stand-in for a true KNN graph. Position is detached (this loss should pull
            # flow fields together, not move the Gaussians themselves).
            if opt.motion_smooth_weight > 0 and flow is not None:
                n_total = neural_xyz.shape[0]
                sample_n = min(opt.motion_smooth_sample, n_total)
                idx = torch.randperm(n_total, device=flow.device)[:sample_n]
                pos_sub = neural_xyz[idx].detach()
                flow_sub = flow[idx]
                with torch.no_grad():
                    dist = torch.cdist(pos_sub, pos_sub)
                    dist.fill_diagonal_(float('inf'))
                    nn_idx = dist.argmin(dim=1)
                motion_smooth_loss = ((flow_sub - flow_sub[nn_idx]) ** 2).mean()
                loss = loss + opt.motion_smooth_weight * motion_smooth_loss

            if opt.motion_gate_weight > 0 and gaussians.use_motion_gate and gaussians._motion_gate.numel() > 0:
                # Pushes the gate toward 0 (no motion) everywhere the photometric loss
                # doesn't need it open, mirroring how the rate loss pushes mask_anchor
                # toward 0 for anchors that aren't worth keeping.
                loss = loss + opt.motion_gate_weight * gaussians.get_motion_gate.mean()

            if bit_per_param is not None:
                # HAC++-style hash-grid rate loss: pushes encoding_xyz's STE-binarized +-1
                # weights to skew toward one value, which is what lets save_hash_grid's
                # Bernoulli arithmetic coder beat a flat 1 bit/weight. Normalized by the same
                # per-anchor element count bit_per_param already averages over, so the two
                # terms are on the same "bits per coded element" scale before lmbda scales them.
                denom = gaussians.get_anchor.shape[0] * (gaussians.feat_dim + 8 + 4 * gaussians.n_offsets)
                _, bit_hash_grid, _, _ = get_binary_vxl_size((gaussians.get_encoding_params() + 1) / 2)
                # The rate loss (and, through it, the per-anchor RD mask's gradient -- mask
                # multiplies every bit_* term) otherwise switches on at full lmbda strength
                # the instant iteration crosses entropy_start_iter, while densification is
                # still running (update_until). That abrupt, lambda-sized kick decides which
                # marginal anchors the mask starts pruning at a moment when the network hasn't
                # adapted to the rate term at all yet -- a threshold effect that can make the
                # final RD point depend on lmbda non-monotonically (e.g. a slightly larger
                # lmbda landing in a better basin than a smaller one). lmbda_ramp_iters ramps
                # the effective lmbda linearly from 0 to opt.lmbda over that many iterations
                # after entropy_start_iter instead, so mask/rate decisions settle in gradually.
                # Default 0 reproduces the old abrupt on/off behavior exactly.
                if opt.lmbda_ramp_iters > 0:
                    ramp = min(1.0, max(0.0, (iteration - dataset.entropy_start_iter) / opt.lmbda_ramp_iters))
                else:
                    ramp = 1.0
                loss = loss + (opt.lmbda * ramp) * (bit_per_param + bit_hash_grid / denom)

            loss.backward()
            
            iter_end.record()

            with torch.no_grad():
                if pipe.vis:
                    if iteration % pipe.vis_interval == 0:
                        gaussians.eval()

                        # TEST VIEW
                        vvmask = prefilter_voxel(test_view, gaussians, pipe,background)
                        render_pkg = render(test_view, gaussians, pipe, background, visible_mask=vvmask, retain_grad=False)
                        img_test = render_pkg["render"]

                        grid = torchvision.utils.make_grid([gt_image, image,
                                                            test_gt, img_test], nrow=2)
                        torchvision.utils.save_image(grid, os.path.join(dataset.model_path, "test_view.png"))

                        gaussians.train()

                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "N": f"{len(scene.gaussians.get_anchor)}"})
                    progress_bar.update(10)
                if iteration == opt.iterations:
                    progress_bar.close()

                if dataset.use_entropy_coding and iteration % 1000 == 0 and iteration > dataset.entropy_start_iter:
                    bits_info = gaussians.estimate_bits(mask_prune_threshold=opt.mask_prune_threshold)
                    logger.info("\n-----[ITER {}] bits info: anchor(placeholder) {:.4f}MB, feat {:.4f}MB, scaling {:.4f}MB, offsets {:.4f}MB, anchor_num {} (kept) / {} (total)-----".format(
                        iteration, bits_info['anchor_MB'], bits_info['feat_MB'], bits_info['scaling_MB'], bits_info['offsets_MB'],
                        bits_info['anchor_num'], bits_info['anchor_num_total']))

                # Online best-checkpoint tracking: see opt.best_ckpt_check_interval's
                # definition for why this exists (a finer-grained alternative to only
                # comparing --save_iterations at the end of training). Checked AFTER
                # densification/optimizer-step-affecting state has settled for this
                # iteration, same as the bits-info block above.
                if (dataset.use_entropy_coding and opt.best_ckpt_check_interval > 0
                        and iteration % opt.best_ckpt_check_interval == 0
                        and iteration > dataset.entropy_start_iter):
                    gaussians.eval()
                    torch.cuda.empty_cache()
                    score = evaluate_test_rd_loss(scene, render, (pipe, background), opt, dataset, iteration, logger)
                    gaussians.train()
                    if score is not None and (best_online_score is None or score < best_online_score):
                        best_online_score = score
                        best_online_iter = iteration
                        best_online_path = os.path.join(scene.model_path, "point_cloud", "best_online")
                        gaussians.save_ply(os.path.join(best_online_path, "point_cloud.ply"))
                        gaussians.save_mlp_checkpoints(best_online_path)
                        bitstream_path = os.path.join(best_online_path, "bitstreams")
                        enc_report = gaussians.conduct_encoding(bitstream_path, mask_prune_threshold=opt.mask_prune_threshold)
                        hash_grid_MB = os.path.getsize(os.path.join(best_online_path, 'encoding_xyz.bin')) / byte2MB_scale
                        mlp_files = ['grid_mlp.pt', 'opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt', 'flow_mlp.pt']
                        mlp_MB = sum(os.path.getsize(os.path.join(best_online_path, f)) for f in mlp_files
                                     if os.path.exists(os.path.join(best_online_path, f))) / byte2MB_scale
                        total_MB = (enc_report['feat_MB'] + enc_report['scaling_MB'] + enc_report['offsets_MB'] +
                                    enc_report['anchor_geom_MB'] + enc_report['mask_MB'] + hash_grid_MB + mlp_MB)
                        best_online_size_report = {**enc_report, 'hash_grid_MB': hash_grid_MB, 'mlp_MB': mlp_MB,
                                                    'total_MB': total_MB, 'iteration': iteration}
                        logger.info("\n[ITER {}] New best online test RD loss {:.6f} (TOTAL {:.4f}MB, anchor_num {}) -- "
                                    "saved to point_cloud/best_online/".format(iteration, score, total_MB, enc_report['anchor_num']))

                # Log and save
                test_rd_score = training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), opt, dataset, wandb, logger)
                if test_rd_score is not None:
                    test_rd_loss_by_iter[iteration] = test_rd_score
                if (iteration in saving_iterations):
                    logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                    scene.save(iteration)

                    if dataset.use_entropy_coding:
                        point_cloud_path = os.path.join(scene.model_path, "point_cloud", "iteration_{}".format(iteration))
                        bitstream_path = os.path.join(point_cloud_path, "bitstreams")
                        enc_report = gaussians.conduct_encoding(bitstream_path, mask_prune_threshold=opt.mask_prune_threshold)

                        hash_grid_MB = os.path.getsize(os.path.join(point_cloud_path, 'encoding_xyz.bin')) / byte2MB_scale
                        mlp_files = ['grid_mlp.pt', 'opacity_mlp.pt', 'cov_mlp.pt', 'color_mlp.pt', 'flow_mlp.pt']
                        mlp_MB = sum(os.path.getsize(os.path.join(point_cloud_path, f)) for f in mlp_files
                                     if os.path.exists(os.path.join(point_cloud_path, f))) / byte2MB_scale
                        # rotation/opacity are excluded here: conduct_encoding no longer stores them
                        # per-anchor (they're frozen constants, stored once and broadcast on decode),
                        # so they no longer contribute meaningfully to size.
                        total_MB = (enc_report['feat_MB'] + enc_report['scaling_MB'] + enc_report['offsets_MB'] +
                                    enc_report['anchor_geom_MB'] + enc_report['mask_MB'] + hash_grid_MB + mlp_MB)

                        logger.info("\n-----[ITER {}] REAL bitstream size: feat {:.4f}MB, scaling {:.4f}MB, offsets {:.4f}MB, "
                                    "anchor(raw,uncoded) {:.4f}MB, mask {:.4f}MB, hash_grid {:.4f}MB, mlps {:.4f}MB, TOTAL {:.4f}MB, "
                                    "anchor_num {}-----".format(
                                        iteration, enc_report['feat_MB'], enc_report['scaling_MB'], enc_report['offsets_MB'],
                                        enc_report['anchor_geom_MB'], enc_report['mask_MB'], hash_grid_MB, mlp_MB, total_MB, enc_report['anchor_num']))

                        size_report_by_iter[iteration] = {**enc_report, 'hash_grid_MB': hash_grid_MB, 'mlp_MB': mlp_MB, 'total_MB': total_MB, 'iteration': iteration}

                # densification
                if iteration < opt.update_until and iteration > opt.start_stat:
                    # add statis
                    activate_DA = (iteration >= opt.da_start_iter)
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, opacity_t, sigma, opt.lambda_temporal_sigma, viewpoint_cam.timestamp, opt, activate_DA, neural_xyz, pixel_error_map, viewpoint_cam, flow_speed)
                    
                    # densification
                    if iteration > opt.update_from and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(opt, check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
                        if not dataset.disable_mip_filter:
                            gaussians.compute_3D_filter(cameras=scene.train_cameras[1.0])
                elif iteration == opt.update_until:
                    del gaussians.opacity_accum
                    del gaussians.offset_gradient_accum
                    del gaussians.photo_error_accum
                    del gaussians.offset_time_accum
                    del gaussians.offset_denom
                    torch.cuda.empty_cache()

                    # Allocate the offset-level mask now that anchor growing/pruning is
                    # done (N is frozen for the rest of training), so its shape [N,
                    # n_offsets] never needs to survive a grow/prune event. Added as its
                    # own optimizer param group rather than through training_setup,
                    # which already ran once at the start of training.
                    if opt.use_offset_mask:
                        gaussians.use_offset_mask = True
                        N_frozen = gaussians.get_anchor.shape[0]
                        mask_offset_init = torch.full(
                            (N_frozen, gaussians.n_offsets), MASK_INIT_LOGIT, device="cuda")
                        gaussians._mask_offset = nn.Parameter(mask_offset_init.requires_grad_(True))
                        gaussians.optimizer.add_param_group(
                            {'params': [gaussians._mask_offset], 'lr': opt.mask_lr, 'name': 'mask_offset'})

                    # Same allocation pattern for the Ex4DGS-inspired per-anchor motion
                    # gate (see motion_gate_weight's docstring in arguments/__init__.py).
                    if opt.motion_gate_weight > 0:
                        gaussians.use_motion_gate = True
                        N_frozen = gaussians.get_anchor.shape[0]
                        motion_gate_init = torch.full((N_frozen, 1), MASK_INIT_LOGIT, device="cuda")
                        gaussians._motion_gate = nn.Parameter(motion_gate_init.requires_grad_(True))
                        gaussians.optimizer.add_param_group(
                            {'params': [gaussians._motion_gate], 'lr': opt.mask_lr, 'name': 'motion_gate'})

                # Optimizer step
                if iteration < opt.iterations:
                    if gaussians.motion_model != 'linear' or opt.force_grad_clip:
                        # The 'basis'/'fourier'/'taylor_residual' motion models combine two
                        # learned factors multiplicatively (coeff(feat) * Phi(dt)) -- a
                        # bilinear parameterization that's well known to be prone to
                        # occasional gradient blow-up (one factor growing feeds back into the
                        # other's gradient) in a way the original purely-linear flow*dt model
                        # never exercised. Global grad-norm clipping is the standard fix;
                        # gated off for 'linear' to leave that already-validated path's
                        # training dynamics completely unchanged.
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in gaussians.optimizer.param_groups for p in group['params']],
                            max_norm=1.0)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                if (iteration in checkpoint_iterations):
                    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

                if iteration == opt.iterations and dataset.use_entropy_coding:
                    # Pick the checkpoint to actually ship: the one with the lowest test-set RD
                    # loss among iterations that both got a test RD-loss measurement (i.e. were in
                    # --test_iterations, past entropy_start_iter) and have an encoded bitstream on
                    # disk (i.e. were in --save_iterations) -- not necessarily the literal final
                    # iteration. If nothing qualifies (e.g. entropy coding never actually turned on
                    # during this run), best_iter falls back to the final iteration, reproducing
                    # the old unconditional behavior exactly.
                    candidates = {it: v for it, v in test_rd_loss_by_iter.items()
                                  if os.path.exists(os.path.join(scene.model_path, "point_cloud",
                                                                  "iteration_{}".format(it), "bitstreams", "uncoded_attrs.pt"))}
                    best_iter = min(candidates, key=candidates.get) if candidates else iteration
                    best_score_from_saves = candidates.get(best_iter)
                    if candidates:
                        logger.info("\n[ITER {}] test RD loss by checkpoint: {} -- best: iteration {}".format(
                            iteration, {k: round(v, 6) for k, v in sorted(candidates.items())}, best_iter))

                    # The online tracker (opt.best_ckpt_check_interval, finer-grained than
                    # --save_iterations) may have found a better checkpoint than any of the
                    # candidates above -- if so it wins, regardless of whether its iteration
                    # number happens to coincide with one of them.
                    use_online_best = (best_online_score is not None
                                        and (best_score_from_saves is None or best_online_score < best_score_from_saves))

                    if use_online_best:
                        # Reload directly from point_cloud/best_online/ -- no need for a fresh
                        # Scene() (expensive: re-reads the whole dataset) since test_dataset's
                        # cameras don't depend on which checkpoint's weights are loaded.
                        logger.info("\n[ITER {}] Best test RD loss was the online-tracked checkpoint at iteration {} "
                                     "(score {:.6f} vs {} from --save_iterations) -- reloading and shipping that "
                                     "instead.".format(iteration, best_online_iter, best_online_score,
                                                        "{:.6f}".format(best_score_from_saves) if best_score_from_saves is not None else "n/a"))
                        best_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight, dataset.disable_denom_weight, dataset.hparam_beta, dataset.max_init_t,
                                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                                              ablation_no_hash_context=dataset.ablation_no_hash_context,
                                              ablation_fixed_quant=dataset.ablation_fixed_quant,
                                              ablation_no_time_agg_mask=dataset.ablation_no_time_agg_mask,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
                        best_gaussians.load_ply_sparse_gaussian(os.path.join(best_online_path, "point_cloud.ply"))
                        best_gaussians.load_mlp_checkpoints(best_online_path)
                        if not dataset.disable_mip_filter:
                            best_gaussians.compute_3D_filter(cameras=scene.train_cameras[1.0])
                        if dataset.appearance_dim > 0:
                            # This reload path intentionally skips Scene() (see comment
                            # above) and with it set_appearance()'s rebuild of
                            # appearance_id_map -- copy it from the live, already-trained
                            # `gaussians` instead (same train camera set, so identical map).
                            best_gaussians.appearance_id_map = gaussians.appearance_id_map
                        best_gaussians.eval()
                        best_gaussians.conduct_decoding(os.path.join(best_online_path, "bitstreams"))
                        best_gaussians.eval()
                        label = "{}_online_compressed".format(best_online_iter)
                        _, compressed_visible_count = render_set(scene.model_path, "test", label,
                                                                   test_dataset, best_gaussians, pipe, background)
                        logger.info("\n[ITER {}] Rendered decoded (compressed) model outputs to test/ours_{}/ "
                                    "(online-tracked best, iteration {}; see the SSIM/PSNR/LPIPS/ALEX report for "
                                    "that folder below)".format(iteration, label, best_online_iter))
                        del best_gaussians
                        torch.cuda.empty_cache()
                        final_size_report = best_online_size_report
                    elif best_iter == iteration:
                        # The final iteration already is (or ties for) the best -- decode the live,
                        # already-in-memory model in place, exactly as before. This is truly the
                        # last thing that happens to `gaussians` this run (after densification and
                        # the optimizer step), so it's safe to overwrite its anchor tensors with the
                        # actually-decoded ones (conduct_decoding mutates them in place) to verify
                        # the bitstream round-trips to a working model. Doing this any earlier in
                        # the same iteration -- e.g. right after conduct_encoding, before
                        # densification -- leaves offset_gradient_accum/offset_denom sized for the
                        # pre-decode anchor count while `gaussians` now holds the (generally
                        # smaller, mask-filtered) decoded anchor count, which crashes adjust_anchor.
                        bitstream_path = os.path.join(scene.model_path, "point_cloud", "iteration_{}".format(iteration), "bitstreams")
                        if os.path.exists(os.path.join(bitstream_path, 'uncoded_attrs.pt')):
                            gaussians.conduct_decoding(bitstream_path)
                            if not dataset.disable_mip_filter:
                                gaussians.compute_3D_filter(cameras=scene.train_cameras[1.0])
                            gaussians.eval()
                            # Render the actually-decoded (compressed) model's test-set outputs to
                            # "ours_{iteration}_compressed" -- these are the images that would actually
                            # ship, so this is the only test-set render/eval this run needs. The caller
                            # (below __main__) skips the separate uncompressed render_sets() pass and
                            # reuses compressed_visible_count when this ran, instead of re-rendering the
                            # pre-decode model into a redundant "ours_{iteration}" folder.
                            _, compressed_visible_count = render_set(scene.model_path, "test", "{}_compressed".format(iteration),
                                                                       test_dataset, gaussians, pipe, background)
                            logger.info("\n[ITER {}] Rendered decoded (compressed) model outputs to "
                                        "test/ours_{}_compressed/ (best test RD loss was also the final "
                                        "iteration; see the SSIM/PSNR/LPIPS/ALEX report for that folder "
                                        "below)".format(iteration, iteration))
                        else:
                            logger.info("\n[ITER {}] Skipping decode verification: {} was final iteration but not in "
                                         "--save_iterations, so no bitstream was encoded to decode.".format(iteration, iteration))
                    else:
                        # A strictly earlier checkpoint generalizes better at essentially the same
                        # size (same lmbda run -- the rate term barely differs between its
                        # checkpoints) -- reload it fresh (the live `gaussians` has since trained
                        # past it) and ship that one instead, mirroring render_sets()'s reload
                        # pattern. final_size_report below is keyed off best_iter, not `iteration`,
                        # so FINAL SUMMARY honestly reports which checkpoint actually shipped.
                        logger.info("\n[ITER {}] Best test RD loss was at iteration {}, not the final iteration "
                                     "-- reloading and shipping that checkpoint instead.".format(iteration, best_iter))
                        best_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight, dataset.disable_denom_weight, dataset.hparam_beta, dataset.max_init_t,
                                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                                              ablation_no_hash_context=dataset.ablation_no_hash_context,
                                              ablation_fixed_quant=dataset.ablation_fixed_quant,
                                              ablation_no_time_agg_mask=dataset.ablation_no_time_agg_mask,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
                        best_scene = Scene(dataset, best_gaussians, load_iteration=best_iter, shuffle=False)
                        if not dataset.disable_mip_filter:
                            best_gaussians.compute_3D_filter(cameras=best_scene.train_cameras[1.0])
                        best_gaussians.eval()
                        bitstream_path = os.path.join(scene.model_path, "point_cloud", "iteration_{}".format(best_iter), "bitstreams")
                        best_gaussians.conduct_decoding(bitstream_path)
                        best_gaussians.eval()
                        _, compressed_visible_count = render_set(scene.model_path, "test", "{}_compressed".format(best_iter),
                                                                   best_scene.getTestCameras(), best_gaussians, pipe, background)
                        logger.info("\n[ITER {}] Rendered decoded (compressed) model outputs to "
                                    "test/ours_{}_compressed/ (selected over the final iteration {} by test "
                                    "RD loss; see the SSIM/PSNR/LPIPS/ALEX report for that folder "
                                    "below)".format(best_iter, best_iter, iteration))
                        del best_gaussians, best_scene
                        torch.cuda.empty_cache()

                    if not use_online_best:
                        final_size_report = size_report_by_iter.get(best_iter)

    return {'size_report': final_size_report, 'compressed_visible_count': compressed_visible_count}

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def evaluate_test_rd_loss(scene : Scene, renderFunc, renderArgs, opt, dataset, iteration, logger=None):
    """Test-set RD loss (distortion + rate, the same formula as the training loss) --
    evaluated over the FULL test-camera set. Caller must have already put scene.gaussians in
    eval() mode. Returns None if dataset.use_entropy_coding is False.

    Used both by training_report (at the coarse --test_iterations checkpoints) and by the
    online best-checkpoint tracker (at the finer best_ckpt_check_interval granularity) --
    factored out once so both stay consistent. Cost: one full render of every test camera
    (300 for N3DV) plus an estimate_bits() call -- not cheap, so callers should not invoke
    this every training iteration, only at a deliberately chosen interval."""
    if not dataset.use_entropy_coding:
        return None
    l1_test = 0.0
    ssim_test = 0.0
    test_cameras = scene.getTestCameras()
    for viewimg, viewpoint in test_cameras:
        viewimg = viewimg.cuda()
        viewpoint = viewpoint.cuda()
        voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewimg, 0.0, 1.0)
        l1_test += l1_loss(image, gt_image).mean().double()
        ssim_test += ssim(image, gt_image).mean().double()
    n = len(test_cameras)
    l1_test /= n
    ssim_test /= n
    # NOTE: render_pkg["bit_per_param"] is NOT usable here -- render() only computes it when
    # pc.get_color_mlp.training is True (see generate_neural_gaussians's `if is_training and
    # pc.use_entropy_coding` branch), and the caller has gaussians in eval() mode, so every
    # render_pkg here has bit_per_param=None. Use estimate_bits() instead -- the same
    # view-independent, no_grad whole-model bit estimate already used for the periodic
    # "bits info" log line during training, which works correctly in either train()/eval().
    distortion = (1.0 - opt.lambda_dssim) * l1_test + opt.lambda_dssim * (1.0 - ssim_test)
    bits_info = scene.gaussians.estimate_bits(mask_prune_threshold=opt.mask_prune_threshold)
    rate_mb = bits_info['feat_MB'] + bits_info['scaling_MB'] + bits_info['offsets_MB']
    if opt.lmbda_ramp_iters > 0:
        ramp = min(1.0, max(0.0, (iteration - dataset.entropy_start_iter) / opt.lmbda_ramp_iters))
    else:
        ramp = 1.0
    rate_term = (opt.lmbda * ramp) * rate_mb
    test_rd_score = (distortion + rate_term).item()
    if logger is not None:
        logger.info("\n[ITER {}] Test RD loss (distortion {:.6f} + rate {:.6f} MB*lmbda = {:.6f})".format(
            iteration, distortion.item(), rate_term, test_rd_score))
    return test_rd_score


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, opt, dataset, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, 'num_points':scene.gaussians.get_anchor.shape[0]})

    # Test-set RD loss (distortion + rate, same formula as the training loss) for the
    # 'test' config only -- returned to the caller so it can track which --save_iterations
    # checkpoint actually generalizes best, for end-of-training checkpoint selection. None
    # unless this is a testing iteration with entropy coding enabled.
    test_rd_score = None

    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                track_rd = (config['name'] == 'test' and dataset.use_entropy_coding)

                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, (viewimg, viewpoint) in enumerate(config['cameras']):
                    viewimg = viewimg.cuda()
                    viewpoint = viewpoint.cuda()

                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewimg, 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    if track_rd:
                        ssim_test += ssim(image, gt_image).mean().double()



                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if track_rd:
                    # NOTE: render_pkg["bit_per_param"] is NOT usable here -- render() only
                    # computes it when pc.get_color_mlp.training is True (see
                    # generate_neural_gaussians's `if is_training and pc.use_entropy_coding`
                    # branch), and scene.gaussians.eval() was called at the top of this
                    # function, so every render_pkg in this loop has bit_per_param=None. Use
                    # estimate_bits() instead -- the same view-independent, no_grad whole-model
                    # bit estimate already used for the periodic "bits info" log line during
                    # training, which works correctly regardless of train()/eval() mode.
                    ssim_test /= len(config['cameras'])
                    distortion = (1.0 - opt.lambda_dssim) * l1_test + opt.lambda_dssim * (1.0 - ssim_test)
                    bits_info = scene.gaussians.estimate_bits(mask_prune_threshold=opt.mask_prune_threshold)
                    rate_mb = bits_info['feat_MB'] + bits_info['scaling_MB'] + bits_info['offsets_MB']
                    if opt.lmbda_ramp_iters > 0:
                        ramp = min(1.0, max(0.0, (iteration - dataset.entropy_start_iter) / opt.lmbda_ramp_iters))
                    else:
                        ramp = 1.0
                    rate_term = (opt.lmbda * ramp) * rate_mb
                    test_rd_score = (distortion + rate_term).item()
                    logger.info("\n[ITER {}] Test RD loss (distortion {:.6f} + rate {:.6f} MB*lmbda = {:.6f}, for "
                                "end-of-training checkpoint selection)".format(iteration, distortion.item(), rate_term, test_rd_score))

                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

    return test_rd_score

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    height, width = views[0][0].shape[1:3]
    cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, (viewimg, view) in enumerate(tqdm(views, desc="Rendering progress")):
        viewimg = viewimg.cuda()
        view = view.cuda()

        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        # render_cached reuses the anchor-level neural-Gaussian outputs across consecutive
        # frames that share a camera (the N3DV/Neu3D test-set protocol: one held-out
        # camera x many timestamps) instead of recomputing the 4 MLPs every frame; it
        # rebuilds automatically if the camera changes, so it's always correct here.
        render_pkg = render_cached(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)
        img = cv2.cvtColor((rendering.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cap.write(img)


        # gts
        gt = viewimg[0:3, :, :]

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    cap.release()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight, dataset.disable_denom_weight, dataset.hparam_beta,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter,
                              use_intra_anchor_context=dataset.use_intra_anchor_context,
                              use_joint_4d_hash=dataset.use_joint_4d_hash,
                              use_hybrid_4d_hash=dataset.use_hybrid_4d_hash,
                              use_full_hybrid_4d_hash=dataset.use_full_hybrid_4d_hash,
                              ablation_no_hash_context=dataset.ablation_no_hash_context,
                              ablation_fixed_quant=dataset.ablation_fixed_quant,
                              ablation_no_time_agg_mask=dataset.ablation_no_time_agg_mask,
                              motion_model=dataset.motion_model,
                              motion_basis_dim=dataset.motion_basis_dim)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if not dataset.disable_mip_filter:
            gaussians.compute_3D_filter(cameras=scene.train_cameras[1.0])
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []
        alexs = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
            alexs.append(alex_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })
            wandb.log({"test_ALEX":torch.stack(alexs).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        logger.info("  ALEX : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(alexs).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/ALEX', torch.tensor(alexs).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item(),
                                                "ALEX": torch.tensor(alexs).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "ALEX": {name: al for al, name in zip(torch.tensor(alexs).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--expname", type=str, default="")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        if (not args.debug):
            saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork", mode=os.environ.get("WANDB_MODE", "online")),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    train_result = training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.num_workers, args.debug_from, wandb, logger)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        train_result = training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.num_workers, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    compressed_visible_count = train_result.get('compressed_visible_count') if train_result else None
    if compressed_visible_count is not None:
        # The decoded (compressed) model's test-set images already got rendered/saved during
        # training (see the "..._compressed" render_set call in training()) -- those are the
        # images that would actually ship. Re-rendering the pre-decode model here would only
        # produce a redundant "ours_{iter}" folder (same renders/gt, doubling disk usage) that
        # nothing downstream needs.
        logger.info("\nSkipping uncompressed re-render: reusing the decoded (compressed) model's "
                    "test-set outputs from training.")
        visible_count = compressed_visible_count
    else:
        logger.info(f'\nStarting Rendering~')
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
        logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    # By the end of a full 120k-iteration run (especially with a large final anchor
    # count), PyTorch's caching allocator can be holding a lot of fragmented-but-unused
    # reserved memory from the training loop -- confirmed via a real, reproducible crash:
    # evaluate()'s LPIPS forward pass (both vgg and alex nets, loaded fresh here) failed
    # with "11.99 GiB reserved... 266 MiB free... tried to allocate 336 MiB", i.e. the
    # training loop's own leftover reservations, not genuinely insufficient device
    # memory. empty_cache() releases what's reserved-but-unused back to the driver so
    # LPIPS's extra allocations have room, at the one-time cost of some reallocation
    # overhead (irrelevant here since training is already finished).
    torch.cuda.empty_cache()
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")

    if train_result and train_result.get('size_report'):
        r = train_result['size_report']
        logger.info("\n=====[FINAL SUMMARY, iteration {}] compressed size: feat {:.4f}MB, scaling {:.4f}MB, "
                    "offsets {:.4f}MB, anchor(raw) {:.4f}MB, mask {:.4f}MB, hash_grid {:.4f}MB, mlps {:.4f}MB, TOTAL {:.4f}MB, "
                    "anchor_num {} -- see the 'ours_{}_compressed' (actually "
                    "decoded) SSIM/PSNR/LPIPS/ALEX block above for quality=====".format(
                        r['iteration'], r['feat_MB'], r['scaling_MB'], r['offsets_MB'],
                        r['anchor_geom_MB'], r['mask_MB'], r['hash_grid_MB'], r['mlp_MB'], r['total_MB'], r['anchor_num'],
                        r['iteration']))
