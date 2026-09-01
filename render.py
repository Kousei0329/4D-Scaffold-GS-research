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
import torch

import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, skip_images, skip_fps):
    if not skip_images:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
        video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders.mp4")
        if not os.path.exists(render_path):
            os.makedirs(render_path)
        if not os.path.exists(gts_path):
            os.makedirs(gts_path)

        height, width = views[0][0].shape[1:3]
        cap = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

        name_list = []
        per_view_dict = {}
        for idx, (viewpoint_image, viewpoint_cam) in enumerate(tqdm(views, desc="Rendering progress")):

            viewpoint_image = viewpoint_image.cuda()
            viewpoint_cam = viewpoint_cam.cuda()

            torch.cuda.synchronize(); t0 = time.time()
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipeline, background)
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize(); t1 = time.time()

            rendering = render_pkg["render"]
            gt = viewpoint_image[0:3, :, :]
            name_list.append('{0:05d}'.format(idx) + ".png")
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

            img = cv2.cvtColor((rendering.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cap.write(img)

        cap.release()

        with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    if not skip_fps:
        t_list = []
        for idx, (viewpoint_image, viewpoint_cam) in enumerate(tqdm(views, desc="Rendering progress")):

            viewpoint_image = viewpoint_image.cuda()
            viewpoint_cam = viewpoint_cam.cuda()

            torch.cuda.synchronize(); t0 = time.time()
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipeline, background)
            render_pkg = render(viewpoint_cam, gaussians, pipeline, background, visible_mask=voxel_visible_mask, use_cache=True)
            torch.cuda.synchronize(); t1 = time.time()
            
            t_list.append(t1-t0)

        t = np.array(t_list[5:])
        fps = 1.0 / t.mean()
        print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_images : bool, skip_fps : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.t_grid_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
                              dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.temporal_opacity, dataset.use_flow, dataset.sigma_denom_weight,
                              use_entropy_coding=dataset.use_entropy_coding, hash_n_features=dataset.hash_n_features, hash_log2_size=dataset.hash_log2_size,
                              noise_start_iter=dataset.noise_start_iter, entropy_start_iter=dataset.entropy_start_iter)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, skip_images, skip_fps)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--skip_fps", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_images, args.skip_fps)
