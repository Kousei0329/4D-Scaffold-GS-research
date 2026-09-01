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
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.encodings import STE_multistep

global_cache = {}

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, save_cache=False, step=0):
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    bit_per_param = None
    mask_anchor_soft = None
    if pc.use_entropy_coding:
        mask_anchor_soft = pc.get_mask_anchor[visible_mask]  # [N_vis, 1], differentiable gate

    if is_training and pc.use_entropy_coding:
        if step == pc.entropy_start_iter:
            pc.update_anchor_bound()

        if step > pc.noise_start_iter:
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * 1.0
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * 0.001
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * 0.2

        if step > pc.entropy_start_iter:
            feat_context = pc.calc_interp_feat(anchor)
            mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                Q_feat_adj, Q_scaling_adj, Q_offsets_adj = pc.forward_grid(feat_context)

            Q_feat = pc.adaptive_Q(1.0, Q_feat_adj)
            Q_scaling = pc.adaptive_Q(0.001, Q_scaling_adj)
            Q_offsets = pc.adaptive_Q(0.2, Q_offsets_adj)

            grid_offsets_flat = grid_offsets.reshape(anchor.shape[0], -1)  # [N_vis, 4*n_offsets]

            bit_feat = pc.entropy_gaussian(feat, mean_feat, scale_feat, Q_feat, x_mean=pc._anchor_feat.mean())
            bit_scaling = pc.entropy_gaussian(grid_scaling, mean_scaling, scale_scaling, Q_scaling, x_mean=pc.get_scaling.mean())
            bit_offsets = pc.entropy_gaussian(grid_offsets_flat, mean_offsets, scale_offsets, Q_offsets, x_mean=pc._offset.mean())

            # rate-distortion mask gates the coding cost: an anchor the model has learned
            # is not worth keeping contributes ~0 bits here (and see below, ~0 opacity too,
            # so dropping it doesn't help the loss unless it is genuinely not needed).
            bit_feat = bit_feat * mask_anchor_soft
            bit_scaling = bit_scaling * mask_anchor_soft
            bit_offsets = bit_offsets * mask_anchor_soft

            bit_per_param = (bit_feat.sum() + bit_scaling.sum() + bit_offsets.sum()) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif (not is_training) and pc.use_entropy_coding and pc.bounds_ready:
        # Deterministic (no-noise) quantization at eval time, so rendered quality reflects
        # what a real decoder would reconstruct from the coded bitstream.
        feat_context = pc.calc_interp_feat(anchor)
        _, _, _, _, _, _, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = pc.forward_grid(feat_context)

        Q_feat = pc.adaptive_Q(1.0, Q_feat_adj)
        Q_scaling = pc.adaptive_Q(0.001, Q_scaling_adj)
        Q_offsets = pc.adaptive_Q(0.2, Q_offsets_adj).unsqueeze(1)  # broadcast over n_offsets

        feat = STE_multistep.apply(feat, Q_feat, pc._anchor_feat.mean())
        grid_scaling = STE_multistep.apply(grid_scaling, Q_scaling, pc.get_scaling.mean())
        grid_offsets = STE_multistep.apply(grid_offsets, Q_offsets, pc._offset.mean())

    ## get view properties for anchor
    ob_view = anchor[:, :3] - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist
    # time
    timestamp = viewpoint_camera.timestamp

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    cat_local_view_wodir = torch.cat([feat, ob_dist], dim=1) # [N, c+1]
    cat_local_view_wodist_wodir = torch.cat([feat, ], dim=1) # [N, c]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodir) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist_wodir)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # flow
    flow = pc.get_flow_mlp(cat_local_view_wodist_wodir).view([-1, 3])

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodir)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist_wodir)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 8]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 4]) # [mask]

    # combine for parallel masking
    if mask_anchor_soft is not None:
        concatenated = torch.cat([grid_scaling, anchor, mask_anchor_soft], dim=-1)
    else:
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, flow], dim=-1)
    masked = concatenated_all[mask]
    if mask_anchor_soft is not None:
        scaling_repeat, repeat_anchor, mask_anchor_repeat, color, scale_rot, offsets, flow = masked.split([8, 4, 1, 3, 8, 4, 3], dim=-1)
    else:
        scaling_repeat, repeat_anchor, color, scale_rot, offsets, flow = masked.split([8, 4, 3, 8, 4, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:,4:7] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,4:8])
    
    # post-process offsets to get centers for gaussians
    t = repeat_anchor[:, 3:4] + offsets[:, 3:4] * scaling_repeat[:, 3:4]
    if (pc.use_flow):
        xyz = repeat_anchor[:, :3] + offsets[:, :3] * scaling_repeat[:, :3] + flow * (t - timestamp)
    else:
        xyz = repeat_anchor[:, :3] + offsets[:, :3] * scaling_repeat[:, :3]

    sigma = torch.nn.ELU()(scale_rot[:,3:4]) + 1
    if pc.temporal_opacity == 'ours':
        opacity_t = torch.exp(-torch.abs((t-timestamp)*sigma)**pc.hparam_beta)
    else:
        opacity_t = torch.exp(-(t-timestamp)**2*sigma)

    if save_cache:
        global_cache["xyz"] = (repeat_anchor[:, :3] + offsets[:, :3] * scaling_repeat[:, :3]).detach()
        global_cache["color"] = color.detach()
        global_cache["opacity"] = opacity.detach()
        global_cache["scaling"] = scaling.detach()
        global_cache["rot"] = rot.detach()
        global_cache["mask"] = mask.detach()
        global_cache["sigma"] = sigma.detach()
        global_cache["flow"] = flow.detach()
        global_cache["t"] = t.detach()

    opacity = opacity * opacity_t
    if mask_anchor_soft is not None:
        opacity = opacity * mask_anchor_repeat

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, opacity_t, sigma, bit_per_param
    else:
        return xyz, color, opacity, scaling, rot, flow, opacity_t, sigma, mask

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, color_mode="rgb", save_cache=False, precolor=None, step=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    bit_per_param = None

    if global_cache.get("xyz") is not None:
        ## use cached values

        timestamp = viewpoint_camera.timestamp
        t = global_cache["t"]
        sigma = global_cache["sigma"]
        if pc.temporal_opacity:
            opacity_t = torch.exp(-((t-timestamp)*sigma)**4)
        else:
            opacity_t = torch.exp(-(t-timestamp)**2*sigma)
        mask = (opacity_t > 0.05).view(-1)
        
        t = t[mask]
        opacity_t = opacity_t[mask]

        flow = global_cache["flow"][mask]
        if (pc.use_flow):
            xyz = global_cache["xyz"][mask] + flow * (t - timestamp)
        else:
            xyz = global_cache["xyz"][mask]

        if color_mode == "flow":
            color = (flow * 3).clamp(-1, 1) * 0.5 + 0.5
        elif color_mode == "sigma":
            scale = 1.0
            r = (sigma[mask]*scale).clamp(0, 1)
            b = 1 - (sigma[mask]*scale).clamp(0, 1)
            g = torch.zeros_like(r)
            color = torch.cat([r, g, b], dim=1)
        elif color_mode == "mask":
            color = torch.ones_like(flow)
        else:
            color = global_cache["color"][mask]
            
        scaling = global_cache["scaling"][mask]
        rot = global_cache["rot"][mask]
        opacity = global_cache["opacity"][mask]
        
        opacity = opacity * opacity_t

        temp_mask = global_cache["mask"].clone()
        temp_mask[global_cache["mask"]] = mask
        mask = temp_mask
    else:
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask, opacity_t, sigma, bit_per_param = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, save_cache=save_cache, step=step)
        else:
            xyz, color, opacity, scaling, rot, flow, opacity_t, sigma, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, save_cache=save_cache)

        if color_mode == "flow":
            color = (flow * 3).clamp(-1, 1) * 0.5 + 0.5
        elif color_mode == "sigma":
            scale = 1.0
            r = (sigma*scale).clamp(0, 1)
            b = 1 - (sigma*scale).clamp(0, 1)
            g = torch.zeros_like(r)
            color = torch.cat([r, g, b], dim=1)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    if precolor is not None:
        precolor = precolor.view(visible_mask.shape[0], -1, 3)[visible_mask, :, :].view(-1, 3)[mask, :]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = precolor if precolor is not None else color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "opacity_t": opacity_t,
                "sigma": sigma,
                "bit_per_param": bit_per_param,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "scaling": scaling,
                }

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor[:,:3], dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor[:,:3]


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
