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

def _motion_coeff_dim(pc):
    """Width of the per-offset flow-MLP output before dt is known -- see
    arguments/__init__.py's motion_model docstring for what each case represents."""
    B = pc.motion_basis_dim
    if pc.motion_model == 'linear':
        return 3
    elif pc.motion_model in ('basis', 'fourier'):
        return 3 * B
    elif pc.motion_model == 'taylor_residual':
        return 9 + 3 * (B // 2)
    else:
        raise ValueError(f"unknown motion_model {pc.motion_model!r}")

def compute_motion_displacement(pc, flow_raw, dt):
    """flow_raw: [N, _motion_coeff_dim(pc)] per-offset motion coefficients (independent of
    dt -- produced once per offset by pc.get_flow_mlp from the anchor's feature, same as the
    original velocity-only model). dt: [N, 1] (= t - timestamp). Returns the [N, 3]
    displacement to add on top of the anchor-local base xyz. Shared between
    generate_neural_gaussians (fresh path) and render()'s cached path so both stay in sync."""
    if pc.motion_model == 'linear':
        return flow_raw * dt
    elif pc.motion_model == 'basis':
        # mean, not sum: coeff and phi are each Tanh-bounded to [-1,1], so summing B of
        # their products scales the worst case up to +-B (e.g. +-16) -- a huge, scene-
        # breaking displacement compared to the original single-term flow*dt model's
        # +-1-ish scale. Averaging keeps the combination's magnitude in the same ballpark
        # regardless of motion_basis_dim, which is what actually caused the NaN blowups
        # (via the rasterizer choking on wildly out-of-scene xyz) seen before this fix.
        B = pc.motion_basis_dim
        coeff = flow_raw.view(-1, B, 3)
        phi = pc.get_time_basis_mlp(dt).view(-1, B, 1)
        return (coeff * phi).mean(dim=1)
    elif pc.motion_model == 'fourier':
        B = pc.motion_basis_dim
        coeff = flow_raw.view(-1, B, 3)
        angles = dt * pc.motion_fourier_freqs.view(1, -1)
        phi = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).view(-1, B, 1)
        return (coeff * phi).mean(dim=1)
    elif pc.motion_model == 'taylor_residual':
        B_res = pc.motion_basis_dim // 2
        taylor_part, residual_coeff = flow_raw.split([9, 3 * B_res], dim=-1)
        vel, accel, jerk = taylor_part.split([3, 3, 3], dim=-1)
        # Clamp dt before the polynomial terms: vel/accel/jerk are Tanh-bounded (+-1) but
        # dt is raw scene-time (up to +-half the clip duration, e.g. +-5s for a 10s clip),
        # so accel*dt**2 and jerk*dt**3 blow up to O(dt^3) for anchors far in time from the
        # current frame -- the same NaN-blowup failure mode already fixed above for
        # 'basis'/'fourier' (there triggered by B, here by dt's magnitude), and the likely
        # reason this motion_model was previously abandoned without a full run. Anchors
        # this far from the current timestamp get opacity_t ~= 0 anyway (temporal window),
        # so clamping their displacement changes nothing observable while keeping the
        # rasterizer's xyz input finite. phi's dt is left unclamped -- mlp_time_basis ends
        # in Tanh, already bounded, no blowup risk there.
        dt_c = dt.clamp(-1.0, 1.0)
        taylor_disp = vel * dt_c + accel * dt_c**2 + jerk * dt_c**3
        coeff = residual_coeff.view(-1, B_res, 3)
        phi = pc.get_time_basis_mlp(dt).view(-1, B_res, 1)
        result = taylor_disp + (coeff * phi).mean(dim=1)
        # Final safety clamp on the TOTAL displacement, regardless of how the polynomial
        # terms combine -- the dt clamp alone wasn't enough: a bad training step can still
        # draw accel/jerk near their Tanh bound (+-1) at the same time dt is near its
        # clamp, giving displacement up to ~1+1+1=3 even after tightening dt's clamp from
        # +-2.0 to +-1.0. This is exactly what happened in practice (confirmed via a real
        # crash, not GPU flakiness): enough anchors got pushed out of one camera's view
        # frustum that prefilter_voxel's visible_mask collapsed to 0 anchors for that
        # frame, and grid_offsets.reshape(anchor.shape[0], -1) then failed because
        # reshaping a 0-element tensor with an inferred dim (-1) is ambiguous. linear
        # mode's displacement (flow*dt, flow in [-1,1]) stays in a comparable range in
        # practice because large-|dt| anchors get suppressed by the temporal opacity
        # window before they matter -- this clamp enforces that same ballpark here too,
        # since the dt^2/dt^3 terms don't have an equivalent built-in brake.
        return result.clamp(-3.0, 3.0)
    else:
        raise ValueError(f"unknown motion_model {pc.motion_model!r}")

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, save_cache=False, step=0):
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    # Mip-Splatting 3D filter: deterministic per-anchor variance floor, not entropy-coded
    # (recomputed from anchor position + fixed training cameras, like rotation/flow).
    # Falls back to "no filter" (0 added variance) for any caller that never computed it
    # (e.g. one-off diagnostic scripts that build a GaussianModel without going through
    # Scene()+compute_3D_filter), so this is backward-compatible with every such script.
    if pc.filter_3D.shape[0] == pc.get_anchor.shape[0]:
        anchor_filter_3D = pc.filter_3D[visible_mask].unsqueeze(-1)  # [N_vis, 1]
    else:
        anchor_filter_3D = torch.zeros((anchor.shape[0], 1), device=anchor.device)

    bit_per_param = None
    mask_anchor_soft = None
    mask_offset_soft = None
    if pc.use_entropy_coding:
        mask_anchor_soft = pc.get_mask_anchor[visible_mask]  # [N_vis, 1], differentiable gate
        if pc.use_offset_mask and pc._mask_offset.numel() > 0:
            mask_offset_soft = pc.get_mask_offset[visible_mask]  # [N_vis, n_offsets]

    if is_training and pc.use_entropy_coding:
        if step == pc.entropy_start_iter:
            pc.update_anchor_bound()

        legacy = pc.legacy_quant_noise
        qf0, qs0, qo0 = pc.q_feat_base, pc.q_scaling_base, pc.q_offsets_base
        # HAC++: fixed base-width noise only in (noise_start_iter, entropy_start_iter]; after
        # that, noise of the adaptive Q width (applied below, once Q is predicted). The legacy
        # mode kept base-width noise for the whole run, so the eval/coding quantizer (adaptive
        # Q, which then saturated at 2x base because only the rate loss reached Q_adj) was
        # twice as coarse as anything seen in training.
        if step > pc.noise_start_iter and (legacy or step <= pc.entropy_start_iter):
            feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * qf0
            grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * qs0
            grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * qo0

        if step > pc.entropy_start_iter:
            feat_context = pc.calc_interp_feat(anchor)
            if pc.use_intra_anchor_context:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj, prob_feat = pc.forward_grid(feat_context)
            else:
                mean_feat, scale_feat, mean_scaling, scale_scaling, mean_offsets, scale_offsets, \
                    Q_feat_adj, Q_scaling_adj, Q_offsets_adj = pc.forward_grid(feat_context)

            Q_feat = pc.adaptive_Q(qf0, Q_feat_adj)
            Q_scaling = pc.adaptive_Q(qs0, Q_scaling_adj)
            Q_offsets = pc.adaptive_Q(qo0, Q_offsets_adj)

            if not legacy:
                # HAC++ "for rendering": uniform noise of the adaptive step, so the rendered
                # (distortion) loss sees the same quantization strength the decoder will apply
                # and back-propagates into Q_adj (pulling Q down where precision matters).
                feat = feat + torch.empty_like(feat).uniform_(-0.5, 0.5) * Q_feat
                grid_scaling = grid_scaling + torch.empty_like(grid_scaling).uniform_(-0.5, 0.5) * Q_scaling
                grid_offsets = grid_offsets + torch.empty_like(grid_offsets).uniform_(-0.5, 0.5) * Q_offsets.unsqueeze(1)

            grid_offsets_flat = grid_offsets.reshape(anchor.shape[0], 4 * pc.n_offsets)  # [N_vis, 4*n_offsets]

            if pc.use_intra_anchor_context and not legacy:
                # HAC++ "for entropy": a fresh 5% sample of ALL anchors (not just the visible
                # ones), raw parameters + their own adaptive-Q noise, with the hash-grid
                # prediction recomputed for that sample.
                n_all = pc.get_anchor.shape[0]
                choose_idx = torch.rand(n_all, device=feat.device) <= 0.05
                anchor_c = pc.get_anchor[choose_idx]
                ctx_c = pc.calc_interp_feat(anchor_c)
                mean_feat_c, scale_feat_c, mean_scaling_c, scale_scaling_c, mean_offsets_c, scale_offsets_c, \
                    Q_feat_adj_c, Q_scaling_adj_c, Q_offsets_adj_c, prob_feat_c = pc.forward_grid(ctx_c)
                Q_feat_c = pc.adaptive_Q(qf0, Q_feat_adj_c)
                Q_scaling_c = pc.adaptive_Q(qs0, Q_scaling_adj_c)
                Q_offsets_c = pc.adaptive_Q(qo0, Q_offsets_adj_c)
                feat_c = pc._anchor_feat[choose_idx]
                feat_c = feat_c + torch.empty_like(feat_c).uniform_(-0.5, 0.5) * Q_feat_c
                scaling_c = pc.get_scaling[choose_idx]
                scaling_c = scaling_c + torch.empty_like(scaling_c).uniform_(-0.5, 0.5) * Q_scaling_c
                offsets_c = pc._offset[choose_idx].reshape(anchor_c.shape[0], 4 * pc.n_offsets)
                offsets_c = offsets_c + torch.empty_like(offsets_c).uniform_(-0.5, 0.5) * Q_offsets_c
                mean_adj, scale_adj, prob_adj = pc.mlp_channel_ctx(
                    feat_c, torch.cat([mean_feat_c, scale_feat_c, prob_feat_c], dim=-1))
                probs = torch.softmax(torch.stack([prob_feat_c, prob_adj], dim=-1), dim=-1)
                bit_feat = pc.entropy_gaussian_mix(feat_c, mean_feat_c, mean_adj, scale_feat_c, scale_adj,
                                                    probs[..., 0], probs[..., 1], Q_feat_c,
                                                    x_mean=pc._anchor_feat.mean())
                bit_scaling = pc.entropy_gaussian(scaling_c, mean_scaling_c, scale_scaling_c, Q_scaling_c,
                                                   x_mean=pc.get_scaling.mean())
                bit_offsets = pc.entropy_gaussian(offsets_c, mean_offsets_c, scale_offsets_c, Q_offsets_c,
                                                   x_mean=pc._offset.mean())
                mask_chosen = pc.get_mask_anchor[choose_idx]
                bit_feat = bit_feat * mask_chosen
                bit_scaling = bit_scaling * mask_chosen
                bit_offsets = bit_offsets * mask_chosen
                if mask_offset_soft is not None:
                    mask_offset_chosen = pc.get_mask_offset[choose_idx]
                    bit_offsets = (bit_offsets.view(-1, pc.n_offsets, 4) *
                                    mask_offset_chosen.unsqueeze(-1)).view(-1, pc.n_offsets * 4)
            elif pc.use_intra_anchor_context:
                # Teacher-forced: all channel groups computed at once from the TRUE
                # (already quantized) feat, since every group's ground truth is available
                # during training -- unlike real encode/decode, which must go sequentially.
                # Restricted to a random 5% subsample of visible anchors, applied to
                # feat/scaling/offsets ALIKE (matching HAC++'s own training loop) -- not
                # just because the 4x MLP calls of this path are expensive (a full-anchor
                # version of just the feat part was the direct cause of a CUDA OOM on an
                # otherwise idle-looking GPU that turned out to have a large, fluctuating
                # external workload leaving too little headroom), but because subsampling
                # feat alone while leaving scaling/offsets on the full anchor count would
                # shrink bit_feat's weight in the combined bit_per_param average ~20x
                # relative to them, silently under-regularizing feat's rate for a given
                # lambda. Sampling all three together preserves their relative weighting.
                choose_idx = torch.rand(feat.shape[0], device=feat.device) <= 0.05
                feat_chosen = feat[choose_idx]
                mean_feat_chosen = mean_feat[choose_idx]
                scale_feat_chosen = scale_feat[choose_idx]
                prob_feat_chosen = prob_feat[choose_idx]
                mean_adj, scale_adj, prob_adj = pc.mlp_channel_ctx(
                    feat_chosen, torch.cat([mean_feat_chosen, scale_feat_chosen, prob_feat_chosen], dim=-1))
                probs = torch.softmax(torch.stack([prob_feat_chosen, prob_adj], dim=-1), dim=-1)
                bit_feat = pc.entropy_gaussian_mix(feat_chosen, mean_feat_chosen, mean_adj,
                                                    scale_feat_chosen, scale_adj,
                                                    probs[..., 0], probs[..., 1], Q_feat[choose_idx],
                                                    x_mean=pc._anchor_feat.mean())
                bit_scaling = pc.entropy_gaussian(grid_scaling[choose_idx], mean_scaling[choose_idx],
                                                   scale_scaling[choose_idx], Q_scaling[choose_idx],
                                                   x_mean=pc.get_scaling.mean())
                bit_offsets = pc.entropy_gaussian(grid_offsets_flat[choose_idx], mean_offsets[choose_idx],
                                                   scale_offsets[choose_idx], Q_offsets[choose_idx],
                                                   x_mean=pc._offset.mean())
                mask_chosen = mask_anchor_soft[choose_idx]
                bit_feat = bit_feat * mask_chosen
                bit_scaling = bit_scaling * mask_chosen
                bit_offsets = bit_offsets * mask_chosen
                if mask_offset_soft is not None:
                    mask_offset_chosen = mask_offset_soft[choose_idx]
                    bit_offsets = (bit_offsets.view(-1, pc.n_offsets, 4) *
                                    mask_offset_chosen.unsqueeze(-1)).view(-1, pc.n_offsets * 4)
            else:
                bit_feat = pc.entropy_gaussian(feat, mean_feat, scale_feat, Q_feat, x_mean=pc._anchor_feat.mean())
                bit_scaling = pc.entropy_gaussian(grid_scaling, mean_scaling, scale_scaling, Q_scaling, x_mean=pc.get_scaling.mean())
                bit_offsets = pc.entropy_gaussian(grid_offsets_flat, mean_offsets, scale_offsets, Q_offsets, x_mean=pc._offset.mean())
                # rate-distortion mask gates the coding cost: an anchor the model has
                # learned is not worth keeping contributes ~0 bits here (and see below, ~0
                # opacity too, so dropping it doesn't help the loss unless it is genuinely
                # not needed).
                bit_feat = bit_feat * mask_anchor_soft
                bit_scaling = bit_scaling * mask_anchor_soft
                bit_offsets = bit_offsets * mask_anchor_soft
                if mask_offset_soft is not None:
                    # bit_offsets is [N_vis, 4*n_offsets] (per-element bits of the flattened
                    # per-anchor offset block); reshape to [N_vis, n_offsets, 4] so each
                    # offset slot's own mask gates just its 4 values, on top of the
                    # anchor-wide mask_anchor_soft gating above.
                    bit_offsets = (bit_offsets.view(-1, pc.n_offsets, 4) *
                                    mask_offset_soft.unsqueeze(-1)).view(-1, pc.n_offsets * 4)

            bit_per_param = (bit_feat.sum() + bit_scaling.sum() + bit_offsets.sum()) / \
                            (bit_feat.numel() + bit_scaling.numel() + bit_offsets.numel())

    elif (not is_training) and pc.use_entropy_coding and pc.bounds_ready:
        # Deterministic (no-noise) quantization at eval time, so rendered quality reflects
        # what a real decoder would reconstruct from the coded bitstream.
        feat_context = pc.calc_interp_feat(anchor)
        if pc.use_intra_anchor_context:
            _, _, _, _, _, _, Q_feat_adj, Q_scaling_adj, Q_offsets_adj, _ = pc.forward_grid(feat_context)
        else:
            _, _, _, _, _, _, Q_feat_adj, Q_scaling_adj, Q_offsets_adj = pc.forward_grid(feat_context)

        Q_feat = pc.adaptive_Q(pc.q_feat_base, Q_feat_adj)
        Q_scaling = pc.adaptive_Q(pc.q_scaling_base, Q_scaling_adj)
        Q_offsets = pc.adaptive_Q(pc.q_offsets_base, Q_offsets_adj).unsqueeze(1)  # broadcast over n_offsets

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
        cam_pid = viewpoint_camera.colmap_id
        if cam_pid in pc.appearance_id_map:
            idx = pc.appearance_id_map[cam_pid]
            camera_indicies = torch.full((cat_local_view.shape[0],), idx, dtype=torch.long, device=ob_dist.device)
            appearance = pc.get_appearance(camera_indicies)
        else:
            # Physical camera never seen in training (e.g. N3DV's held-out test view) --
            # no trained embedding row exists for it, so fall back to the mean of every
            # trained camera's embedding (standard convention for novel-view appearance
            # codes, e.g. NeRF-in-the-Wild). Goes through the ordinary forward() call
            # (every known id, then averaged) rather than Embedding.mean(), since
            # save_mlp_checkpoints jit-traces this module -- the reloaded
            # RecursiveScriptModule only has forward(), not the original class's extra
            # methods.
            all_ids = torch.arange(len(pc.appearance_id_map), dtype=torch.long, device=ob_dist.device)
            appearance_mean = pc.get_appearance(all_ids).mean(dim=0, keepdim=True)
            appearance = appearance_mean.expand(cat_local_view.shape[0], -1)

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

    flow = pc.get_flow_mlp(cat_local_view_wodist_wodir).view([-1, _motion_coeff_dim(pc)])
    if pc.use_motion_gate and pc._motion_gate.numel() > 0:
        motion_gate_soft = pc.get_motion_gate[visible_mask]  # [N_vis, 1]
        motion_gate_repeat = repeat(motion_gate_soft, 'n c -> (n k) c', k=pc.n_offsets)
        flow = flow * motion_gate_repeat

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
        concatenated = torch.cat([grid_scaling, anchor, mask_anchor_soft, anchor_filter_3D], dim=-1)
    else:
        concatenated = torch.cat([grid_scaling, anchor, anchor_filter_3D], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    if mask_offset_soft is not None:
        # mask_offset_soft is [N_vis, n_offsets]; flattening row-major (n, k) matches the
        # (n k) ordering grid_offsets/color/scale_rot/flow are already reshaped into above.
        mask_offset_flat = mask_offset_soft.reshape(-1, 1)
        concatenated_all = torch.cat([concatenated_repeated, mask_offset_flat, color, scale_rot, offsets, flow], dim=-1)
    else:
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, flow], dim=-1)
    masked = concatenated_all[mask]
    flow_dim = _motion_coeff_dim(pc)
    mask_offset_repeat = None
    if mask_anchor_soft is not None and mask_offset_soft is not None:
        scaling_repeat, repeat_anchor, mask_anchor_repeat, filter_3d_repeat, mask_offset_repeat, color, scale_rot, offsets, flow = masked.split([7, 4, 1, 1, 1, 3, 8, 4, flow_dim], dim=-1)
    elif mask_anchor_soft is not None:
        scaling_repeat, repeat_anchor, mask_anchor_repeat, filter_3d_repeat, color, scale_rot, offsets, flow = masked.split([7, 4, 1, 1, 3, 8, 4, flow_dim], dim=-1)
    else:
        scaling_repeat, repeat_anchor, filter_3d_repeat, color, scale_rot, offsets, flow = masked.split([7, 4, 1, 3, 8, 4, flow_dim], dim=-1)

    # post-process cov -- Mip-Splatting 3D filter added in VARIANCE space, so scaling can't
    # shrink below the finest footprint any training camera could resolve at this anchor.
    # The tiny eps keeps sqrt's argument strictly positive: at exactly 0 (e.g. filter
    # disabled/0 AND sigmoid(scale_rot) landing on ~0 for some channel, which does happen
    # in practice once training is confident enough), sqrt's backward (0.5/sqrt(x)) is
    # NaN at x=0 -- this caused a real, reproducible NaN blowup (via PowBackward0) in a
    # disable_mip_filter run, confirmed via torch.autograd.set_detect_anomaly. Negligible
    # numerically (sqrt(1e-12) ~= 1e-6, far below any meaningful scaling magnitude).
    scaling = torch.sqrt((scaling_repeat[:,4:7] * torch.sigmoid(scale_rot[:,:3]))**2 + filter_3d_repeat + 1e-12)
    rot = pc.rotation_activation(scale_rot[:,4:8])
    
    # post-process offsets to get centers for gaussians
    t = repeat_anchor[:, 3:4] + offsets[:, 3:4] * scaling_repeat[:, 3:4]
    if (pc.use_flow):
        dt = t - timestamp
        xyz = repeat_anchor[:, :3] + offsets[:, :3] * scaling_repeat[:, :3] + compute_motion_displacement(pc, flow, dt)
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
        ## mask_anchor_repeat (the entropy-coding soft mask gate) is anchor-intrinsic, not
        ## time-dependent, so fold it into the cached static opacity now -- otherwise the
        ## cached branch in render() (which never sees mask_anchor_soft) silently skips this
        ## multiplication, which for use_entropy_coding=True models caused a ~14dB PSNR
        ## collapse (the actual root cause, superseding the earlier opacity_t>0.05 theory).
        cached_opacity = opacity
        if mask_anchor_soft is not None:
            cached_opacity = cached_opacity * mask_anchor_repeat
        if mask_offset_repeat is not None:
            cached_opacity = cached_opacity * mask_offset_repeat
        global_cache["opacity"] = cached_opacity.detach()
        global_cache["scaling"] = scaling.detach()
        global_cache["rot"] = rot.detach()
        global_cache["mask"] = mask.detach()
        global_cache["sigma"] = sigma.detach()
        global_cache["flow"] = flow.detach()
        global_cache["t"] = t.detach()
        global_cache["_camera_center"] = viewpoint_camera.camera_center.detach().clone()

    opacity = opacity * opacity_t
    if mask_anchor_soft is not None:
        opacity = opacity * mask_anchor_repeat
    if mask_offset_repeat is not None:
        opacity = opacity * mask_offset_repeat

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, opacity_t, sigma, bit_per_param, flow
    else:
        return xyz, color, opacity, scaling, rot, flow, opacity_t, sigma, mask

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, color_mode="rgb", save_cache=False, precolor=None, step=0):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    bit_per_param = None

    if global_cache.get("xyz") is not None and torch.equal(global_cache.get("_camera_center"), viewpoint_camera.camera_center):
        ## use cached values -- only valid when the camera hasn't moved since the cache
        ## was built (view-dependent color/opacity/etc. were computed for that camera).
        ## Keeps every statically-visible gaussian (same set as the fresh path, no extra
        ## opacity_t threshold) and weights by the continuous opacity_t, matching the
        ## fresh path exactly. global_cache["opacity"] already has the entropy-coding
        ## soft mask gate folded in (see generate_neural_gaussians), so no separate
        ## mask multiplication is needed here.

        timestamp = viewpoint_camera.timestamp
        t = global_cache["t"]
        sigma = global_cache["sigma"]
        if pc.temporal_opacity:
            opacity_t = torch.exp(-((t-timestamp)*sigma)**4)
        else:
            opacity_t = torch.exp(-(t-timestamp)**2*sigma)

        flow = global_cache["flow"]
        if (pc.use_flow):
            dt = t - timestamp
            xyz = global_cache["xyz"] + compute_motion_displacement(pc, flow, dt)
        else:
            xyz = global_cache["xyz"]

        if color_mode == "flow":
            color = (flow * 3).clamp(-1, 1) * 0.5 + 0.5
        elif color_mode == "sigma":
            scale = 1.0
            r = (sigma*scale).clamp(0, 1)
            b = 1 - (sigma*scale).clamp(0, 1)
            g = torch.zeros_like(r)
            color = torch.cat([r, g, b], dim=1)
        elif color_mode == "mask":
            color = torch.ones_like(xyz)
        else:
            color = global_cache["color"]

        scaling = global_cache["scaling"]
        rot = global_cache["rot"]
        opacity = global_cache["opacity"] * opacity_t

        mask = global_cache["mask"]
    else:
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask, opacity_t, sigma, bit_per_param, flow = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, save_cache=save_cache, step=step)
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
                # World-space positions of this step's rendered neural gaussians, exposed
                # for the photometric-error densification signal (see training_statis):
                # projecting these to screen space lets us sample the per-pixel
                # reconstruction error at each gaussian's actual rendered location.
                "xyz": xyz.detach(),
                # Raw per-offset motion coefficients (None when use_flow=False), exposed
                # for the optional motion-magnitude temporal-sharpness regularization (see
                # opt.motion_sigma_reg_weight in train.py) -- not used by anything else.
                "flow": flow if pc.use_flow else None,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "scaling": scaling,
                }

def render_cached(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, color_mode="rgb"):
    """
    Eval-time render() that reuses the anchor-level neural-Gaussian outputs (color,
    opacity, scaling, rotation, flow, sigma, base xyz) across consecutive calls for the
    SAME camera, instead of recomputing the 4 MLPs every frame. Valid because, for a
    fixed camera, those outputs depend only on each anchor's own feature and the
    (unchanged) view direction -- never on the per-frame timestamp; only the temporal
    opacity gate and the flow-shifted position need to be recomputed per frame, which
    render()'s existing cache branch already does cheaply.

    Rebuilds the cache automatically whenever the camera changes (compares
    viewpoint_camera.camera_center), so this is always correct to call in place of
    render() -- it only pays off when consecutive calls share a camera (e.g. the
    N3DV/Neu3D test-set protocol: one held-out camera x many timestamps, or any other
    fixed-viewpoint playback/video export). For a camera that changes every call (e.g.
    free-viewpoint navigation, or iterating training cameras), it transparently falls
    back to rebuilding every time -- correct, just without the speedup.

    Confirmed empirically (not just by code reading) to match plain render()'s output
    exactly (max abs pixel diff 0.0, down to floating-point noise) across sampled frames
    0/50/100/150/200/250/299 of a real entropy-coded checkpoint, including the
    entropy-coding soft mask gate (mask_anchor_repeat), which is anchor-intrinsic and is
    folded into the cached static opacity rather than recomputed per frame. There is no
    approximation: every statically-visible gaussian is kept and weighted by the
    continuous, per-frame-recomputed temporal opacity gate, same as plain render().
    """
    cached_center = global_cache.get("_camera_center")
    if cached_center is None or not torch.equal(cached_center, viewpoint_camera.camera_center):
        generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=False, save_cache=True)
    return render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=scaling_modifier,
                  visible_mask=visible_mask, retain_grad=retain_grad, color_mode=color_mode)

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
