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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.width, cam_info.height # cam_info.image.size

    if args.resolution in [1, 2, 3, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        scale = resolution_scale * args.resolution
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    
    cx = cam_info.cx / scale
    cy = cam_info.cy / scale
    fl_y = cam_info.fl_y / scale
    fl_x = cam_info.fl_x / scale
    
    loaded_mask = None
    if not args.dataloader:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        gt_image = resized_image_rgb[:3, ...]

        if resized_image_rgb.shape[0] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]
    else:
        gt_image = cam_info.image
    
    if cam_info.depth is not None:
        depth = PILtoTorch(cam_info.depth, resolution) * 255 / 10000
    else:
        depth = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  timestamp=cam_info.timestamp,
                  cx=cx, cy=cy, fl_x=fl_x, fl_y=fl_y, depth=depth, resolution=resolution, image_path=cam_info.image_path,
                  meta_only=args.dataloader
                  )

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

# modify the code in https://github.com/hustvl/4DGaussians/blob/master/scene/neural_3D_dataset_NDC.py
def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def render_path_spiral(c2w, up, rads, zrate, N_rots=2, N=120):
    render_poses = []

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        d = np.dot(
            c2w[:3,:3],
            np.array([np.cos(theta), np.sin(theta), 1.]) * rads
        )
        c = c2w[:3,3] + d
        z = normalize(zrate * c2w[:3,2] - d)
        render_poses.append(viewmatrix(z, up, c))
    return render_poses

def get_spiral(c2ws_all, near, far, rads_scale=0.25, N_views=120):
    """
    Generate a set of poses using spiral camera trajectory as validation poses.
    """

    # test cam is the center
    c2w = c2ws_all[0,:3,:] 
    up = c2ws_all[0, :3, 1]

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    zrate = (1.0 - dt) * (near + far)

    # Get radii for spiral path
    tt = c2ws_all[1:, :3, 3] - c2ws_all[0:1, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale

    render_poses = render_path_spiral(
        c2w, up, rads, zrate, N_rots=3, N=N_views
    )
    return np.stack(render_poses)
