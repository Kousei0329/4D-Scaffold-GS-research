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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                # Set explicitly: torch's TORCH_CUDA_ARCH_LIST auto-detection bails out
                # (returns no -gencode flags at all, so nvcc silently falls back to sm_52)
                # because it does a naive substring check for "arch" in every nvcc flag,
                # and the "-I.../research/..." path above contains "arch" (from "research").
                "-gencode=arch=compute_86,code=sm_86",
            ]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
