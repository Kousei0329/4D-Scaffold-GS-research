# 4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction

[Woong Oh Cho](https://raikuma.github.io/), In Cho, Seoha Kim, Jeongmin Bae, Youngjung Uh, Seon Joo Kim <br />

[[`arxiv`](https://arxiv.org/abs/2411.17044)]

## Overview

The official implementation of '4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction'.

<p align="center">
<img src="assets/teaser.png" width=100% height=100% 
class="center">
</p>

<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>

## Installation

We tested on a server configured with Ubuntu 20.04, cuda 11.6. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/raikuma/4D-Scaffold-GS.git
cd 4D-Scaffold-GS
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate 4d_scaffold
```

## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organized as follows for N3DV dataset:

```
data/
├── N3DV/
│   ├── cook_spinach/
│   │   ├── images
│   │   │   ├── cam00_0000.png
│   │   │   ├── cam00_0001.png
│   │   │   ├── ...
│   │   ├── transforms_train.json
│   │   ├── transforms_test.json
│   │   ├── points3d.ply
│   ├── cut_roasted_beef/
│   │   ├── images
│   │   │   ├── cam00_0000.png
│   │   │   ├── cam00_0001.png
│   │   │   ├── ...
│   │   ├── transforms_train.json
│   │   ├── transforms_test.json
│   │   ├── points3d.ply
...
```

And for technicolor dataset:

```
data/
├── technicolor_50/
│   ├── Birthday/
│   │   ├── images
│   │   │   ├── cam00
│   │   │   │   ├── 0000.png
│   │   │   │   ├── 0001.png
│   │   │   │   ├── ...
│   │   │   ├── cam01
│   │   │   │   ├── 0000.png
│   │   │   │   ├── 0001.png
│   │   │   │   ├── ...
│   │   ├── colmap
│   │   │   ├── dense
│   │   │   │   ├── workspace
│   │   │   │   │   ├── sparse
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   ├── points3D_downsample.ply
...
```

You can process N3DV dataset following the instructions in [4DGS](https://github.com/fudan-zvg/4d-gaussian-splatting) and technicolor dataset following [E-D3DGS](https://github.com/JeongminB/E-D3DGS)


## Training

For training a single scene, run the corresponding script in the ```scripts/``` folder, e.g., for training the ```cook_spinach``` scene in N3DV dataset, run:

```
bash ./scripts/train_n3dv.sh cook_spinach
```

This script will store the log (with running-time code) into ```outputs/dataset_name/scene_name/exp_name/cur_time``` automatically.

## Evaluation

We've integrated the rendering and metrics calculation process into the training code. So, when completing training, the ```rendering results```, ```fps``` and ```quality metrics``` will be printed automatically. And the rendering results will be save in the log dir. Mind that the ```fps``` is roughly estimated by 

```
torch.cuda.synchronize();t_start=time.time()
rendering...
torch.cuda.synchronize();t_end=time.time()
```

which may differ somewhat from the original 3D-GS, but it does not affect the analysis.

Meanwhile, we keep the manual rendering function with a similar usage of the counterpart in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), one can run it by 

```
python render.py -m <path to trained model> # Generate renderings and measure fps
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Contact

- Woong Oh Cho: wocho@yonsei.ac.kr

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{4dscaffoldgs,
  title={4D Scaffold Gaussian Splatting with Dynamic-Aware Anchor Growing for Efficient and High-Fidelity Dynamic Scene Reconstruction},
  author={Woong Oh Cho and In Cho and Seoha Kim and Jeongmin Bae and Youngjung Uh and Seon Joo Kim},
  booktitle={Arxiv},
  year={2025}
}
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

Most of the code is built upon the excellent work of **[Scaffold-GS](https://github.com/city-super/Scaffold-GS)**. We gratefully acknowledge their contribution.
