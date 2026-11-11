# Technicolor (light field) results: STAC high rate and uncompressed 4D-Scaffold

Two result sets on Technicolor, shared so the other machine can compare against them:

- **stac_lambda0.01**: STAC (4D-Scaffold-GS plus entropy coding), highest rate (lambda = 0.01).
  - Recipe: the fixed-quantization version (commit 46425bf), 120k iterations.
  - Settings: `--q_scaling_base 0.0001`, t_grid 0.02, densify 0.0002/0.1, `--disable_mip_filter`.
  - The model that was evaluated is the one decoded from the bitstream.
- **4dscaffold_nocomp**: 4D-Scaffold without compression, 120k iterations.
  - It uses t_grid 0.001 with the mip filter on, so its settings are not identical to STAC's.

## Protocol

- 50 frames x 16 cameras at 2048x1088. cam10 is held out as the test view; the other 15 cameras are used for training.
- Timestamps are frame/50.
- Data: `/workspace/data/technicolor_50/<scene>`, in the E-D3DGS layout.
- Metrics: PSNR, SSIM, LPIPS-ALEX and LPIPS-VGG, all computed on the 50 cam10 frames.

## Checkpoint selection

- STAC: the best test checkpoint, tracked online every 2k iterations. That checkpoint is the one compressed, decoded and evaluated. Its folder name says which iteration it was.
- 4D-Scaffold: the final iteration (120k).

## Size

- STAC: total size of the compressed bitstream, including MLPs and hash grid.
- 4D-Scaffold: size of `point_cloud/iteration_120000` (uncompressed ply plus MLPs).
- Both are in MB = 1024^2 bytes. Earlier chat reports quoted 1e6-byte MB for 4D-Scaffold, so those figures are about 5% larger.

## Scores

| Scene | Method | PSNR | SSIM | LPIPS (ALEX) | LPIPS (VGG) | Size (MB) | Eval checkpoint |
|---|---|---|---|---|---|---|---|
| Train | stac_lambda0.01 | 32.80 | 0.9409 | 0.044 | 0.087 | 7.42 | ours_112000_online_compressed |
| Train | 4dscaffold_nocomp | 32.92 | 0.9486 | 0.028 | 0.065 | 325.43 | ours_120000 |
| Theater | stac_lambda0.01 | 31.33 | 0.8934 | 0.125 | 0.226 | 4.33 | ours_100000_online_compressed |
| Theater | 4dscaffold_nocomp | 32.30 | 0.8996 | 0.087 | 0.187 | 253.39 | ours_120000 |
| Birthday | stac_lambda0.01 | 32.69 | 0.9438 | 0.040 | 0.093 | 8.14 | ours_116000_online_compressed |
| Birthday | 4dscaffold_nocomp | 32.30 | 0.9479 | 0.030 | 0.080 | 751.74 | ours_120000 |
| Painter | stac_lambda0.01 | 36.84 | 0.9337 | 0.080 | 0.159 | 3.63 | ours_118000_online_compressed |
| Painter | 4dscaffold_nocomp | 37.43 | 0.9419 | 0.068 | 0.141 | 349.50 | ours_120000 |
| Fabien | stac_lambda0.01 | 34.32 | 0.8773 | 0.167 | 0.316 | 1.59 | ours_118000_online_compressed |
| Fabien | 4dscaffold_nocomp | 35.04 | 0.9015 | 0.119 | 0.254 | 135.34 | ours_120000 |
| **Mean** | stac_lambda0.01 | 33.60 | 0.9178 | 0.091 | 0.176 | 5.02 | |
| **Mean** | 4dscaffold_nocomp | 34.00 | 0.9279 | 0.066 | 0.145 | 363.08 | |

## Files in this directory

For each method and scene, `<method>/<scene>/` contains:

- `results.json`: overall metrics.
- `per_view.json`: per-frame metrics.
- `summary.json`: the overall metrics plus size, run directory and evaluated checkpoint.
- `cfg_args`: the full training arguments.

`scores.json` has every row of the table above in one file.

## Test images

The rendered cam10 frames are attached to the GitHub Release **`technicolor-lf-results-v1`**. GT frames are not included because they are identical to `images/cam10/0000.png` and the following frames in the dataset.

- `technicolor_lf_stac_lambda0.01_test_renders.tar` (642MB)
- `technicolor_lf_4dscaffold_nocomp_test_renders.tar` (675MB)

Each tar is laid out as `<method>/<scene>/renders/000NN.png` (frames 00000 to 00049) plus `renders.mp4`. For STAC, `per_view_count.json` is also included.

To download:

```bash
gh release download technicolor-lf-results-v1 -R Kousei0329/4D-Scaffold-GS-research
tar -xf technicolor_lf_stac_lambda0.01_test_renders.tar
```

Without gh, download the files from the Releases page in a browser. The repository is private, so you need to be logged in.
