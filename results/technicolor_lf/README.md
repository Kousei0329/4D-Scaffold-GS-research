# Technicolor (light field) results: STAC at every lambda, and uncompressed 4D-Scaffold

Scores only. No rendered images are included.

## Result sets

- **stac_lambda{0.01,0.02,0.03,0.04,0.05}**: STAC (4D-Scaffold-GS plus entropy coding), one rate point per lambda. 0.01 is the highest rate.
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

## Mean over the 5 scenes

| Method | PSNR | SSIM | LPIPS (ALEX) | LPIPS (VGG) | Size (MB) |
|---|---|---|---|---|---|
| stac_lambda0.01 | 33.60 | 0.9178 | 0.091 | 0.176 | 5.02 |
| stac_lambda0.02 | 33.34 | 0.9150 | 0.099 | 0.185 | 3.98 |
| stac_lambda0.03 | 33.20 | 0.9132 | 0.103 | 0.189 | 3.53 |
| stac_lambda0.04 | 33.24 | 0.9130 | 0.107 | 0.193 | 3.26 |
| stac_lambda0.05 | 32.74 | 0.9071 | 0.113 | 0.201 | 2.88 |
| 4dscaffold_nocomp | 34.00 | 0.9279 | 0.066 | 0.145 | 363.08 |

## Per scene

| Scene | Method | PSNR | SSIM | LPIPS (ALEX) | LPIPS (VGG) | Size (MB) | Eval checkpoint |
|---|---|---|---|---|---|---|---|
| Train | stac_lambda0.01 | 32.80 | 0.9409 | 0.044 | 0.087 | 7.42 | ours_112000_online_compressed |
| Train | stac_lambda0.02 | 32.67 | 0.9436 | 0.040 | 0.078 | 6.69 | ours_118000_online_compressed |
| Train | stac_lambda0.03 | 32.68 | 0.9415 | 0.045 | 0.081 | 5.87 | ours_98000_online_compressed |
| Train | stac_lambda0.04 | 33.09 | 0.9455 | 0.044 | 0.081 | 5.67 | ours_98000_online_compressed |
| Train | stac_lambda0.05 | 32.08 | 0.9318 | 0.062 | 0.109 | 4.82 | ours_118000_online_compressed |
| Train | 4dscaffold_nocomp | 32.92 | 0.9486 | 0.028 | 0.065 | 325.43 | ours_120000 |
| Theater | stac_lambda0.01 | 31.33 | 0.8934 | 0.125 | 0.226 | 4.33 | ours_100000_online_compressed |
| Theater | stac_lambda0.02 | 31.28 | 0.8851 | 0.144 | 0.249 | 2.74 | ours_116000_online_compressed |
| Theater | stac_lambda0.03 | 31.38 | 0.8872 | 0.142 | 0.244 | 2.75 | ours_118000_online_compressed |
| Theater | stac_lambda0.04 | 31.15 | 0.8880 | 0.146 | 0.249 | 2.39 | ours_114000_online_compressed |
| Theater | stac_lambda0.05 | 31.23 | 0.8880 | 0.146 | 0.248 | 2.23 | ours_118000_online_compressed |
| Theater | 4dscaffold_nocomp | 32.30 | 0.8996 | 0.087 | 0.187 | 253.39 | ours_120000 |
| Birthday | stac_lambda0.01 | 32.69 | 0.9438 | 0.040 | 0.093 | 8.14 | ours_116000_online_compressed |
| Birthday | stac_lambda0.02 | 32.52 | 0.9426 | 0.045 | 0.099 | 6.39 | ours_120000_compressed |
| Birthday | stac_lambda0.03 | 32.18 | 0.9400 | 0.048 | 0.103 | 5.63 | ours_118000_online_compressed |
| Birthday | stac_lambda0.04 | 32.37 | 0.9396 | 0.052 | 0.107 | 5.29 | ours_118000_online_compressed |
| Birthday | stac_lambda0.05 | 31.76 | 0.9306 | 0.057 | 0.113 | 4.66 | ours_112000_online_compressed |
| Birthday | 4dscaffold_nocomp | 32.30 | 0.9479 | 0.030 | 0.080 | 751.74 | ours_120000 |
| Painter | stac_lambda0.01 | 36.84 | 0.9337 | 0.080 | 0.159 | 3.63 | ours_118000_online_compressed |
| Painter | stac_lambda0.02 | 36.37 | 0.9302 | 0.085 | 0.166 | 2.88 | ours_118000_online_compressed |
| Painter | stac_lambda0.03 | 36.16 | 0.9271 | 0.091 | 0.173 | 2.41 | ours_116000_online_compressed |
| Painter | stac_lambda0.04 | 36.26 | 0.9236 | 0.096 | 0.180 | 2.04 | ours_116000_online_compressed |
| Painter | stac_lambda0.05 | 35.99 | 0.9212 | 0.100 | 0.185 | 1.85 | ours_120000_compressed |
| Painter | 4dscaffold_nocomp | 37.43 | 0.9419 | 0.068 | 0.141 | 349.50 | ours_120000 |
| Fabien | stac_lambda0.01 | 34.32 | 0.8773 | 0.167 | 0.316 | 1.59 | ours_118000_online_compressed |
| Fabien | stac_lambda0.02 | 33.86 | 0.8736 | 0.179 | 0.331 | 1.21 | ours_106000_online_compressed |
| Fabien | stac_lambda0.03 | 33.59 | 0.8703 | 0.190 | 0.342 | 0.97 | ours_100000_online_compressed |
| Fabien | stac_lambda0.04 | 33.32 | 0.8685 | 0.194 | 0.347 | 0.89 | ours_108000_online_compressed |
| Fabien | stac_lambda0.05 | 32.67 | 0.8638 | 0.199 | 0.352 | 0.82 | ours_114000_online_compressed |
| Fabien | 4dscaffold_nocomp | 35.04 | 0.9015 | 0.119 | 0.254 | 135.34 | ours_120000 |

## Files in this directory

For each method and scene, `<method>/<scene>/` contains:

- `results.json`: overall metrics.
- `per_view.json`: per-frame metrics.
- `summary.json`: the overall metrics plus size, run directory and evaluated checkpoint.
- `cfg_args`: the full training arguments.

`scores.json` has every row of the per-scene table in one file.
