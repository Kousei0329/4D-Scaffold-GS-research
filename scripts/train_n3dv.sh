data=N3DV/${1}
gpu=1
num_workers=8
vsize=0.001
update_init_factor=16
appearance_dim=0
resolution=2
time_duration='0.0 10.0'
n_offsets=10
t_grid_size=0.0333
CONFIGS="--iterations 120000 --position_lr_max_steps 120000 --offset_lr_max_steps 120000 --mlp_opacity_lr_max_steps 120000 --mlp_cov_lr_max_steps 120000 --mlp_color_lr_max_steps 120000 --mlp_featurebank_lr_max_steps 120000 --appearance_lr_max_steps 120000 --update_until 60000 --test_iterations 30000 60000 90000 120000 --save_iterations 30000 60000 90000 120000 "
# CONFIGS="--iterations 30000 --position_lr_max_steps 30000 --offset_lr_max_steps 30000 --mlp_opacity_lr_max_steps 30000 --mlp_cov_lr_max_steps 30000 --mlp_color_lr_max_steps 30000 --mlp_featurebank_lr_max_steps 30000 --appearance_lr_max_steps 30000 --update_until 60000 --test_iterations 30000 --save_iterations 30000  "

# Rate-distortion sweep: one full training run per lambda, run sequentially (same GPU) so
# they don't contend for memory. Each gets its own timestamped exp_name so results don't
# clobber each other -- afterward, compare the FINAL SUMMARY / SSIM-PSNR-LPIPS-ALEX blocks
# across outputs/${data}/lmbda_*/*/outputs.log to build the RD curve.
# LAMBDAS=(0.0005 0.001 0.002 0.004 0.008)
LAMBDAS=(0.01)


for lmbda in "${LAMBDAS[@]}"; do
    exp_name="lmbda_${lmbda}"
    time=$(date "+%Y-%m-%d_%H-%M-%S")
    echo "=== Running lmbda=${lmbda} -> outputs/${data}/${exp_name}/${time} ==="
    python train.py --eval -s ../data/${data} --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} -r ${resolution} -m outputs/${data}/${exp_name}/$time --time_duration ${time_duration} --num_workers ${num_workers} --dataloader --n_offsets ${n_offsets} --t_grid_size ${t_grid_size} --temporal_opacity ours --use_flow --sigma_denom_weight --vis --vis_interval 1000 --vis_idx 299 --use_wandb $CONFIGS --use_entropy_coding --lmbda ${lmbda}
done