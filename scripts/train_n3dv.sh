#!/usr/bin/env bash
set -uo pipefail

# Resolve paths from this script's location so the script works from any CWD.
# Layout assumed:  <parent>/4D-Scaffold-GS-research/scripts/train_n3dv.sh
#                  <parent>/data/N3DV/<scene>/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_ROOT="${DATA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/data}"

# Usage:
#   bash ./scripts/train_n3dv.sh                                # all preprocessed scenes under $DATA_ROOT/N3DV
#   bash ./scripts/train_n3dv.sh cook_spinach                    # one scene
#   bash ./scripts/train_n3dv.sh cook_spinach coffee_martini ... # several scenes, run sequentially
scenes=("$@")

if [ ${#scenes[@]} -eq 0 ]; then
    if [ ! -d "${DATA_ROOT}/N3DV" ]; then
        echo "error: ${DATA_ROOT}/N3DV not found" >&2
        exit 1
    fi
    scenes=()
    for d in "${DATA_ROOT}/N3DV"/*/; do
        [ -d "${d}" ] && scenes+=("$(basename "${d}")")
    done
    if [ ${#scenes[@]} -eq 0 ]; then
        echo "error: no scenes found under ${DATA_ROOT}/N3DV" >&2
        exit 1
    fi
    echo "no scene given -- running all scenes under ${DATA_ROOT}/N3DV: ${scenes[*]}"
fi

cd "${REPO_ROOT}"

# train.py calls wandb.login() on every process start (train.py:639), which prompts
# interactively (account / offline / disabled) unless WANDB_MODE is already set. Picking
# "Don't visualize my results" never writes ~/.netrc, so checking for that file alone still
# re-prompts on every later scene. Instead, resolve the choice once and cache it to a marker
# file so both the rest of this loop and future runs of this script reuse it silently.
WANDB_MODE_FILE="${REPO_ROOT}/.wandb_mode"

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_MODE="${WANDB_MODE:-online}"
elif [ -f "${WANDB_MODE_FILE}" ]; then
    export WANDB_MODE="$(cat "${WANDB_MODE_FILE}")"
else
    wandb login || true
    if [ -f "${HOME}/.netrc" ]; then
        export WANDB_MODE="online"
    else
        export WANDB_MODE="offline"
    fi
    echo "${WANDB_MODE}" > "${WANDB_MODE_FILE}"
fi

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

# Used by the "already_done" resume check below to recognize a completed run -- parsed from
# CONFIGS itself so it always matches whichever --iterations value is active above.
if [[ "${CONFIGS}" =~ --iterations\ ([0-9]+) ]]; then
    iterations="${BASH_REMATCH[1]}"
else
    echo "error: could not parse --iterations from CONFIGS" >&2
    exit 1
fi

# Rate-distortion sweep: one full training run per lambda, run sequentially (same GPU) so
# they don't contend for memory. All scene/lambda combos from a single invocation of this
# script share ONE timestamped experiment dir (matching the reference multi-lambda runner's
# outputs/experiments/{name}_{TIMESTAMP}/{dataset}/{scene}/{lambda} layout) -- afterward,
# compare the FINAL SUMMARY / SSIM-PSNR-LPIPS-ALEX blocks across
# outputs/4DHAC_experiments_${time}/${data}/lmbda_*/outputs.log to build the RD curve.
LAMBDAS=(0.001 0.002 0.003 0.004 0.005)
# LAMBDAS=(0.01 0.02)

# One shared timestamp for this whole sweep invocation (not per scene/lambda) -- re-running
# the script later (e.g. to pick up scenes/lambdas that failed or weren't run yet) gets its
# own new experiment dir; the resume check below searches across all of them.
time=$(date "+%Y-%m-%d_%H-%M-%S")
EXPERIMENT_DIR="outputs/4DHAC_experiments_${time}"

skipped=()
failed=()

for scene in "${scenes[@]}"; do
    data="N3DV/${scene}"
    source_path="${DATA_ROOT}/${data}"

    if [ ! -d "${source_path}" ]; then
        echo "warning: scene directory not found, skipping: ${source_path}" >&2
        skipped+=("${scene}")
        continue
    fi

    # The N3DV loader takes the Blender branch in scene/__init__.py, which needs the
    # transforms_*.json produced by the 4DGS preprocessing. Without them training dies
    # on "Could not recognize scene type!" only after the run dir has been created.
    # -s (not -e) also catches a truncated/empty file (e.g. a 0-byte points3d.ply from a
    # failed copy), which otherwise crashes much later with a confusing
    # "'NoneType' object has no attribute 'points'" deep inside gaussian_model.py.
    missing=()
    for f in transforms_train.json transforms_test.json points3d.ply; do
        [ -s "${source_path}/${f}" ] || missing+=("${f}")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "warning: ${source_path} is missing or has empty ${missing[*]} -- run the 4DGS preprocessing for this scene first. Skipping." >&2
        skipped+=("${scene}")
        continue
    fi

    for lmbda in "${LAMBDAS[@]}"; do
        exp_name="lmbda_${lmbda}"

        # Resume support: a prior completed run for this scene+lambda can be sitting under any
        # earlier invocation's outputs/4DHAC_experiments_<timestamp>/ dir. train.py only reaches
        # evaluate() -- which prints "FINAL SUMMARY, iteration N" -- after a full, successful pass
        # (see train.py:668-677); a crash (disk full, OOM, ...) anywhere before that never prints
        # it. Treat that line as the completion marker and skip re-running scene+lambda combos
        # that already have it, so re-invoking this script only retries what previously failed or
        # never ran.
        already_done=0
        for prev_log in outputs/4DHAC_experiments_*/"${data}"/"${exp_name}"/outputs.log; do
            [ -f "${prev_log}" ] || continue
            if grep -q "FINAL SUMMARY, iteration ${iterations}" "${prev_log}"; then
                already_done=1
                break
            fi
        done
        if [ ${already_done} -eq 1 ]; then
            echo "=== Skipping scene=${scene} lmbda=${lmbda} (already completed in a previous run) ==="
            continue
        fi

        run_dir="${EXPERIMENT_DIR}/${data}/${exp_name}"
        echo "=== Running scene=${scene} lmbda=${lmbda} on ${source_path} -> ${run_dir} ==="
        python train.py --eval -s "${source_path}" --gpu ${gpu} --voxel_size ${vsize} --update_init_factor ${update_init_factor} --appearance_dim ${appearance_dim} -r ${resolution} -m "${run_dir}" --time_duration ${time_duration} --num_workers ${num_workers} --dataloader --n_offsets ${n_offsets} --t_grid_size ${t_grid_size} --temporal_opacity ours --use_flow --sigma_denom_weight --vis --vis_interval 1000 --vis_idx 299 --use_wandb $CONFIGS --use_entropy_coding --lmbda ${lmbda}
        status=$?
        if [ ${status} -ne 0 ]; then
            echo "error: training failed for scene=${scene} lmbda=${lmbda} (exit ${status})" >&2
            failed+=("${scene}:lmbda_${lmbda}")
        fi
    done
done

echo "=== Done. scenes requested: ${scenes[*]} ==="
[ ${#skipped[@]} -gt 0 ] && echo "skipped (missing data/preprocessing): ${skipped[*]}" >&2
[ ${#failed[@]} -gt 0 ] && echo "failed runs: ${failed[*]}" >&2

if [ ${#skipped[@]} -gt 0 ] || [ ${#failed[@]} -gt 0 ]; then
    exit 1
fi
