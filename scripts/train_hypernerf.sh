#!/usr/bin/env bash
set -uo pipefail

# HyperNeRF (real, monocular "vrig" captures) training at the codebase defaults (30000 iters).
# Layout assumed:  <parent>/4D-Scaffold-GS-research/scripts/train_hypernerf.sh
#                  <parent>/data/HyperNeRF/<scene>/{dataset.json,metadata.json,scene.json,camera/,rgb/2x/,points.npy}
# Loader: scene/dataset_readers.py readHyperNeRFSceneInfo (4DGS protocol: rgb/2x images, train/val
# split from dataset.json, time = warp_id / max). Images are already 2x-downsampled, so -r 1.
#
# Usage:
#   bash scripts/train_hypernerf.sh <gpu_id> [scene1 scene2 ...]   # no scene -> all under data/HyperNeRF
#
# Env overrides (same meaning as scripts/train_dnerf.sh):
#   LAMBDAS="0.01 0.02"  ITERATIONS=30000  T_GRID=auto|<float>  VOXEL=0.001  MAX_INIT_T=0.0
#   GRAD_TH=0.0002  BATCH=1  SEED=  DATA_ROOT=/path
#   INIT_PC=colmap|npy   initial points: colmap = points3D_downsample2.ply (4DGS protocol, default;
#                        tagged _pccolmap), npy = dataset points.npy (runs before 2026-11-04 23:20 = "baseline")
#   T_GRID=auto (default) = 2/n_time_steps per scene (n = distinct warp_ids), the 4DGS rule.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_ROOT="${DATA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/data}"

GPU="${1:?usage: train_hypernerf.sh <gpu_id> [scene ...]}"; shift
scenes=("$@")
read -r -a LAMBDAS <<< "${LAMBDAS:-0.01}"
T_GRID="${T_GRID:-auto}"
VOXEL="${VOXEL:-0.001}"
MAX_INIT_T="${MAX_INIT_T:-0.0}"
GRAD_TH="${GRAD_TH:-0.0002}"
BATCH="${BATCH:-1}"
SEED="${SEED:-}"
INIT_PC="${INIT_PC:-colmap}"
seed_flag=""; [ -n "${SEED}" ] && seed_flag="--seed ${SEED}"

if [ ${#scenes[@]} -eq 0 ]; then
    for d in "${DATA_ROOT}/HyperNeRF"/*/; do
        [ -f "${d}/dataset.json" ] && scenes+=("$(basename "${d}")")
    done
    [ ${#scenes[@]} -eq 0 ] && { echo "error: no scenes under ${DATA_ROOT}/HyperNeRF" >&2; exit 1; }
    echo "no scene given -- running all: ${scenes[*]}"
fi

cd "${REPO_ROOT}"

iterations="${ITERATIONS:-30000}"
q=$((iterations / 4))
SCHED=""
for p in position offset mlp_opacity mlp_cov mlp_color mlp_flow mlp_featurebank appearance encoding_xyz mlp_grid; do
    SCHED+=" --${p}_lr_max_steps ${iterations}"
done
SCHED+=" --update_until $((iterations / 2))"
SCHED+=" --test_iterations $q $((2*q)) $((3*q)) ${iterations} --save_iterations $q $((2*q)) $((3*q)) ${iterations}"

run_tag="baseline"
[ "${INIT_PC}" != "npy" ] && run_tag+="_pc${INIT_PC}"
[ "${iterations}" != "30000" ] && run_tag+="_it${iterations}"
[ "${T_GRID}" != "auto" ] && run_tag+="_tg${T_GRID}"
[ "${VOXEL}" != "0.001" ] && run_tag+="_vx${VOXEL}"
[ "${MAX_INIT_T}" != "0.0" ] && run_tag+="_mit${MAX_INIT_T}"
[ "${GRAD_TH}" != "0.0002" ] && run_tag+="_gth${GRAD_TH}"
[ "${BATCH}" != "1" ] && run_tag+="_bs${BATCH}"
[ -n "${SEED}" ] && run_tag+="_seed${SEED}"

skipped=()
failed=()

for scene in "${scenes[@]}"; do
    source_path="${DATA_ROOT}/HyperNeRF/${scene}"
    missing=()
    for f in dataset.json metadata.json scene.json points.npy; do
        [ -s "${source_path}/${f}" ] || missing+=("${f}")
    done
    [ -d "${source_path}/rgb/2x" ] || missing+=("rgb/2x")
    [ "${INIT_PC}" = "colmap" ] && { [ -s "${source_path}/points3D_downsample2.ply" ] || missing+=("points3D_downsample2.ply"); }
    if [ ${#missing[@]} -gt 0 ]; then
        echo "warning: ${source_path} missing ${missing[*]}, skipping" >&2
        skipped+=("${scene}")
        continue
    fi

    t_grid="${T_GRID}"
    if [ "${T_GRID}" = "auto" ]; then
        t_grid=$(python -I -c 'import json,os,sys
p=sys.argv[1]; m=json.load(open(os.path.join(p,"metadata.json"))); ids=json.load(open(os.path.join(p,"dataset.json")))["ids"]
n=len({m[i]["warp_id"] for i in ids}); print("%.5f" % (2.0/n))' "${source_path}") || { echo "error: could not compute t_grid for ${scene}" >&2; failed+=("${scene}:t_grid"); continue; }
    fi

    for lmbda in "${LAMBDAS[@]}"; do
        run_dir="outputs/HyperNeRF/${scene}/${run_tag}/lmbda_${lmbda}"
        if [ -f "${run_dir}/outputs.log" ] && grep -q "FINAL SUMMARY, iteration" "${run_dir}/outputs.log"; then
            echo "=== Skipping ${scene} lmbda=${lmbda} (already completed) ==="
            continue
        fi
        echo "=== [GPU ${GPU}] ${scene} lmbda=${lmbda} t_grid=${t_grid} -> ${run_dir} ==="
        python train.py --eval -s "${source_path}" --gpu "${GPU}" -m "${run_dir}" \
            --time_duration 0.0 1.0 --dataloader --num_workers 8 \
            --voxel_size "${VOXEL}" --init_voxel_scale 1.0 --max_init_t "${MAX_INIT_T}" --hyper_init_points "${INIT_PC}" \
            --densify_grad_threshold "${GRAD_TH}" --batch_size "${BATCH}" --update_init_factor 16 \
            --appearance_dim 0 -r 1 --n_offsets 10 --t_grid_size "${t_grid}" \
            --temporal_opacity ours --use_flow --sigma_denom_weight \
            --disable_mip_filter --use_joint_4d_hash --use_intra_anchor_context \
            --use_entropy_coding --lmbda "${lmbda}" ${seed_flag} \
            --iterations ${iterations} ${SCHED}
        status=$?
        if [ ${status} -ne 0 ]; then
            echo "error: training failed for ${scene} lmbda=${lmbda} (exit ${status})" >&2
            failed+=("${scene}:lmbda_${lmbda}")
        fi
    done
done

echo "=== Done [GPU ${GPU}]: ${scenes[*]} ==="
[ ${#skipped[@]} -gt 0 ] && echo "skipped: ${skipped[*]}" >&2
[ ${#failed[@]} -gt 0 ] && echo "failed: ${failed[*]}" >&2
{ [ ${#skipped[@]} -gt 0 ] || [ ${#failed[@]} -gt 0 ]; } && exit 1
exit 0
