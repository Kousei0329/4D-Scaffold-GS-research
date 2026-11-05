#!/usr/bin/env bash
set -uo pipefail

# D-NeRF (synthetic, monocular) training at the codebase defaults (30000 iters).
# Layout assumed:  <parent>/4D-Scaffold-GS-research/scripts/train_dnerf.sh
#                  <parent>/data/D-NeRF/<scene>/{train,test,val,transforms_*.json}
#
# Usage:
#   bash scripts/train_dnerf.sh <gpu_id> [scene1 scene2 ...]   # no scene -> all under data/D-NeRF
#
# Env overrides:
#   LAMBDAS="0.01 0.02"   rate-distortion lambdas, run sequentially per scene (default: 0.01)
#   WHITE_BG=1|0          composite RGBA onto white (D-NeRF paper convention, default) or black
#   ITERATIONS=120000     training length (default 30000); lr schedules/update_until scale with it
#   T_GRID=0.0067|auto    anchor time-grid size (default 0.0333 = 1 frame on N3DV's 0-10 axis;
#                         D-NeRF time is 0-1, so 1 frame is ~1/(n_frames-1)).
#                         auto = 2/n_frames per scene (n_frames = unique timestamps in
#                         train+test), i.e. n_frames/2 time bins -- the same rule 4DGS uses
#                         for its per-scene K-Planes time resolution on both D-NeRF and DyNeRF.
#   RES=1                 -r downscale factor (default 2 = 400x400; 1 = 800x800 like 4DGS)
#   VOXEL=0.005           anchor voxel_size (default 0.001)
#   MAX_INIT_T=1.0        random-init anchors get t ~ U(0, MAX_INIT_T) (default 0.0 = all at t=0;
#                         D-NeRF has no point cloud, so every initial anchor otherwise starts at t=0)
#   INIT_PTS=100000       random init point count (default 10000; 4DGS/4D-RotorGS use 100000)
#   GRAD_TH=0.00005       densify_grad_threshold (default 0.0002; 4D-RotorGS uses 5e-5)
#   BATCH=4               views per optimizer step (default 1; 4DGS uses 4, 4D-RotorGS 3)
#   SEED=1                RNG seed (default: train.py's fixed seeding); for variance studies
#   DATA_ROOT=/path       defaults to <parent>/data
#
# 2-GPU example:
#   bash scripts/train_dnerf.sh 0 bouncingballs hellwarrior hook jumpingjacks &
#   bash scripts/train_dnerf.sh 1 lego mutant standup trex &
#   wait

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
DATA_ROOT="${DATA_ROOT:-$(cd "${REPO_ROOT}/.." && pwd)/data}"

GPU="${1:?usage: train_dnerf.sh <gpu_id> [scene ...]}"; shift
scenes=("$@")
read -r -a LAMBDAS <<< "${LAMBDAS:-0.01}"
WHITE_BG="${WHITE_BG:-1}"
T_GRID="${T_GRID:-0.0333}"
RES="${RES:-2}"
VOXEL="${VOXEL:-0.001}"
SEED="${SEED:-}"
MAX_INIT_T="${MAX_INIT_T:-0.0}"
INIT_PTS="${INIT_PTS:-10000}"
GRAD_TH="${GRAD_TH:-0.0002}"
BATCH="${BATCH:-1}"
seed_flag=""; [ -n "${SEED}" ] && seed_flag="--seed ${SEED}"

if [ ${#scenes[@]} -eq 0 ]; then
    for d in "${DATA_ROOT}/D-NeRF"/*/; do
        [ -d "${d}" ] && scenes+=("$(basename "${d}")")
    done
    [ ${#scenes[@]} -eq 0 ] && { echo "error: no scenes under ${DATA_ROOT}/D-NeRF" >&2; exit 1; }
    echo "no scene given -- running all: ${scenes[*]}"
fi

cd "${REPO_ROOT}"

iterations="${ITERATIONS:-30000}"
bg_flag=""
bg_tag="blackbg"
if [ "${WHITE_BG}" = "1" ]; then bg_flag="--white_background"; bg_tag="whitebg"; fi

# Schedule: every *_lr_max_steps follows the iteration count and densification stops at the
# halfway point (codebase default 30000/15000; N3DV uses 120000/60000). Evaluate/save at
# quarter points. Non-default lengths get their own output dir so runs never collide.
q=$((iterations / 4))
SCHED=""
for p in position offset mlp_opacity mlp_cov mlp_color mlp_flow mlp_featurebank appearance encoding_xyz mlp_grid; do
    SCHED+=" --${p}_lr_max_steps ${iterations}"
done
SCHED+=" --update_until $((iterations / 2))"
SCHED+=" --test_iterations $q $((2*q)) $((3*q)) ${iterations} --save_iterations $q $((2*q)) $((3*q)) ${iterations}"
run_tag="baseline_${bg_tag}"
[ "${iterations}" != "30000" ] && run_tag+="_it${iterations}"
[ "${T_GRID}" != "0.0333" ] && run_tag+="_tg${T_GRID}"
[ "${RES}" != "2" ] && run_tag+="_r${RES}"
[ "${VOXEL}" != "0.001" ] && run_tag+="_vx${VOXEL}"
[ "${MAX_INIT_T}" != "0.0" ] && run_tag+="_mit${MAX_INIT_T}"
[ "${INIT_PTS}" != "10000" ] && run_tag+="_pts${INIT_PTS}"
[ "${GRAD_TH}" != "0.0002" ] && run_tag+="_gth${GRAD_TH}"
[ "${BATCH}" != "1" ] && run_tag+="_bs${BATCH}"
[ -n "${SEED}" ] && run_tag+="_seed${SEED}"

skipped=()
failed=()

for scene in "${scenes[@]}"; do
    source_path="${DATA_ROOT}/D-NeRF/${scene}"
    missing=()
    for f in transforms_train.json transforms_test.json; do
        [ -s "${source_path}/${f}" ] || missing+=("${f}")
    done
    if [ ! -d "${source_path}" ] || [ ${#missing[@]} -gt 0 ]; then
        echo "warning: ${source_path} missing or incomplete (${missing[*]:-dir}), skipping" >&2
        skipped+=("${scene}")
        continue
    fi

    t_grid="${T_GRID}"
    if [ "${T_GRID}" = "auto" ]; then
        t_grid=$(python -I -c 'import json,os,sys
p=sys.argv[1]; t=set()
for f in ("transforms_train.json","transforms_test.json"):
    t |= {x["time"] for x in json.load(open(os.path.join(p,f)))["frames"]}
print(f"{2/len(t):.5f}")' "${source_path}") || { echo "error: could not compute t_grid for ${scene}" >&2; failed+=("${scene}:t_grid"); continue; }
    fi

    for lmbda in "${LAMBDAS[@]}"; do
        run_dir="outputs/D-NeRF/${scene}/${run_tag}/lmbda_${lmbda}"

        # Completion marker: train.py only logs "FINAL SUMMARY" after a full pass. N is the
        # iteration of the checkpoint that shipped (online best-ckpt tracking can pick one
        # before the last iteration, e.g. lego -> 28000), so don't require N == iterations.
        if [ -f "${run_dir}/outputs.log" ] && grep -q "FINAL SUMMARY, iteration" "${run_dir}/outputs.log"; then
            echo "=== Skipping ${scene} lmbda=${lmbda} (already completed) ==="
            continue
        fi

        echo "=== [GPU ${GPU}] ${scene} lmbda=${lmbda} ${bg_tag} t_grid=${t_grid} -> ${run_dir} ==="
        python train.py --eval -s "${source_path}" --gpu "${GPU}" -m "${run_dir}" \
            --time_duration 0.0 1.0 --dataloader --num_workers 8 \
            --voxel_size "${VOXEL}" --init_voxel_scale 1.0 --max_init_t "${MAX_INIT_T}" --random_init_pts "${INIT_PTS}" --densify_grad_threshold "${GRAD_TH}" --batch_size "${BATCH}" --update_init_factor 16 \
            --appearance_dim 0 -r "${RES}" --n_offsets 10 --t_grid_size "${t_grid}" \
            --temporal_opacity ours --use_flow --sigma_denom_weight \
            --disable_mip_filter --use_joint_4d_hash --use_intra_anchor_context \
            --use_entropy_coding --lmbda "${lmbda}" ${bg_flag} ${seed_flag} \
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
