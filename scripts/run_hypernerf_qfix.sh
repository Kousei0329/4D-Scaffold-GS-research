#!/usr/bin/env bash
# HyperNeRF recipe after the HAC++ quantization-noise fix (commit 46425bf), validated on vrig-chicken.
# Same recipe as the Technicolor runs:
#   - quantization fix: on by default in the code (do NOT pass --legacy_quant_noise)
#   - --q_scaling_base 1e-4: anchor scalings here are ~1e-3..4e-3, so the HAC default 1e-3
#     (adaptive range up to 2e-3) rounded 20-40% of the xyz scalings (and ~97% of offsets) to 0
#   - densification 0.0002 / 0.1 (default Scaffold), max_init_t 1.0, update_until 25k (best legacy config)
#
# Usage:
#   bash scripts/run_hypernerf_qfix.sh <gpu_id> [scene ...]      # no scene -> all under data/HyperNeRF
#   LAMBDAS="0.01 0.005" bash scripts/run_hypernerf_qfix.sh 0 vrig-3dprinter
# Output: outputs/HyperNeRF/<scene>/baseline_pccolmap_mit1.0_uu25k_qfix_qs1e-4/lmbda_<lambda>/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:?usage: run_hypernerf_qfix.sh <gpu_id> [scene ...]}"; shift
GRAD_TH=0.0002 PHOTO_TH=0.1 MAX_INIT_T=1.0 LAMBDAS="${LAMBDAS:-0.01}" \
EXTRA="--update_until 25000 --q_scaling_base 0.0001" EXTRA_TAG=uu25k_qfix_qs1e-4 \
exec bash "${SCRIPT_DIR}/train_hypernerf.sh" "${GPU}" "$@"
