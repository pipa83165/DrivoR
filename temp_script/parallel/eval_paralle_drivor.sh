#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NAVSIM_EXP_ROOT="/high_perf_store3/world-model/weixiaobao/yzj/DrivoR/exp" 
export NAVSIM_DEVKIT_ROOT="/high_perf_store3/world-model/weixiaobao/yzj/DrivoR" 
export OPENSCENE_DATA_ROOT="./dataset"
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
export RAY_DEDUP_LOGS=0

export CUDA_VISIBLE_DEVICES=0,1,2,3



CKPT_EXPERIMENT="${CKPT_EXPERIMENT:-paralle_drivoR_10ep}"
EVAL_SPLIT="${EVAL_SPLIT:-navtest}"
EXPERIMENT="${EXPERIMENT:-paralle_drivoR_${EVAL_SPLIT}}"
CKPT_PATH="${CKPT_PATH:-}"
AGENT=drivoR

if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(ls -t "${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}"/*/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)
fi

if [ -z "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found under ${NAVSIM_EXP_ROOT}/${CKPT_EXPERIMENT}/"
    echo "Set CKPT_PATH=/path/to/last.ckpt or CKPT_EXPERIMENT=<experiment_name>."
    exit 1
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split="$EVAL_SPLIT" \
    agent=$AGENT \
    agent.checkpoint_path="$CKPT_PATH" \
    experiment_name=$EXPERIMENT \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.long_trajectory_additional_poses=2 \
    ++trainer.params.logger=false \
    ++trainer.params.enable_checkpointing=false \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    "$@"
