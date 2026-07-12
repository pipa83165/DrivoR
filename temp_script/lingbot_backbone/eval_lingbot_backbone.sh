#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$REPO_ROOT/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REPO_ROOT/dataset}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CKPT_EXPERIMENT="${CKPT_EXPERIMENT:-lingbot_backbone_10ep}"
EVAL_SPLIT="${EVAL_SPLIT:-navtest}"
EXPERIMENT="${EXPERIMENT:-lingbot_backbone_${EVAL_SPLIT}}"
CKPT_PATH="${CKPT_PATH:-}"

if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(ls -t "${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}"/*/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)
fi

if [ -z "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found; set CKPT_PATH or CKPT_EXPERIMENT."
    exit 1
fi

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split="$EVAL_SPLIT" \
    agent=drivoR_lingbot \
    agent.checkpoint_path="$CKPT_PATH" \
    experiment_name="$EXPERIMENT" \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
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
    agent.config.vggt_geometry.enabled=false \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    "$@"
