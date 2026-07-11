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
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH="${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT}"
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODE="${MODE:-normal}" # normal | shuffle | noise; drop is invalid for geo_only
if [ "$MODE" = "drop" ]; then
    echo "Error: geo_only forbids MODE=drop because decoder memory would be empty."
    exit 1
fi

VGGT_GEOMETRY_CACHE_DIR="${VGGT_GEOMETRY_CACHE_DIR:-$REPO_ROOT/vggtomega_geometry_tokens}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_GPUS="${NUM_GPUS:-4}"
BASE_LR="${BASE_LR:-0.0002}"
SEED="${SEED:-2}"
EXPERIMENT="${EXPERIMENT:-geo_only_${MODE}_${MAX_EPOCHS}ep}"
AGENT=drivoR

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs="$MAX_EPOCHS" \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size="$BATCH_SIZE" \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr="$BASE_LR" \
    agent.num_gpus="$NUM_GPUS" \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    ++agent.config.vggt_geometry.enabled=true \
    ++agent.config.vggt_geometry.mode="$MODE" \
    ++agent.config.vggt_geometry.source=cache \
    ++agent.config.vggt_geometry.cache_dir="$VGGT_GEOMETRY_CACHE_DIR" \
    ++agent.config.vggt_geometry.checkpoint_path=weights/vggt_omega_1b_512.pt \
    ++agent.config.vggt_geometry.vggt_dim=2048 \
    ++agent.config.vggt_geometry.num_registers=16 \
    ++agent.config.vggt_geometry.use_camera_token=false \
    ++agent.config.vggt_geometry.joint_forward=true \
    ++agent.config.vggt_geometry.preprocess_mode=balanced \
    ++agent.config.vggt_geometry.image_resolution=512 \
    ++agent.config.vggt_geometry.shuffle_seed=20260704 \
    ++agent.config.vggt_geometry.force_ignore_fingerprint=false \
    ++agent.config.vggt_geometry.geo_only=true \
    ++agent.config.vggt_geometry.use_layerscale_gate=false \
    seed="$SEED" \
    "$@"
