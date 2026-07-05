#!/usr/bin/env bash
set -euo pipefail

# C1 validation entry. The C1 validation config is written here as Hydra overrides,
# following the README.md style, instead of using c1_training.yaml.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$REPO_ROOT/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REPO_ROOT/dataset}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

AGENT="${AGENT:-drivoR}"
MODE="${MODE:-normal}"                         # normal | shuffle | noise | drop
SEED="${SEED:-0}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-C1_${MODE}_val}"
TRAIN_TEST_SPLIT="${TRAIN_TEST_SPLIT:-navtrain}"
CACHE_PATH="${CACHE_PATH:-$NAVSIM_EXP_ROOT/navsim_cache_nommcv}"
export VGGT_GEOMETRY_CACHE_DIR="${VGGT_GEOMETRY_CACHE_DIR:-$REPO_ROOT/c1/vggt_geometry_tokens}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/c1/${EXPERIMENT_NAME}/\${experiment_uid}}"
CKPT_PATH="${CKPT_PATH:-}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BASE_LR="${BASE_LR:-0.0002}"
NUM_GPUS="${NUM_GPUS:-1}"

if [ -z "$CKPT_PATH" ]; then
  echo "ERROR: CKPT_PATH is required, for example:" >&2
  echo "  CKPT_PATH=/path/to/checkpoint.ckpt bash c1/validate_c1.sh" >&2
  exit 2
fi

python3 "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
  agent="$AGENT" \
  experiment_name="$EXPERIMENT_NAME" \
  output_dir="$OUTPUT_DIR" \
  train_test_split="$TRAIN_TEST_SPLIT" \
  cache_path="$CACHE_PATH" \
  use_cache_without_dataset=true \
  force_cache_computation=false \
  validation_run=true \
  train_ckpt_path="$CKPT_PATH" \
  seed="$SEED" \
  dataloader.params.batch_size="$BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  dataloader.params.pin_memory=true \
  dataloader.params.prefetch_factor="$PREFETCH_FACTOR" \
  trainer.params.max_epochs="$MAX_EPOCHS" \
  trainer.params.check_val_every_n_epoch=1 \
  trainer.params.val_check_interval=1.0 \
  trainer.params.limit_train_batches=1.0 \
  trainer.params.limit_val_batches=1.0 \
  trainer.params.accelerator=gpu \
  trainer.params.strategy=ddp \
  trainer.params.precision=16-mixed \
  trainer.params.num_nodes=1 \
  trainer.params.num_sanity_val_steps=0 \
  trainer.params.fast_dev_run=false \
  trainer.params.accumulate_grad_batches=1 \
  trainer.params.gradient_clip_val=0.0 \
  trainer.params.gradient_clip_algorithm=norm \
  trainer.params.default_root_dir="$OUTPUT_DIR" \
  agent.lr_args.base_lr="$BASE_LR" \
  agent.scheduler_args.num_epochs="$MAX_EPOCHS" \
  agent.scheduler_args.dataset_size=85000 \
  agent.num_gpus="$NUM_GPUS" \
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
  "$@"
