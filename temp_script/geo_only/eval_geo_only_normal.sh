#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=normal CKPT_EXPERIMENT="${CKPT_EXPERIMENT:-geo_only_normal_10ep}" EXPERIMENT="${EXPERIMENT:-geo_only_normal_${EVAL_SPLIT:-navtest}}" exec "$SCRIPT_DIR/eval_geo_only.sh" "$@"
