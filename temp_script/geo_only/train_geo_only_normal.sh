#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=normal EXPERIMENT="${EXPERIMENT:-geo_only_normal_${MAX_EPOCHS:-10}ep}" exec "$SCRIPT_DIR/train_geo_only.sh" "$@"
