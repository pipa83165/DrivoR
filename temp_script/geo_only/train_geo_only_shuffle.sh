#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=shuffle EXPERIMENT="${EXPERIMENT:-geo_only_shuffle_${MAX_EPOCHS:-10}ep}" exec "$SCRIPT_DIR/train_geo_only.sh" "$@"
