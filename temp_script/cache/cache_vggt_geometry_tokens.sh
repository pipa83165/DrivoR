SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "$0")"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NAVSIM_EXP_ROOT="./exp" 
export NAVSIM_DEVKIT_ROOT="./" 
export OPENSCENE_DATA_ROOT="./dataset"
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH=$NAVSIM_EXP_ROOT
export RAY_DEDUP_LOGS=0

python -m navsim.agents.drivoR.scripts.cache_vggt_geometry_tokens   \
    --checkpoint weights/vggt_omega_1b_512.pt   \
    --output-dir ./vggtomega_geometry_tokens   \
    --split trainval   \
    --forward-mode joint   \
    --preprocess-mode balanced