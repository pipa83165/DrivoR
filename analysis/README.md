# DrivoR Analysis Scripts Usage

This directory contains analysis scripts adapted from `vit_ar-Merged_sh/quantize/analysis`
for the current DrivoR repository. The migrated set intentionally excludes
`dxdy`, `dvdyaw`, `a/alpha`, and tokenizer-related analysis.

## Environment

Run commands from the repository root:

```bash
cd /high_perf_store3/world-model/weixiaobao/yzj/DrivoR
```

Recommended environment variables:

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NAVSIM_EXP_ROOT="./exp"
export NAVSIM_DEVKIT_ROOT="./"
export OPENSCENE_DATA_ROOT="./dataset"
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
```

If you use VGGT geometry cache mode, pass the same Hydra overrides used for
training/evaluation, especially:

```bash
agent.config.vggt_geometry.enabled=true
agent.config.vggt_geometry.source=cache
agent.config.vggt_geometry.cache_dir=c1/vggtomega_geometry_tokens
```

## 1. Score CSV Distribution

Analyze a PDM score CSV and export bottom-score tokens.

```bash
python3 analysis/analyze_score_distribution.py \
  --csv_path /path/to/pdm_scores.csv \
  --output_dir analysis_output/score_analysis \
  --low_score_ratio 0.3
```

Outputs:

- `score_statistics.csv`
- `score_histograms.png`
- `score_boxplots.png`
- `score_correlation.png`
- `low_score_tokens_bottom30pct.txt`
- `low_score_cases_bottom30pct.csv`

Default behavior:

- Drops pandas `Unnamed:*` index columns.
- Excludes `valid=False` rows when a `valid` column exists.
- Excludes the summary row where `token == "average"`.

Use `--include_invalid` or `--include_average` only for debugging.

## 2. Low-Score BEV Visualization

Visualize model prediction versus GT trajectory for tokens exported by score
analysis.

```bash
python3 analysis/visualize_low_score_cases.py \
  --token_list analysis_output/score_analysis/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/lightning_logs/version_0/checkpoints/last.ckpt \
  --config_path navsim/planning/script/config/training/default_training.yaml \
  --split navtest \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --output_dir analysis_output/low_score_viz \
  --num_scenes 20 \
  --batch_size 1
```

By default it visualizes `predictions["trajectory"]`, i.e. the model-selected
trajectory. To draw a specific proposal instead:

```bash
python3 analysis/visualize_low_score_cases.py \
  --token_list analysis_output/score_analysis/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/last.ckpt \
  --proposal_index 0
```

For VGGT geometry cache mode, add repeatable Hydra overrides:

```bash
python3 analysis/visualize_low_score_cases.py \
  --token_list analysis_output/score_analysis/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/last.ckpt \
  --hydra_override agent.config.vggt_geometry.enabled=true \
  --hydra_override agent.config.vggt_geometry.source=cache \
  --hydra_override agent.config.vggt_geometry.cache_dir=c1/vggtomega_geometry_tokens
```

## 3. GT Trajectory PDM Score

Evaluate human GT trajectories through the same PDM scoring stack.

```bash
python3 analysis/evaluate_gt_trajectory.py \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache" \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --split navtest \
  --num_scenes 20 \
  --output_dir analysis_output/gt_score_analysis
```

Output:

- `gt_pdm_scores.csv`

For a quick smoke test, use `--num_scenes 2`.

## 4. ViT Attention Visualization

Visualize DrivoR image-backbone attention maps. This script uses current
`default_training.yaml`, current Dataset construction, and current image encoder
path. It does not use `quantize.utils` or AR/tokenizer targets.

```bash
python3 analysis/visualize_vit_attention.py \
  --ckpt_path /path/to/lightning_logs/version_0/checkpoints/last.ckpt \
  --config_path navsim/planning/script/config/training/default_training.yaml \
  --split navtrain \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/trainval" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/trainval" \
  --output_dir analysis_output/attention_viz \
  --batch_size 1 \
  --max_scenes 1 \
  --layers 11 \
  --heads 0 \
  --queries 0
```

Notes:

- `--queries` uses prefix-token indices. Scene tokens are normally
  `0..num_scene_tokens-1`.
- First patch token is inferred from the actual ViT sequence length; do not
  assume it equals `num_scene_tokens`.
- `--merge_lora` merges LoRA adapters before hook registration if you want to
  inspect the merged base ViT path.

## 5. Image Backbone Latency

Benchmark only the DrivoR `ImgEncoder`. This does not include full-agent
decoding or online VGGT teacher cost.

```bash
python3 analysis/benchmark_backbone_latency.py \
  --agent-config navsim/planning/script/config/common/agent/drivoR.yaml \
  --device cuda \
  --batch-size 1 \
  --num-cams 4 \
  --scene-tokens 1 4 8 16 32 64 \
  --iters 100 \
  --warmup 10
```

Quick smoke test:

```bash
python3 analysis/benchmark_backbone_latency.py \
  --device cpu \
  --scene-tokens 16 \
  --iters 5 \
  --warmup 1
```

Optional pruning benchmark:

```bash
python3 analysis/benchmark_backbone_latency.py \
  --device cuda \
  --scene-tokens 16 \
  --prune-last-blocks 2
```

## Verification

Syntax check:

```bash
python3 -m py_compile \
  analysis/drivor_analysis_utils.py \
  analysis/analyze_score_distribution.py \
  analysis/evaluate_gt_trajectory.py \
  analysis/visualize_low_score_cases.py \
  analysis/visualize_vit_attention.py \
  analysis/benchmark_backbone_latency.py
```

Static scope check:

```bash
grep -R "from quantize\\|import quantize\\|trajectory_indices\\|history_trajectory_indices\\|tokenizer\\|dvdyaw\\|dv_dyaw\\|dxdy\\|acc_alpha" -n analysis/*.py
```

Expected result: no matches, except explanatory text if you intentionally grep
Markdown files.
