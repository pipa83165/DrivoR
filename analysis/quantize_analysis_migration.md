# vit_ar-Merged_sh/quantize analysis scripts migration notes

## Scope

Goal: analyze what must change to migrate useful data-analysis scripts from
`../vit_ar-Merged_sh/quantize/analysis` into this DrivoR repository.

Explicitly out of scope per request:

- `(dx, dy)` distribution / codebook work.
- `dv/dyaw` distribution, clustering, encode/decode demos, and related AR-token work.
- `(a, alpha)` / acceleration-alpha analysis and outlier inspection.
- Tokenizer export or tokenizer configuration.

Current-project facts used as authority:

- Project memory: `.codex/PROJECT_MEMORY.md`.
- Main training entry: `navsim/planning/script/run_training_full.py`.
- Main multi-GPU PDM scoring entry: `navsim/planning/script/run_pdm_score_multi_gpu.py`.
- Main agent config: `navsim/planning/script/config/common/agent/drivoR.yaml`.
- Current VGGT geometry config block is `agent.config.vggt_geometry`, not `c1_vggt`.
- Current image backbone path is `navsim.agents.drivoR.layers.image_encoder.dinov2_lora.ImgEncoder`.
- Current DrivoR target builder returns `trajectory` and `token`; it does not produce
  `trajectory_indices` or `history_trajectory_indices`.

## Source scripts to exclude

Do not migrate these as part of this request:

- `analyze_trajectory_distribution.py`: direct `(dx, dy)` analysis and tokenizer/codebook outputs.
- `analyze_vel_yaw_distribution.py`: `dv/dyaw/state` analysis and tokenizer-aligned outputs.
- `cluster_dv_dyaw.py`: `dv/dyaw` clustering.
- `demo_encode_decode_dv_dyaw_state.py`: tokenizer/encode-decode demo.
- `analyze_acc_alpha_distribution.py`: `(a, alpha)` analysis.
- `inspect_outliers.py`: depends on `analyze_acc_alpha_distribution.py` and dumps `a/alpha` outliers.
- `export_centers_to_tokenizer.py`: emits tokenizer center code.

## Source scripts worth migrating

These scripts are not inherently tied to excluded fields:

- `analyze_score_distribution.py`
- `evaluate_gt_trajectory.py`
- `visualize_low_score_cases.py`
- `visualize_vit_attention.py`
- `benchmark_backbone_latency.py`

Recommended destination: keep them under `analysis/` or `analysis/quantize_migration/`
instead of creating a top-level `quantize` package. The current repo is not a
quantization repo, and importing `quantize.utils.*` would drag in ModelOpt and
old AR-token assumptions.

## Common migration rules

1. Replace source-project Hydra assumptions.

   The source attention script uses `CONFIG_NAME = "default_training_fast"` and
   imports `quantize.utils.datasets`. This repo only has standard training
   configs such as `default_training.yaml`; project memory identifies
   `run_training_full.py` as the authoritative training entry. Analysis scripts
   should use `default_training` or an explicit `--config_path` / Hydra override.

2. Preserve current VGGT geometry data flow.

   If a migrated script runs model inference and VGGT geometry is enabled with
   `vggt_geometry.source=cache`, build data through the current
   `Dataset`/`CacheOnlyDataset` path and pass `vggt_geometry_cfg`. That is what
   `run_training_full.py` and `run_pdm_score_multi_gpu.py` do. Do not silently
   drop VGGT geometry tokens when cache tokens are missing; current dataset code
   intentionally raises on missing/invalid cache.

3. Remove AR-token injection.

   Source scripts sometimes attach `targets["trajectory_indices"]` or
   `targets["history_trajectory_indices"]` into features. Current
   `DrivoRTargetBuilder` does not emit those keys, and current `DrivoRModel`
   predicts continuous proposals. Remove that logic from migrated scripts.

4. Avoid importing `quantize.utils.config`.

   It imports `modelopt.torch.quantization` at module import time. None of the
   retained analysis scripts need ModelOpt.

5. Use current checkpoint loading semantics.

   Either set `cfg.agent.checkpoint_path = ckpt_path`, instantiate the agent,
   call `agent.initialize()`, and then `agent.eval()`, or implement a tiny local
   checkpoint loader that maps `agent._drivor_model.*` to `_drivor_model.*`.
   Do not depend on `quantize.utils.checkpoint`.

6. Keep camera-order assumptions explicit.

   The DrivoR image feature builder orders active cameras as front, back, left0,
   right0 under the default config. VGGT geometry cache metadata uses front,
   left0, right0, back. Do not reuse one order label list for both paths.

## Script-specific changes

### `analyze_score_distribution.py`

Porting effort: low.

What already matches:

- Current PDM score CSV columns are the same core columns:
  `no_at_fault_collisions`, `drivable_area_compliance`, `ego_progress`,
  `time_to_collision_within_bound`, `comfort`,
  `driving_direction_compliance`, and `score`.
- The script is model-independent and does not care about VGGT geometry.

Required changes:

- Drop pandas-generated index columns such as `Unnamed: 0` if present.
- If a `valid` column exists, filter to valid rows for statistics by default.
- Exclude `token == "average"` from per-scene statistics and low-score token
  extraction.
- Add a clear error for missing score columns instead of a raw `KeyError`.
- Keep the low-score token output format; it can feed low-score visualization.

Optional improvement:

- Add `--include_invalid` and `--include_average` flags for debugging.

### `evaluate_gt_trajectory.py`

Porting effort: low to medium.

What already matches:

- It uses current NAVSIM scoring primitives: `MetricCacheLoader`, `pdm_score`,
  `PDMSimulator`, and `PDMScorer`.
- It is model-independent, so no VGGT-geometry-specific model changes are needed.

Required changes:

- Replace the hard-coded `navtest.yaml` load with `--split` or a Hydra-style
  config selection. Default can stay `navtest`, but the script should accept
  `navtrain`, `navmini`, etc.
- Keep default scoring config at
  `navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml`.
- Use current `MetricCacheLoader.get_from_token(token)` instead of manually
  opening `metric_cache_loader.metric_cache_paths[token]`; this matches current
  dataloader API and avoids duplicating lzma/pickle logic.
- Make overlap ordering deterministic, e.g. preserve scene-loader token order
  and filter by metric-cache membership instead of `list(set(...))`.
- Add `valid == True` filtering before the average row.

Validation:

- Run on a tiny subset, e.g. `--num_scenes 2`, because it needs metric cache and
  map/scoring dependencies.

### `visualize_low_score_cases.py`

Porting effort: medium to high.

Source-script incompatibilities:

- It contains AR decoder target injection for `trajectory_indices` and
  `history_trajectory_indices`; remove this.
- It manually builds feature batches, so it bypasses current Dataset logic that
  attaches cached `vggt_geometry_tokens`.
- It selects `predictions["proposals"][:, 0]`, but current model computes
  `predictions["trajectory"]` by choosing a proposal with its scorer. For
  low-score diagnostics, visualize `predictions["trajectory"]` by default.

Required changes:

- Build the agent from current config and checkpoint:
  `cfg.agent.checkpoint_path = ckpt_path`, instantiate, call `initialize()`,
  move to device, and eval.
- Build `SceneLoader` from the current split scene filter, then restrict
  `scene_filter.tokens` to the low-score token list.
- For model inference, prefer current `Dataset(..., append_token_to_batch=True,
  vggt_geometry_cfg=OmegaConf.select(cfg, "agent.config.vggt_geometry"))` plus
  a `DataLoader`. This keeps VGGT geometry normal/shuffle/noise/drop data behavior aligned
  with `run_pdm_score_multi_gpu.py`.
- Use `AgentLightningModule.predict_step` or equivalent direct batched
  inference over Dataset batches. If doing direct inference, carry token order
  from the Dataset and keep non-tensor batch fields intact.
- Keep `scene_loader.get_scene_from_token(token)` for BEV map and GT drawing.
- Preserve the existing BEV drawing functions; they exist in current
  `navsim.visualization`.

Recommended CLI:

```bash
python analysis/visualize_low_score_cases.py \
  --token_list analysis_output/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/last.ckpt \
  --config_path navsim/planning/script/config/training/default_training.yaml \
  --split navtest \
  --data_path ./dataset/navsim_logs/test \
  --sensor_blobs_path ./dataset/sensor_blobs/test \
  --metric_cache_path ./exp/metric_cache \
  --output_dir ./analysis_output/low_score_viz \
  --num_scenes 20
```

The exact dataset paths should follow the local `OPENSCENE_DATA_ROOT` layout
used by the current experiment scripts.

### `visualize_vit_attention.py`

Porting effort: high.

Source-script incompatibilities:

- Imports `quantize.utils.checkpoint`, `quantize.utils.model_utils`,
  `quantize.utils.datasets`, and `quantize.utils.config`.
- Uses `CONFIG_NAME = "default_training_fast"`, which this repo does not have.
- Contains AR `trajectory_indices` attachment.
- Some token-index comments assume a prefix layout that does not match current
  DINOv2-reg + scene-token ordering.

Required changes:

- Switch to current `CONFIG_NAME = "default_training"` or make the training
  config explicit via CLI/Hydra.
- Replace `build_supervised_dataloaders` with local logic copied from current
  `run_training_full.py`, including `vggt_geometry_cfg` support.
- Replace checkpoint loading with local helper or `agent.initialize()`.
- Replace LoRA merge import with a local helper. Current LoRA wrapper has the
  same `w_As`/`w_Bs`/`lora_vit` structure, but the import path is current
  `navsim.agents.drivoR.layers.image_encoder.dinov2_lora`.
- Remove AR target-index logic.
- When registering hooks, resolve the real ViT:
  `vit = agent._drivor_model.image_backbone.model`; if LoRA is not merged and
  `hasattr(vit, "lora_vit")`, hook `vit.lora_vit.blocks`. If LoRA is merged,
  hook `vit.blocks`.
- Compute prefix layout from the actual ViT sequence:
  `num_patches = grid_h * grid_w`, `prefix_len = N - num_patches`.
  With the current default, prefix is `scene tokens + cls + 4 reg`.
- Do not assume first patch index equals `num_scene_tokens`; first patch is
  `prefix_len`. Scene-token queries are `0 .. num_scene_tokens-1`.
- Camera labels must follow DrivoR image feature order, not VGGT geometry cache order.

Keep vs remove:

- Keep scene-token attention and learnable scene-token embedding similarity:
  these are not tokenizer-related.
- Remove tokenizer/AR examples from shell wrappers and docstrings.

### `benchmark_backbone_latency.py`

Porting effort: medium.

Source-script incompatibilities:

- Imports `ImgEncoder` from old path `navsim.agents.drivoR.dinov2_lora`.
- Imports LoRA merge from `quantize.utils.model_utils`.
- Hard-codes a minimal config dataclass instead of reading current
  `drivoR.yaml`.
- The pruning sweep mutates the same pruned model repeatedly across
  `scene_tokens` values, so later measurements are not clean.

Required changes:

- Import `ImgEncoder` from
  `navsim.agents.drivoR.layers.image_encoder.dinov2_lora`.
- Add `--agent-config navsim/planning/script/config/common/agent/drivoR.yaml`
  and build the image-backbone config from `cfg.config.image_backbone` plus
  `image_size`, `num_scene_tokens`, and `tf_d_model`.
- Replace `quantize.utils.model_utils.merge_lora_to_backbone` with a local
  helper or put the helper in an analysis-only common module.
- Rebuild the pruned model for each `num_scene_tokens`, or apply pruning once
  per model, not cumulatively inside the sweep.
- State clearly that this is image-backbone latency only. It does not measure
  online VGGT teacher cost. If online teacher latency matters, write a separate
  full-agent benchmark with `vggt_geometry.source=online`.

## Suggested helper module

If implementing the migration, create one small helper module rather than
copying helper code into every script:

`analysis/drivor_analysis_utils.py`

Responsibilities:

- Load Hydra/OmegaConf config from current training config.
- Instantiate and initialize `DrivoRAgent` from checkpoint.
- Build token-filtered `SceneLoader`.
- Build current `Dataset`/`CacheOnlyDataset` with `vggt_geometry_cfg`.
- Move nested batches to device.
- Optionally merge LoRA into the current image backbone.

Do not put ModelOpt or quantization helpers in this module.

## Minimal verification matrix

After migration, run these smoke tests:

- Score CSV: run `analyze_score_distribution.py` on one current PDM CSV and
  confirm average/invalid rows are excluded.
- GT scoring: run `evaluate_gt_trajectory.py --num_scenes 2` with a known
  metric cache.
- Low-score BEV: run one or two tokens with VGGT geometry disabled, then one token with
  `vggt_geometry.enabled=true` and cache mode if the cache is available.
- Attention: run batch size 1, one layer, one head, one or two query indices.
- Latency: run CPU or GPU with `--iters 5 --warmup 1` to verify imports and
  shapes before running long benchmarks.

Recommended syntax check once files exist:

```bash
python3 -m py_compile \
  analysis/analyze_score_distribution.py \
  analysis/evaluate_gt_trajectory.py \
  analysis/visualize_low_score_cases.py \
  analysis/visualize_vit_attention.py \
  analysis/benchmark_backbone_latency.py \
  analysis/drivor_analysis_utils.py
```

## Bottom line

The score and GT-evaluation scripts are mostly reusable. Low-score
visualization, ViT attention visualization, and backbone latency need real
adaptation because the source project expected a `quantize` package,
`default_training_fast`, AR/tokenizer targets, and older import paths. In this
repo, any inference-oriented analysis must preserve the current Dataset path so
VGGT geometry tokens and controls behave exactly like training/evaluation.
