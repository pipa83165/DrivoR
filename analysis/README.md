# DrivoR 分析脚本使用说明


公共辅助函数集中在 `drivor_analysis_utils.py`（配置组合、agent 实例化、SceneLoader/Dataset
构建、PDM simulator/scorer 加载、LoRA 合并等）。`analyze_score_distribution.py` 与
`compare_scene_scores.py` 是纯 pandas 脚本，不依赖 torch/navsim，可在任意机器上单独运行。

## 环境准备

所有命令都在仓库根目录下执行：

```bash
cd /high_perf_store3/world-model/weixiaobao/yzj/DrivoR
```

推荐设置以下环境变量：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="./dataset/maps"
export NAVSIM_EXP_ROOT="./exp"
export NAVSIM_DEVKIT_ROOT="./"
export OPENSCENE_DATA_ROOT="./dataset"
export HYDRA_FULL_ERROR=1
export SUBSCORE_PATH="$NAVSIM_EXP_ROOT"
```

如果使用 VGGT 几何缓存模式，请传入与训练/评测时相同的 Hydra 覆盖项，尤其是：

```bash
agent.config.vggt_geometry.enabled=true
agent.config.vggt_geometry.source=cache
agent.config.vggt_geometry.cache_dir=c1/vggtomega_geometry_tokens
```

## 1. 分数 CSV 分布分析

分析 PDM 分数 CSV，并导出低分场景 token 列表。

```bash
python3 analysis/analyze_score_distribution.py \
  --csv_path /path/to/pdm_scores.csv \
  --output_dir analysis_output/score_analysis \
  --low_score_ratio 0.3
```

输出文件：

- `score_statistics.csv`：各指标的统计量
- `score_histograms.png`：各子分数直方图
- `score_boxplots.png`：子分数箱线图
- `score_correlation.png`：分数相关性热力图
- `low_score_tokens_bottom30pct.txt`：低分场景 token 列表
- `low_score_cases_bottom30pct.csv`：低分场景明细

默认行为：

- 自动丢弃 pandas 的 `Unnamed:*` 索引列。
- 若存在 `valid` 列，剔除 `valid=False` 的行。
- 剔除 `token == "average"` 的汇总行。

`--include_invalid` 和 `--include_average` 仅用于调试。

## 2. 低分场景 BEV 可视化

对分数分析导出的 token，可视化模型预测轨迹与 GT 轨迹的对比。

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

默认可视化 `predictions["trajectory"]`，即模型最终选中的轨迹。如需画某一条指定
proposal：

```bash
python3 analysis/visualize_low_score_cases.py \
  --token_list analysis_output/score_analysis/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/last.ckpt \
  --proposal_index 0
```

VGGT 几何缓存模式下追加可重复的 Hydra 覆盖项：

```bash
python3 analysis/visualize_low_score_cases.py \
  --token_list analysis_output/score_analysis/low_score_tokens_bottom30pct.txt \
  --ckpt_path /path/to/last.ckpt \
  --hydra_override agent.config.vggt_geometry.enabled=true \
  --hydra_override agent.config.vggt_geometry.source=cache \
  --hydra_override agent.config.vggt_geometry.cache_dir=c1/vggtomega_geometry_tokens
```

## 3. GT 轨迹 PDM 评分

将人类 GT 轨迹送入同一套 PDM 评分流程打分。

```bash
python3 analysis/evaluate_gt_trajectory.py \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache" \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --split navtest \
  --num_scenes 20 \
  --output_dir analysis_output/gt_score_analysis
```

输出：

- `gt_pdm_scores.csv`

快速冒烟测试可用 `--num_scenes 2`。

## 4. Proposal 生成 vs 排序诊断

导出最终解码器每条 proposal 的真实 PDM 质量，以及 DrivoR 学习到的 scorer
实际选中的 proposal：

```bash
python3 analysis/export_proposal_diagnostics.py \
  --ckpt_path /path/to/last.ckpt \
  --metric_cache_path "$NAVSIM_EXP_ROOT/metric_cache" \
  --split navtest \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --output_dir analysis_output/baseline_proposal_diagnostics
```

评测带几何分支的 checkpoint 时，传入与评测阶段相同的可重复 `--hydra_override`。
脚本走当前 Dataset 路径，包含几何缓存注入。

输出文件：

- `proposal_diagnostics.csv`：每个场景一行，含选中分数、oracle 分数、排序
  regret、hit@1/hit@5，以及选中/oracle 的各子分数。
- `proposal_scores.csv`：每条 proposal 一行，含预测分数、真实 PDM 分数、
  各子分数，以及是否被选中/是否为 oracle 的标记。

定义：

```text
oracle_score = max(每条 proposal 的真实 PDM 分数)
ranking_regret = oracle_score - selected_score
```

proposal 真实分数使用 `navsim.evaluate.pdm_score` 和标准评分配置计算。每条
proposal 独立地对照缓存的 PDM 参考轨迹评测，与常规单轨迹评测语义一致；脚本
不依赖仅训练期可用的 `metric_cache.pdm_progress`。

## 5. 场景级配对差值与切片分析

对比两份普通 PDM 分数 CSV，或两份 `proposal_diagnostics.csv`。输入按 token
对齐；invalid/average 行会被剔除，未匹配的 token 会单独报告。

```bash
python3 analysis/compare_scene_scores.py \
  --baseline_csv analysis_output/baseline/proposal_diagnostics.csv \
  --variant_csv analysis_output/geometry/proposal_diagnostics.csv \
  --baseline_name drivor \
  --variant_name geometry_normal \
  --output_dir analysis_output/drivor_vs_geometry
```

对 proposal 诊断输入，脚本会逐场景校验以下恒等式：

```text
delta_selected = delta_oracle + (baseline_regret - variant_regret)
```

输出包括配对场景明细、缺失 token 列表、差值分位数、胜/平/负比例、bootstrap
置信区间、ECDF/直方图，以及最大提升/最大回退的场景列表。

切片对比前，先提取与模型无关的场景属性：

```bash
python3 analysis/extract_scene_attributes.py \
  --data_path "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensor_blobs_path "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --split navtest \
  --output_csv analysis_output/navtest_scene_attributes.csv

python3 analysis/compare_scene_scores.py \
  --baseline_csv analysis_output/baseline/proposal_diagnostics.csv \
  --variant_csv analysis_output/geometry/proposal_diagnostics.csv \
  --attributes_csv analysis_output/navtest_scene_attributes.csv \
  --min_slice_size 30 \
  --output_dir analysis_output/drivor_vs_geometry_sliced
```

默认切片覆盖：地图、机动类型、速度、未来路径几何复杂度、路口、红绿灯状态、
周边 agent 密度/距离、弱势道路使用者(VRU)。切片定义使用固定阈值，不依赖任何
模型分数。

## 6. ViT 注意力可视化

可视化 DrivoR 图像 backbone 的注意力图。脚本使用当前的
`default_training.yaml`、当前 Dataset 构建方式和当前图像编码器路径，不依赖
`quantize.utils` 或 AR/tokenizer 目标。

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

注意事项：

- `--queries` 使用前缀 token 索引。scene token 通常是
  `0..num_scene_tokens-1`。
- 第一个 patch token 的位置由实际 ViT 序列长度推断，不要假设它等于
  `num_scene_tokens`。
- `--merge_lora` 会在注册 hook 之前把 LoRA 适配器合并进基础 ViT，用于观察
  合并后的 ViT 路径。

## 7. 图像 Backbone 延迟基准

只测 DrivoR 的 `ImgEncoder`，不包含完整 agent 解码或在线 VGGT teacher 的开销。

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

快速冒烟测试：

```bash
python3 analysis/benchmark_backbone_latency.py \
  --device cpu \
  --scene-tokens 16 \
  --iters 5 \
  --warmup 1
```

可选的剪枝基准：

```bash
python3 analysis/benchmark_backbone_latency.py \
  --device cuda \
  --scene-tokens 16 \
  --prune-last-blocks 2
```

## 验证

语法检查：

```bash
python3 -m py_compile \
  analysis/drivor_analysis_utils.py \
  analysis/analyze_score_distribution.py \
  analysis/evaluate_gt_trajectory.py \
  analysis/export_proposal_diagnostics.py \
  analysis/compare_scene_scores.py \
  analysis/extract_scene_attributes.py \
  analysis/visualize_low_score_cases.py \
  analysis/visualize_vit_attention.py \
  analysis/benchmark_backbone_latency.py

python3 -m unittest analysis/test_scene_score_analysis.py
```

静态范围检查：

```bash
grep -R "from quantize\\|import quantize\\|trajectory_indices\\|history_trajectory_indices\\|tokenizer\\|dvdyaw\\|dv_dyaw\\|dxdy\\|acc_alpha" -n analysis/*.py
```

预期结果：无匹配（若有意 grep Markdown 文件，说明性文字除外）。
