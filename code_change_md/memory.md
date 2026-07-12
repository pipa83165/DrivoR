# 项目记忆(handoff)— DrivoR × 冻结 VGGT-Ω / LingBot-Vision

> 只记录**已实现**的代码与约定。设计细节见 `code_change_md/design/`(统一格式:目标 / 接口设计 / 配置与运行)。
> 项目约定:不用 git 做版本管理/溯源,一律记文件内容 sha256(`vggt_geometry.file_sha256`)。

## 1. 这个 repo 是什么

NAVSIM devkit + DrivoR agent(DINOv2 ViT-S + LoRA + 每相机 16 scene token + 双 decoder)的 fork,在其上验证外部视觉/几何主干对端到端规划的增益。顶层 `vggt_omega/` 与 `lingbot_vision/` 是 vendored 的官方模型代码,不要改(py3.9 例外见 §7)。

四条实验路径:

| 实验名 | 定义 | 开关 | 设计文档 |
|---|---|---|---|
| **并联**(`paralle`) | VGGT-Ω register token 投影后**拼接**到 decoder memory(128 token) | `++agent.config.vggt_geometry.enabled=true` | `design/parallel.md` |
| **geo_only** | 几何 token **独占** memory(64 token),DINO 不构造 | 并联开关 + `++...geo_only=true ++...use_layerscale_gate=false` | `design/geo_only.md` |
| **vggtomega_backbone** | 冻结 VGGT-Ω 1B 替换 DINO 主干 + 可学习 scene token;可选 Q/V LoRA | `agent=drivoR_vggt_omega` | `design/vggtomega_backbone_implementation.md` |
| **lingbot_backbone** | DINOv2 换 LingBot-Vision ViT-S/16(替换而非拼接) | `agent=drivoR_lingbot` | `design/lingbot_vision_backbone.md` |

## 2. 代码地图

| 文件 | 职责 |
|---|---|
| `navsim/agents/drivoR/vggt_geometry.py` | 几何分支全部设施:projector、冻结教师、缓存 provider(shuffle/noise)、`preprocess_arrays_for_teacher`、指纹、`file_sha256` |
| `navsim/agents/drivoR/drivor_model.py` | 消费侧:构造 projector/教师,`_extend_memory_with_vggt_geometry` 拼 memory;geo_only 跳过 backbone;vggt_omega_1b 哨兵分发 |
| `navsim/agents/drivoR_vggt_omega/` | VGGT-Ω 主干替换:`SceneTokenAggregator`、`VggtOmegaImgEncoder`、LoRA、builder、agent |
| `navsim/agents/drivoR/layers/image_encoder/lingbot_vision_lora.py` | LingBot backbone adapter(`ImgEncoder` 按 `impl` 分派) |
| `navsim/planning/training/dataset.py` | 注入 `features["vggt_geometry_tokens"]`;构造期指纹校验;provider seed offset = `rank*num_workers+worker_id` |
| `run_training_full.py` / `run_pdm_score_multi_gpu.py` | 训练/评测入口,把 `agent.config.vggt_geometry` 传给 Dataset |
| `navsim/agents/drivoR/scripts/cache_vggt_geometry_tokens.py` | 离线缓存脚本(module 方式运行) |
| `config/common/agent/drivoR.yaml` | `vggt_geometry` 默认块(`enabled: false`)+ `image_backbone.impl: timm` |
| `temp_script/` | 全部运行入口 shell 脚本(§6) |

## 3. 几何分支数据流(cache 路径 = 默认)

1. **离线缓存**:教师(`VGGTOmega(enable_camera/depth/alignment=False)`)对 4 相机 jpg 走官方预处理,联合前向取 registers → 每样本 `(4, 16, 2048)` fp16;产物 = token `.pt`(按 `token[:2]` 分片)+ `metadata.json`(指纹)+ `token_index.json` + `noise_stats.pt`;写文件均原子替换,支持 shard/skip-existing。
2. **数据层**:构造时 `validate_fingerprint`(checkpoint sha256、维度、camera_order、预处理参数、`load_fn.py` sha256 等,不匹配抛错);`__getitem__` 按 mode 塞 token。
3. **模型**:projector 投影 → 拼到 `scene_features` 尾部;`output["vggt_geometry_memory_len"]`(并联 128,geo_only 64)。零初始化门控 ⇒ 并联冷启动等价基线。
4. `source=online`(延迟评测用):builder 生成 `vggt_teacher_images`,模型内教师现场前向;教师存 `self.__dict__`(不进 checkpoint)。

**四种 mode**:`normal` 本样本真实 token;`shuffle` 随机换成另一样本的(容量对照);`noise` 按 `noise_stats.pt` 生成噪声(位置对照);`drop` 训练拼接、eval 物理不拼接(测依赖度;geo_only 下构造期禁止)。

## 4. 关键配置

`vggt_geometry` 核心键:`mode / source / cache_dir / checkpoint_path(weights/vggt_omega_1b_512.pt) / vggt_dim=2048(=cat(frame,global)) / num_registers=16 / use_camera_token=false / joint_forward=true / preprocess_mode=balanced / image_resolution=512 / shuffle_seed=20260704 / force_ignore_fingerprint=false`。

geo_only 两键(`geo_only` / `use_layerscale_gate`)运行时 `++` 注入,不进缓存指纹;训练评测必须同值。

## 5. 硬约束(改代码前必读)

1. **融合点只在 decoder memory**,不得把 VGGT token 喂进 DrivoR ViT 主干。
2. **归一化恰好一次**:builder/缓存输出 [0,1],ImageNet 归一化只在 VGGT Aggregator 内部。
3. **预处理只有一份实现**:`preprocess_arrays_for_teacher`,禁止复刻数值逻辑。
4. **教师恒冻结**:eval() + requires_grad_(False) + no_grad;可训练只有 projector。
5. **drop 不置零**:eval 期物理移除 token。
6. **geo_only 不构造 DINO 分支**(DDP `find_unused_parameters=False` 约束);lidar+geo_only 构造期抛错。
7. **指纹不匹配 = 重新生成缓存**,不要开 `force_ignore_fingerprint` 蒙混。
8. `enabled=false` 时对基线零影响:唯一差异是 cfg 读取,无模块构造、无 import 副作用。

## 6. 运行(脚本在 `temp_script/`,面向训练服务器)

parallel 目录部分脚本硬编码服务器路径 `/high_perf_store3/world-model/weixiaobao/yzj/DrivoR`,其余用 `$REPO_ROOT`。

- **缓存生成**:`temp_script/cache/cache_vggt_geometry_tokens.sh`(trainval → `./vggtomega_geometry_tokens`);navtest 版 `cache_vggt_geometry_tokens_test.sh` → `./vggtomega_geometry_tokens_navtest`。
- **并联**:训练 `temp_script/parallel/train_paralle_{normal,shuffle,noise}.sh`;基线 `train_paralle_drivor.sh`;`train_paralle_drop.sh` **是评测**(读 normal checkpoint)不训练;评测 `eval_paralle_*.sh`。checkpoint 自动查找 `${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}/*/lightning_logs/version_*/checkpoints/last.ckpt`(注意 `ke/`),或显式 `CKPT_PATH`。
- **geo_only**:`temp_script/geo_only/train_geo_only_{normal,shuffle}.sh` / `eval_geo_only_*.sh`;环境变量 `EXPERIMENT / SEED / MAX_EPOCHS / BATCH_SIZE / NUM_GPUS / BASE_LR / VGGT_GEOMETRY_CACHE_DIR / EVAL_SPLIT / CKPT_PATH`。
- **vggtomega_backbone**:`temp_script/vggtomega_backbone/{train,eval}_vggtomega_backbone.sh`;LoRA 变体两侧追加 `agent.config.image_backbone.use_lora=true`。
- **lingbot_backbone**:`temp_script/lingbot_backbone/{train,eval}_lingbot_backbone.sh`。
- **检查脚本**:`temp_script/check_vggt_geometry_preprocess.py`(预处理正确性);`check_vggt_token_norms.py`(token 范数;2026-07-10 实测无高范数 outlier)。

## 7. 命名与例外

1. 并联的脚本/实验名统一用 `paralle`;训练缓存默认 `$REPO_ROOT/vggtomega_geometry_tokens`,评测缓存 `..._navtest`。
2. 权重目录是 `weights/`(服务器);指纹校验读该文件算 sha256,路径错在构造期抛 FileNotFoundError。
3. **py3.9 兼容例外(2026-07-12)**:vendored `lingbot_vision/vit.py`、`layers.py` 用了 PEP604 `float | None`,已在两文件顶部各加一行 `from __future__ import annotations`(不改运行时逻辑)。这是唯一破坏"vendored 与上游逐字节一致"的地方,从上游同步这两个文件时要重新补上。
