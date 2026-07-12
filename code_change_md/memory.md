# 项目记忆(handoff)— DrivoR × 冻结 VGGT-Ω 几何分支

> 只记录**已实现**的代码设计与实现。生成日期 2026-07-10。
> 项目约定:不用 git 做版本管理/溯源,一律记文件内容 sha256(`vggt_geometry.file_sha256`)。

## 1. 这个 repo 是什么

NAVSIM devkit + DrivoR agent(DINOv2 ViT-S + LoRA + 每相机 16 scene token + 双 decoder)的 fork。在其上验证冻结 VGGT-Ω 1B 的几何 token 对端到端规划的增益。仓库顶层 `vggt_omega/` 是 vendored 的官方 VGGT-Ω 模型代码(`models/aggregator.py`、`utils/load_fn.py` 等),不要改。

已实现三条实验路径:

| 实验名 | 定义 | 开关 |
|---|---|---|
| **并联**(`paralle`) | 冻结 VGGT-Ω register token 投影后**拼接**到 DrivoR decoder memory(64 DINO + 64 几何 = 128 token) | `++agent.config.vggt_geometry.enabled=true` |
| **geo_only** | 几何 token **独占** decoder memory(64 token),DINO 主干不构造不前向 | 并联开关 + `++agent.config.vggt_geometry.geo_only=true ++agent.config.vggt_geometry.use_layerscale_gate=false` |
| **vggtomega_backbone** | 冻结 VGGT-Ω 1B 替换 DINO 主干,注入可学习 scene token;可选 Q/V-only LoRA | `agent=drivoR_vggt_omega` |

## 2. 代码地图

| 文件 | 职责 |
|---|---|
| `navsim/agents/drivoR/vggt_geometry.py` | 全部几何分支设施:`VggtGeometryProjector`(投影)、`FrozenVggtGeometryTeacher`(冻结教师)、`VggtGeometryTokenProvider`(缓存读取 + shuffle/noise)、`preprocess_arrays_for_teacher`(官方预处理桥接)、指纹构建/校验、`file_sha256` |
| `navsim/agents/drivoR/drivor_model.py` | 消费侧:构造 projector/教师,`_extend_memory_with_vggt_geometry` 在 decoder memory 处拼接;`geo_only` 时跳过 image backbone 构造、forward 从空 memory 起手 |
| `navsim/agents/drivoR_vggt_omega/` | VGGT-Ω 主干替换:`SceneTokenAggregator`、`VggtOmegaImgEncoder`、Q/V-only LoRA、官方预处理 builder 与 `DrivoRVggtOmegaAgent` |
| `navsim/planning/training/dataset.py` | 数据层注入 `features["vggt_geometry_tokens"]`(`Dataset` 与 `CacheOnlyDataset` 都接 `vggt_geometry_cfg`);构造期做缓存指纹校验;provider 按 `rank*num_workers+worker_id` 做 seed offset |
| `navsim/planning/script/run_training_full.py` | 训练入口,`OmegaConf.select(cfg, "agent.config.vggt_geometry")` 传给 Dataset |
| `navsim/planning/script/run_pdm_score_multi_gpu.py` | 评测入口(navtest 缓存评测),同样把 `vggt_geometry` 传给 Dataset |
| `navsim/agents/drivoR/drivor_features.py:130-144` | 仅 `source=online` 时生成 `features["vggt_teacher_images"]` |
| `navsim/agents/drivoR/scripts/cache_vggt_geometry_tokens.py` | 离线缓存脚本(module 方式运行) |
| `navsim/planning/script/config/common/agent/drivoR.yaml` | `vggt_geometry` 默认配置块(`enabled: false`,基线零影响) |
| `temp_script/` | 全部运行入口 shell 脚本(见 §6) |
| `code_change_md/design/` | 设计文档:`parallel.md`、`geo_only.md`、`vggtomega_backbone.md` |

## 3. 数据流(cache 路径,训练与评测的默认路径)

1. **离线缓存**:`cache_vggt_geometry_tokens.py` 用 `FrozenVggtGeometryTeacher`(`VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)`,strict=False 加载但强制 aggregator 键齐全)对每个样本 4 相机 jpg 走官方 `load_and_preprocess_images`(balanced/512/patch16),4 帧**联合前向**取 `camera_and_register_tokens`,默认去掉 camera token → 每样本 `(4, 16, 2048)` fp16,按 `token[:2]/token.pt` 分片存储。产物:token `.pt` + `metadata.json`(指纹)+ `token_index.json` + `noise_stats.pt`(全体 per-dim mean/std,Welford 累计)。支持 `--shard-index/--num-shards/--skip-existing/--no-finalize`,写文件均为原子替换。
2. **数据层**:Dataset 构造时 `validate_fingerprint`(严格键:checkpoint sha256、vggt_dim、num_registers、camera_order、预处理参数、`load_fn.py` sha256、joint_forward、use_camera_token、cache_dtype;不匹配直接抛错,`force_ignore_fingerprint=true` 只降级为 warning)。`__getitem__` 里 provider 按 mode 给出 token,塞进 `features["vggt_geometry_tokens"]`;shape/dtype 不符抛错。
3. **模型**:`VggtGeometryProjector` = `LayerNorm(2048) → Linear(2048→256) → +branch_embed +cam_embed → LayerNorm → LayerScale(init=0)`。零初始化门控 ⇒ 并联冷启动时几何贡献严格为 0(等价基线);geo_only 用 `use_layerscale_gate=false` 换成 `nn.Identity`。投影输出 reshape 成 `(B, 64, 256)` 拼到 `scene_features` 尾部;`output["vggt_geometry_memory_len"]` 记录 memory 长度(并联 128,geo_only 64)。

`source=online` 路径(延迟评测用):builder 生成 `vggt_teacher_images`,模型内冻结教师现场前向;教师存在 `self.__dict__` 里(不注册为子模块,不进 checkpoint)。

### 四种 mode(数据层/模型层配合)

- `normal`:读本样本真实 token。
- `shuffle`:provider 从缓存全体 token 里随机取**另一个样本**的 token(容量对照);seed = `shuffle_seed + worker_offset`。
- `noise`:用 `noise_stats.pt` 的 mean/std 生成噪声 token(位置对照)。
- `drop`:训练时正常拼接,**eval 时物理不拼接**(memory 回到 64,不是置零);读 normal checkpoint 测依赖度。geo_only 下构造期禁止(memory 会空)。

## 4. 关键配置

`drivoR.yaml` 的 `vggt_geometry` 块默认 `enabled: false`,所有实验用 `++` override 打开(见 `temp_script/` 脚本)。核心键:`mode / source / cache_dir / checkpoint_path(weights/vggt_omega_1b_512.pt) / vggt_dim=2048 / num_registers=16 / use_camera_token=false / joint_forward=true / preprocess_mode=balanced / image_resolution=512 / shuffle_seed=20260704 / force_ignore_fingerprint=false`。

geo_only 专属键(不在 yaml 里,运行时 `++` 注入,不进缓存指纹):`geo_only=true`、`use_layerscale_gate=false`。训练与评测必须传同样的值(state_dict 键集兜底:gate/backbone 有无不匹配加载会报错)。

`vggt_dim=2048` 语义 = `cat(frame_attn_1024, global_attn_1024)`(aggregator 末层输出)。

## 5. 硬约束(改代码前必读)

1. **融合点只在 decoder memory**。不得把 VGGT token 喂进 DrivoR ViT 主干,否则实验语义变了。
2. **归一化恰好一次**:builder/缓存输出 [0,1],ImageNet 归一化只在 VGGT Aggregator 内部做。encoder 外再归一化 = 静默双重归一化。
3. **预处理只有一份实现**:在线路径必须走 `preprocess_arrays_for_teacher`(官方 `load_fn` 桥接),禁止复刻数值逻辑。
4. **教师恒冻结**:`eval()` + `requires_grad_(False)` + `@torch.no_grad()`。可训练的只有 projector(proj/embeds/gate)。
5. **drop 不置零**:eval 期物理移除 token(避免 zero key 分走 attention)。
6. **geo_only 不构造 DINO 分支**(不是"跑了不用"):DDP `find_unused_parameters=False` 下构造而无梯度会报错;lidar + geo_only 构造期抛错。
7. **指纹不匹配 = 重新生成缓存**,不要开 `force_ignore_fingerprint` 蒙混。
8. 对基线/复现零影响:`enabled=false` 时唯一差异是 `drivor_model.py` 的 cfg 读取,无模块构造、无 import 副作用。

## 6. 运行(脚本都在 `temp_script/`,面向训练服务器)

部分脚本硬编码服务器路径 `/high_perf_store3/world-model/weixiaobao/yzj/DrivoR`(parallel 目录下的),geo_only 目录下的用 `$REPO_ROOT` 相对路径。本机(Windows)只改代码,不跑训练。

- **缓存生成**:`temp_script/cache/cache_vggt_geometry_tokens.sh`(trainval → `./vggtomega_geometry_tokens`);navtest 缓存使用 `temp_script/cache/cache_vggt_geometry_tokens_test.sh` 生成到 `./vggtomega_geometry_tokens_navtest`(并联方案见 `design/parallel.md`)。
- **并联训练**:`temp_script/parallel/train_paralle_normal.sh | train_paralle_shuffle.sh | train_paralle_noise.sh`(navtrain、10 epoch、batch 16、AdamW 2e-4、4 GPU、seed 2,无 navtest 段);`train_paralle_drivor.sh` 是 DrivoR 基线;`train_paralle.sh` 是带 navtest 段的旧入口;`train_paralle_drop.sh` **是评测**(读 normal checkpoint,`MODE=drop`),不训练。
- **并联评测**:`temp_script/parallel/eval_paralle_normal.sh | eval_paralle_shuffle.sh | eval_paralle_noise.sh | eval_paralle_drivor.sh`,走 `run_pdm_score_multi_gpu.py` + navtest token 缓存;checkpoint 自动查找路径是 `${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}/*/lightning_logs/version_*/checkpoints/last.ckpt`(注意中间的 `ke/`),或显式 `CKPT_PATH=...`。
- **geo_only**:`temp_script/geo_only/train_geo_only_normal.sh | train_geo_only_shuffle.sh`(转发到 `train_geo_only.sh`,多带 geo_only 两键;`MODE=drop` 脚本层直接拒绝);评测 `eval_geo_only_normal.sh | eval_geo_only_shuffle.sh`。可用环境变量:`EXPERIMENT / SEED / MAX_EPOCHS / BATCH_SIZE / NUM_GPUS / BASE_LR / VGGT_GEOMETRY_CACHE_DIR / EVAL_SPLIT / CKPT_PATH`。
- **vggtomega_backbone**:`temp_script/vggtomega_backbone/train_vggtomega_backbone.sh` 训练,`eval_vggtomega_backbone.sh` 评测;LoRA 变体在训练和评测两侧都追加 `agent.config.image_backbone.use_lora=true`。
- **检查脚本**:`temp_script/check_vggt_geometry_preprocess.py`(单图预处理正确性:[0,1]、patch16 对齐、内部归一化);`temp_script/check_vggt_token_norms.py`(缓存 token 范数分布;2026-07-10 实测**无高范数 outlier**,projector 前置 LN 的保留理由已改为"官方读出约定 + 数值调理 + 与并联单变量对照",见 `design/geo_only.md` §2)。

## 7. 当前路径与命名约定

1. 并联入口统一位于 `temp_script/parallel/`,脚本文件、默认实验名与 checkpoint 实验名统一使用 `paralle`。训练缓存默认读取 `$REPO_ROOT/vggtomega_geometry_tokens`,评测缓存默认读取 `$REPO_ROOT/vggtomega_geometry_tokens_navtest`。
2. 几何分支实现统一位于 `navsim/agents/drivoR/vggt_geometry.py`,`preprocess_arrays_for_teacher`、投影器、教师和缓存 provider 均从该模块使用。
3. 权重目录是 `weights'。指纹校验会读该文件算 sha256,路径错在构造期就抛 FileNotFoundError。

## 8. LingBot-Vision backbone 替换(与 §1~7 的 VGGT-Ω 几何分支相互独立)

新增第四条实验路径,替换(而非拼接)image backbone,把 DINOv2 ViT-S 换成 LingBot-Vision ViT-S/16。设计文档:`code_change_md/design/lingbotvisionbackbonereplacement.md`(已按"只留工程实现"精简,理论/假设内容不在其中)。

- 源码 vendor 在仓库根目录 `lingbot_vision/`(与上游本应逐字节一致,但见下方 py3.9 例外)。
- 新文件 `navsim/agents/drivoR/layers/image_encoder/lingbot_vision_lora.py`:`LingBotBackbone` 包一层 `LingBotVisionTransformer`,手写 `forward_features(x, scene_tokens)`(scene tokens 拼在序列最前,RoPE 的 `prefix = N - H*W` 运行时推断天然把它们当无位置 token);`build_lingbot_backbone(config)` 走 `load_config → build_backbone_from_cfg → load_state_dict` 分步路径(不用 `load_pretrained_backbone`,那个会锁 bf16/eval/frozen)。
- `dinov2_lora.py::ImgEncoder.__init__` 按 `config.impl ∈ {timm, lingbot}` 分派构建 `self.model`,其余(LoRA 手术、grid_mask、neck、pooling)完全复用不动——因为 `LingBotBackbone` 对外接口(`forward_features`/`num_features`/`patch_size`/`blocks`)和 timm 版一致。
- Config:`drivoR.yaml` 加了 `image_backbone.impl: timm`(默认,不影响基线);新增 `drivoR_lingbot.yaml`(`impl: lingbot`, `variant: small`, `model_weights: weight/lingbot-vision-vit-small/`, `image_size: [1152, 672]`,patch16 不整除 1148 故改分辨率)。
- 脚本:`temp_script/lingbot_backbone/{train,eval}_lingbot_backbone.sh`,逐行照抄 `temp_script/vggtomega_backbone/` 只换 `agent=drivoR_lingbot`。

**py3.9 兼容性例外(2026-07-12)**:vendored `lingbot_vision/vit.py` 和 `layers.py` 用了 PEP604 `float | None` 写法且没有 `from __future__ import annotations`,在 Python 3.9(本项目实际运行环境)下 import 时直接 `TypeError`。已给这两个文件顶部各加一行 `from __future__ import annotations`(只让注解变成延迟求值的字符串,不改变任何运行时逻辑)。这是当前唯一破坏"vendored 代码与上游逐字节一致"约定的地方——以后从上游同步这两个文件时要记得重新补上这一行。
