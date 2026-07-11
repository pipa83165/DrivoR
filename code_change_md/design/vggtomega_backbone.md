# vggtomega_backbone — DrivoR 图像主干换冻结 VGGT-Ω 1B(含 LoRA 开关)

> 本文档 = 设计思路 + 改动清单 + 代码骨架 + 验收断言的**合并定稿**,后续代码改动以本文档为准。
> 实验名 vggtomega_backbone(冻结版)/ vggtomega_backbone_lora 只用于文档与读数台账;**代码实体一律按内容命名**(包 `drivoR_vggt_omega`、类 `VggtOmegaImgEncoder/DrivoRVggtOmegaAgent/...`)。
> 版本溯源同并联:**不使用 git,一律文件内容 sha256**(复用 `vggt_geometry.file_sha256`)。
> 文中全部行号与源码事实已于 2026-07-10 对着当前代码核实(§4 勘察表);三轮 `question.md` 审查问题(8 + 5 + 5)已全部核实并吸收(§14 v5/v6/v7)。

---

## 1. 目标与定位

在 DrivoR(DINOv2 ViT-S + LoRA + 16 register/相机 + 双 decoder)上,把图像主干**直接换成**冻结的 VGGT-Ω 1B,回答:**最强公开几何主干"裸替"到端到端规划里,到底是什么水平?**

定位(预注册,勿漂移):

- **参考点,预期显著掉点**——它是混合方案(并联/蒸馏系)的对照背景,不是候选部署方案;
- vggtomega_backbone_lora **仅在冻结版意外接近或超过基线时才跑**,因此 LoRA 以"实现好、开关关"的方式一并落地,复活时零代码改动;
- 与并联(`parallel.md`)的分工:并联是"DINO 主干 + 并联几何 token"测增益上界;本实验是"没有 DINO、只有几何主干"测下界/替换可行性;
- 与 geo_only(`geo_only.md`)的分工:geo_only 读教师**自带**的 register(固定读出、无梯度进主干);本实验是**可学习** scene token 注入冻结主干、在线前向、梯度穿透主干流回 token。教师 register 在本实验中仍在前缀中参与前向,但**读出只取新 scene 段**,教师 register 的信息只能间接经 attention 流入——对比读数时注意这个口径差异:vggtomega_backbone − geo_only ≈ "可学习读出比教师自带 register 多榨出多少"。

## 2. 总体思路

**DrivoR 的 pipeline 一行不变,只把 `ImgEncoder`(timm DINOv2)换成一个满足同一接口契约的 `VggtOmegaImgEncoder`**;DrivoR 的感知压缩机制——可学习 scene token 注入主干、穿过网络被"写入"图像信息、出口取回 16×4=64 token 进 decoder——原样平移到 VGGT-Ω 上。

对外契约(`DrivoRModel` 无感知):

```
forward(img (B,N,3,H,W), scene_tokens (B,N,16,num_features)) → (B, N*16, tf_d_model)
num_features 属性 → 决定 scene_embeds 维度(=1024, VGGT hidden)
```

decoder、scorer、损失、优化器、训练协议、`drivor_agent.py` 全部复用,零改动。scene token 仍由 `DrivoRModel` 以 `randn*1e-6` 创建(`drivor_model.py:65` 不动,维度由 `num_features=1024` 自动派生)。

**唯一的 pipeline 内部差异——联合前向**:DINO 版 `(b n)` 展平 = 每相机独立前向;若照搬,VGGT 的 frame/inter-frame 交替 attention 在单帧输入下完全退化(global 作用域 == frame),多视几何交互——换主干的核心动机——被丢掉。因此 encoder 内部把 4 相机拼成一个 N=4 帧序列联合前向,**前视排第 0 帧**(VGGT reference frame 有专属 camera/register token,`aggregator.py:81-82` 形状 `(1,2,·,1024)` + `slice_expand_and_flatten`)。此差异完全封装在 encoder 内,对外形状不变;"独立 vs 联合"保留 reshape 级开关可做消融。

## 3. 关键设计决策

### 3.1 接入方式:哨兵分发,不是新模型类

`drivor_model.py` 的 backbone 构造点(**当前在 59-64 行**,位于 `if self.num_cams > 0 and not self.vggt_geometry_geo_only:` 分支内)加一个以 `model_name == "vggt_omega_1b"` 为哨兵的分支(惰性 import,防循环导入);互斥校验放在**所有分支之前**(§7)。这是对既有代码的**唯一**修改。

- 否掉方案 A(初版):修改 drivoR 包 5+ 文件 → 与"改动要小"冲突;
- 否掉方案 B(中间版):drivoR 包零修改,借道父类构造 + 覆盖内部属性的 hack → 引入一次性丢弃模型、配置复位等隐患,不值得;
- ✅ 采纳:最小哨兵分发。复现/基线/并联的 timm 名称恒走原路径,执行序列逐指令不变(零影响论证与 bitwise 验证见 §7.1)。

新增部分自成一包 `navsim/agents/drivoR_vggt_omega/`(带 `drivoR_` 前缀,避免与顶层官方包 `vggt_omega/` 同名);agent 只覆盖 `get_feature_builders` 一个方法。

**Agent 类名必须含 `DrivoR`**:`AgentLightningModule.validation_step` 按 `'drivor' in agent.name() or "DrivoR" in agent.name()` 分发 DrivoR 专用验证(`agent_lightning_module.py:66`),`name()` 返回类名(`drivor_agent.py:135-137`);checkpoint 回调监控的 `val/score_epoch` 只在该分支产生(`drivor_agent.py:282-286` + `validation_step` 的 `val/score` on_epoch 日志)。类名定为 **`DrivoRVggtOmegaAgent`**,否则新 agent 走通用 `_step`、不产生 monitor 指标、最优 checkpoint 静默失效、验证协议与基线不可比(验收 [7] 兜底)。

### 3.2 "新 register":可学习 scene token 注入冻结主干

每帧 token 序列 `[camera(1), 教师register(16), 新scene(16), patch(1032)]`,`patch_token_start` 17→**33**。两个"免费兼容"事实(已核实):

- RoPE 只作用于序列尾部 patch 段(`attention.py:92-96` 的 `prefix = N - sin.shape[-2]`),前缀新增 token **自动免 RoPE**,零改动;
- register-attention 层按 `self.patch_token_start` 切前缀(`aggregator.py:190-217`),改该属性后新 scene token **自动参与跨相机交互**。

注入维度 1024 = `num_features`,`DrivoRModel` 的 `scene_embeds` 形状自动派生正确、初始化沿用 `randn*1e-6` 惯例。

**初始化尺度观察点**:`randn*1e-6` 惯例在 DrivoR 里配合可训练 LoRA 主干工作;这里主干全冻结,梯度只回流 token 本身,且 pre-norm block 的 LayerNorm 会把 ~3e-5 范数的向量拉到单位尺度(方向随机)。数值上没错、忠实于"原样平移",但优化动力学与 DINO 版不同——**训练日志定期打印 `scene_embeds` 的范数与梯度范数**(验收 [2] 附带);若训练不动,第一个查这里(初始化尺度/学习率),再解读读数。默认值不改。

### 3.3 输入尺寸:真实原图 → 官方预处理 → 688×384

builder 拿 1920×1080 原图,走官方 `load_and_preprocess_images` 等价逻辑(`balanced`, `image_resolution=512`:AR 保持 + 面积归一 ≈512² + patch16 对齐),对 NAVSIM 输入**确定性输出 688×384**(1080/1920=0.5625 → 688×384,每帧 43×24=1032 patch)。配置里的 `image_size: [688, 384]` 只是记录值。

- 不能沿用 DrivoR 的 1148×672:1148/16 不整除,前向直接报错;
- 不能用原生分辨率:1080/16 不整除;裁齐后每帧 8040 patch、4 相机联合 attention >3.2 万 token(单层成本 ≈60 倍,反传不可行),且远离 checkpoint 的 ≈512² 训练分布,是分布外输入,特征更差而非更好;
- 预处理**直接复用已过三方 cosine>0.999 验收的 `vggt_geometry.preprocess_arrays_for_teacher`**,禁止复刻数值逻辑。

### 3.4 归一化只做一次,位置随官方

builder 输出 **[0,1]**(官方 load_fn 约定);ImageNet 归一化在 aggregator 内部(官方位置 `aggregator.py:108`,统计量用继承来的 `_resnet_mean/std` buffer,不手抄常量)。VGGT 的统计量与 DrivoR builder 用的 ImageNet 值相同,但**归一化只能做一次**——本 builder 不做归一化,谁也别在 encoder 外面再做(验收 [1] 兜底)。

### 3.5 读出:最末层 frame‖global 拼接(2048 维)

读出取最末 block 的 `cat(frame_attn 输出, global_attn 输出)` = 2048 维(`aggregator.py:150`,官方只在 `cached_layer_indices` 缓存,末层 23 在内)的 scene 段,与并联缓存语义完全一致(跨实验对比"同一种教师特征")。neck `Linear(2048→256)` 在 encoder 内部,`num_features`(注入维度 1024)与读出维度(2048)的差异对外不可见。

neck 不加前置 LayerNorm 是有意的(读出对象是可学习 token,冷启动分布可控,与并联 geo_proj 消费教师 register 的场景不同);若训练不稳,可给 neck 前加 `LayerNorm(2048)`(与并联 `input_ln` 同位),暂不默认加。

### 3.6 冻结为主、LoRA 为开关;trunk LoRA 本阶段禁用

- 冻结版:可训练参数 = decoder/scorer/heads(同基线量级)+ `scene_embeds` 65k + `neck` 0.5M。教师无 dropout/BN,`train()` 覆写把 aggregator 钉死 eval;
- LoRA 版:VGGT 的 `attn.qkv` 是融合 `Linear(dim, 3*dim)`,unbind 布局与 timm 完全一致(§4)→ **直接复用 DrivoR 的 `_LoRA_qkv_timm`**,零复刻。打 `frame_blocks + inter_frame_blocks`(24+24=48 个 block,+6.3M 参数);只加 q/v、rank 32(DrivoR 惯例);B 零初始化保证冷启动数值上等于冻结版;
- **本阶段锁定手写 Q/V-only LoRA,不使用标准 PEFT**:标准 PEFT 若直接 target 融合 `qkv` Linear,会对完整 3072 维输出加增量,即同时改变 Q/K/V,不再是当前的 Q/V-only 单变量。PEFT custom module dispatch 仍需自行实现 Q/V 切片与加载注册,不能消除核心定制。未来若采用标准 PEFT,必须作为独立的 `vggtomega_backbone_full_qkv_lora` 变体重新预注册、命名和报告,不得与当前结果混用;
- **trunk(DINOv3 `patch_embed.blocks`,24 层)本阶段禁止作 LoRA target,构造期抛错**。原因:`patch_embed(images)` 在 checkpoint 循环**之前**执行(§6.1);冻结时其前向不建图、零激活成本,但一旦打 LoRA,24 层 trunk 激活需为反传全程保留,不在梯度检查点保护内,§9 显存预算与验收 [6] 全部失效。未来确需 trunk LoRA 时,单独实现 trunk checkpointing 并重做显存/速度验收;
- LoRA 参数随整体 checkpoint 保存(state_dict 自带),不需要 DrivoR 的 safetensors 单独存取路径;
- **铁律:先整体冻结、再做 LoRA 手术**(新建层默认可训练;顺序反了 LoRA 会被一起冻掉)。

### 3.7 训练可行性:梯度检查点 + 门控铁律

scene token(和 LoRA)的梯度要穿透 1B 主干(**depth=24 层,每层 = frame block + inter-frame block,共 48 个 block**):不开 checkpoint 单样本激活 >5GB,不可行;开启后 ≈0.4GB/样本,A100-80G 单卡 batch 16 预算内(≈6.4GB 激活 + 4.9GB fp32 权重 + 优化器只覆盖 <20M 可训练参数),按验收 [6] 实测校准;OOM 时回退方案见 §8.4。

**门控铁律**:aggregator 被钉死在 eval,`self.training` 恒 False,**checkpoint 门控只能用 `torch.is_grad_enabled()`**——训练 step 梯度开启走 checkpoint;Lightning 验证与 `run_pdm_score`(`abstract_agent.py:78`)都在 no_grad 下,自动走原速前向。验收 [6] 用重算计数探针证明 checkpoint 实际生效,防止静默失效导致显存预算作废。

### 3.8 其余固定决策

- **与并联/geo_only 分支互斥**:`model_name == "vggt_omega_1b"` 且 `vggt_geometry.enabled=true`(含 geo_only)构造期直接抛错。**校验必须放在 backbone/geo_only 分支之前**(§7)——若放在 backbone 分支内,`geo_only=true` 会跳过整个分支,校验被绕过,配置静默退化成 geo_only,实验名与实际运行模型不一致;
- **特征来源:首次训练用在线构建**(`cache_path=null`,`use_cache_without_dataset=false`)。`default_training.yaml:18-19` 默认 cache-only(读 `navsim_cache_nommcv`),该缓存不含新 builder 名 `drivor_vggt_omega_feature` 对应的 `.gz`,照默认启动会在首批数据加载时报错。若后续改离线缓存,必须先用新唯一名生成独立缓存,并核算 ≈12.7MB/样本 × 85k ≈ **1.1TB** 磁盘;
- GridMask 保留代码路径,`use_grid_mask` 默认 **false**(对冻结主干是训练分布外输入);LoRA 版可开。开启时作用在 [0,1] 图上(归一化之前,与 DrivoR 的"归一化之后"不同,已知差异,记录在案);
- **bf16 autocast** 包裹 aggregator(镜像官方 `vggt_omega.py:39-41`),读出后 fp32 进 neck;
- 相机顺序 `[f0, l0, r0, b0]`(前视第 0 帧),与并联缓存的 `VGGT_GEOMETRY_CAMERA_ORDER` 相同,跨实验口径一致;
- 权重加载:**复用 `FrozenVggtGeometryTeacher` 的解包逻辑**(`model`/`state_dict` 双键 + `module.` 前缀清理,`vggt_geometry.py:203-206`),再过滤 `aggregator.*` 前缀,最后 **strict=True**(键集精确匹配)。不要在新包里维护第二套不一致的 checkpoint 解析——只处理 `model` 键会把合法 checkpoint 解析成错误键集,strict=True 下报"缺少全部 aggregator 参数",极难定位;
- checkpoint 含冻结 1.2B(≈4.9GB/份),接受,换取 `initialize()`/断点续训零改动。

## 4. 代码勘察结论(事实依据,2026-07-10 对当前代码逐条核实)

| 事项 | 结论 | 出处 |
|---|---|---|
| backbone 构造点 | `self.image_backbone = ImgEncoder(config_image_backbone)`,前三行已把 `image_size/num_scene_tokens/tf_d_model` 注入 backbone 配置 → 新 encoder 免费获得;整段在 `not vggt_geometry_geo_only` 分支内 | `drivor_model.py:59-64` |
| scene_embeds 创建 | `randn*1e-6`,维度取 `image_backbone.num_features` → `num_features=1024` 自动正确,**该行不改** | `drivor_model.py:65` |
| vggt_geometry_enabled 解析 | 在 backbone 构造**之前**(51 行)已算好 → 前置互斥校验可直接引用 | `drivor_model.py:49-51` |
| image 消费点 | `features["image"]` → `self.image_backbone(img, scene_tokens)` | `drivor_model.py:192-202` |
| 验证分支分发 | `validation_step` 按 `'drivor' in agent.name() or "DrivoR" in agent.name()` 走 DrivoR 专用评分;否则通用 `_step` | `agent_lightning_module.py:66-114` |
| agent.name() | 返回 `self.__class__.__name__` → 类名必须含 `DrivoR` | `drivor_agent.py:135-137` |
| checkpoint monitor | `ModelCheckpoint(monitor='val/score_epoch', mode=max)`,指标只在 DrivoR 验证分支产生(`val/score` on_step+on_epoch → `_epoch` 后缀) | `drivor_agent.py:282-286`;`agent_lightning_module.py:97` |
| 默认训练配置 | cache-only:`cache_path=$NAVSIM_EXP_ROOT/navsim_cache_nommcv`、`use_cache_without_dataset: true`;20 epochs、batch 64 → **必须覆盖** | `default_training.yaml:17-34` |
| batch 口径 | `drivoR.yaml:142` 注释明确 dataloader batch_size 是 **per-GPU**;基线脚本 batch 16/卡 × 4 卡 = 全局 64 | `drivoR.yaml:141-142`;`temp_script/parallel/train_paralle_drivor.sh` |
| LR/scheduler 计算 | `global_batchsize = agent.batch_size × agent.num_gpus`;`lr = base_lr × sqrt(global/base_batch_size)`;`T_max = ceil(dataset_size/global) × num_epochs`,scheduler 按 optimizer step 步进;**梯度累积不进任何一项计算**;`agent.batch_size` 默认插值自 `dataloader.params.batch_size`(`drivoR.yaml:141`),可被 Hydra 显式覆盖 | `drivor_agent.py:237-275` |
| GPU 数控制 | `trainer.params` 无 `devices` 键 → Lightning 用全部可见 GPU;实际 world size 由 `CUDA_VISIBLE_DEVICES` 决定;`agent.num_gpus` 只进 LR/T_max 计算,不控制设备数 | `default_training.yaml:32-53`;基线脚本 `export CUDA_VISIBLE_DEVICES` |
| Aggregator 结构 | `depth=24`;每层 = `frame_blocks[i]` + `inter_frame_blocks[i]`(共 48 block);register-attention 层 idx `[2,6,9,14,20]`;`patch_token_start = 1+16 = 17`;输出只在 `cached_layer_indices=(4,11,17,23)` 缓存 | `aggregator.py:21-89,149-152` |
| qkv 布局(LoRA 可移植性) | `qkv = Linear(dim, 3*dim)`;`reshape(B,N,3,heads,hd); unbind(2)` → q 前 1/3、v 后 1/3,与 timm 一致;`_LoRA_qkv_timm` 往 `[:, :, :dim]` 加 new_q、`[:, :, -dim:]` 加 new_v | `attention.py:78,128-129`;`dinov2_lora.py:126-134` |
| RoPE 前缀处理 | `prefix = N - sin.shape[-2]`,前缀 token 不加 RoPE → 新增前缀 token 零改动兼容 | `attention.py:92-99` |
| register-attention 切片 | 按 `self.patch_token_start` 切,改该属性即自动带上新 token | `aggregator.py:190-217` |
| 2048 维语义 | 末层 `cat([frame_tokens, tokens], -1)`,前 1024 frame-attn、后 1024 global-attn | `aggregator.py:150` |
| reference frame | `camera_token/register_token` 形状 `(1,2,·,1024)`,`slice_expand_and_flatten` 给第 0 帧专属 token → 前视排第 0 | `aggregator.py:81-82,246` |
| 归一化位置 | ImageNet mean/std 在 `Aggregator.forward` 内部;输入应为 [0,1] | `aggregator.py:108` |
| 官方 autocast/读出 | `VGGTOmega.forward` bf16 autocast 包 aggregator,`camera_and_register_tokens = final_tokens[:, :, :patch_token_start]` | `vggt_omega.py:39-48` |
| checkpoint 解包(现成实现) | `model`/`state_dict` 双键 + `module.` 前缀清理 | `vggt_geometry.py:203-206` |
| trunk 位置 | `patch_embed(images)` 在 checkpoint 循环之前执行;trunk = `aggregator.patch_embed.blocks`(24 层)→ trunk LoRA 不受检查点保护 | `aggregator.py:114`;`vision_transformer.py:166` |
| 优化器 | `AdamW(self._drivor_model.parameters())`,grad=None 的冻结参数在 step 中被跳过,无需过滤;LoRA 新参数自动进入 | `drivor_agent.py:245` |
| checkpoint 加载(agent 侧) | `initialize()` 只做 key 前缀替换,与模型类无关,继承即正确 | `drivor_agent.py:139-148` |
| feature builder 唯一名 | 缓存按 `get_unique_name()` 区分 → 新 builder **必须换名** | `drivor_features.py:34-36` |
| cam_K/world_2_cam | 只在 `drivor_features.py` 产出,全仓库无消费 → 新 builder 不输出无影响 | 全仓库 grep |
| 评测 no_grad | `compute_trajectory` 在 `torch.no_grad()` 下 → checkpoint 门控自动关闭 | `abstract_agent.py:78` |
| 教师输入尺寸 | 1920×1080 → balanced/512/patch16 → **688×384**(每帧 33+1032=1065 token,4 帧联合 4260) | `load_fn.py` 逻辑,同并联 |
| 在线预处理桥接 | `preprocess_arrays_for_teacher` 已过三方 cosine>0.999 验收 → **直接 import,禁止再写一份** | `navsim/agents/drivoR/vggt_geometry.py:239` |

## 5. 文件清单

| 操作 | 文件 | 内容 |
|---|---|---|
| **修改** | `navsim/agents/drivoR/drivor_model.py` | 前置互斥校验 + 构造分发(§7,约 8 行) |
| 新增 | `navsim/agents/drivoR_vggt_omega/__init__.py` | 空 |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py` | `SceneTokenAggregator`、`apply_lora_to_blocks`、`VggtOmegaImgEncoder` |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py` | `VggtOmegaFeatureBuilder(DrivoRFeatureBuilder)`,唯一名 `drivor_vggt_omega_feature` |
| 新增 | `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py` | `DrivoRVggtOmegaAgent(DrivoRAgent)`:仅覆盖 `get_feature_builders`(类名含 `DrivoR`,§3.1) |
| 新增 | `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml` | agent 配置 |
| 新增 | `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh` | 训练入口(锁定协议 + 在线特征覆盖 + GPU 数锁定,§8.4) |
| 新增 | `temp_script/vggtomega_backbone/eval_vggtomega_backbone.sh` | navtest 评测入口(checkpoint 查找 + LoRA 开关同步,§8.5) |
| 新增 | `scripts/vggt_omega_acceptance_checks.py` | golden 留档模式 + 验收断言(§10) |

**不改**:`drivor_agent.py`、`drivor_features.py`、decoder、scorer、损失、`dataset.py`、`run_training_full.py`、`default_training.yaml`、并联/geo_only 全部代码(唯一依赖是 `vggt_geometry.preprocess_arrays_for_teacher`,纯函数)。**不新增** training yaml——仓库惯例是 shell 脚本 + Hydra override(同 `temp_script/parallel/`),训练参数全部显式写在脚本里,避免静默继承 `default_training` 的 cache-only/20ep/batch64。

## 6. 新增 `navsim/agents/drivoR_vggt_omega/vggt_omega_backbone.py`

### 6.1 `SceneTokenAggregator` — 子类化官方 Aggregator,只改 forward

```python
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt_omega.models.aggregator import Aggregator, slice_expand_and_flatten
from navsim.agents.drivoR.layers.image_encoder.dinov2_lora import _LoRA_qkv_timm
from navsim.agents.drivoR.layers.image_encoder.grid_mask import GridMask


class SceneTokenAggregator(Aggregator):
    """官方 Aggregator + 每帧前缀区注入可学习 scene token(新 register)。
    前缀布局: [camera(1), 教师register(16), 新scene(S), patch...]
    patch_token_start 由 17 -> 17+S,register-attention 层与 RoPE 前缀逻辑自动适配。"""

    def __init__(self, num_scene_tokens: int = 16, grad_checkpointing: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.num_scene_tokens = num_scene_tokens
        self.grad_checkpointing = grad_checkpointing
        self.scene_token_start = self.patch_token_start          # 17: scene 段起点
        self.patch_token_start = self.patch_token_start + num_scene_tokens   # 33

    def forward_full(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """images: (B, N, 3, H, W) in [0,1];scene_tokens: (B, N, S, 1024) 可学习。
        4 相机作为一个 N 帧序列联合前向(前视=第0帧 reference frame)。
        返回最末层 cat(frame, global) 的全 token: (B, N, num_tokens, 2048)。
        生产路径只用 forward();本方法同时服务验收 [1](S=0 时取 camera+register 段
        与官方对比 —— forward() 的 scene 切片在 S=0 下为空,不能用于该验收)。"""
        B, N, C, H, W = images.shape
        images = (images - self._resnet_mean) / self._resnet_std   # 官方位置的唯一一次归一化
        images = images.view(B * N, C, H, W)

        camera_token = slice_expand_and_flatten(self.camera_token, B, N)
        register_token = slice_expand_and_flatten(self.register_token, B, N)
        scene = scene_tokens.reshape(B * N, self.num_scene_tokens, -1)

        # patch_embed(DINOv3 trunk)在 checkpoint 循环之外:冻结 + 输入无梯度时
        # 不建图、零激活成本;正因如此 trunk 禁止打 LoRA(§3.6,构造期兜底)。
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, scene, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        grid = (H // self.patch_size, W // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=grid[0], W=grid[1])
            frame_rope = (rope_sin.to(patch_tokens.device, torch.float32),
                          rope_cos.to(patch_tokens.device, torch.float32))

        def run_block(tokens, idx):
            tokens, frame_tokens = self._run_frame_block(
                tokens, B, N, num_tokens, embed_dim, idx, frame_rope)
            tokens = self._run_inter_frame_attention_block(
                tokens, B, N, num_tokens, embed_dim, idx,
                self.inter_frame_attention_types[idx])
            return tokens, frame_tokens

        frame_tokens = None
        for idx in range(self.depth):
            # 门控禁止用 self.training:encoder.train() 把本模块钉死在 eval,
            # self.training 恒为 False,checkpoint 会被静默关闭。
            # 训练/评测的区分交给梯度开关:训练 step 梯度开启;Lightning 验证与
            # run_pdm_score(abstract_agent.py:78)都在 no_grad 下 → 自动走原速前向。
            if self.grad_checkpointing and torch.is_grad_enabled():
                tokens, frame_tokens = checkpoint(run_block, tokens, idx, use_reentrant=False)
            else:
                tokens, frame_tokens = run_block(tokens, idx)

        return torch.cat([frame_tokens, tokens], dim=-1)           # (B, N, num_tokens, 2048)

    def forward(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """生产路径:只取 scene 段 (B, N, S, 2048)。"""
        out = self.forward_full(images, scene_tokens)
        return out[:, :, self.scene_token_start:self.patch_token_start]
```

要点:`_run_frame_block` / `_run_inter_frame_attention_block` / `_resnet_mean` / `rope_embed` 全部继承官方实现(已核对 `aggregator.py:100-154` 官方 forward,骨架逐调用一致),数值逻辑零复刻;只保留最末层输出(官方 `cached_layer_indices` 中间层缓存不需要);checkpoint 门控见 §3.7 铁律。

### 6.2 `apply_lora_to_blocks` — 复用 DrivoR 的 LoRA 包装

```python
def apply_lora_to_blocks(blocks: nn.ModuleList, r: int) -> list:
    """对一组 SelfAttentionBlock 做 qkv 手术(只加 q/v,同 DrivoR 惯例)。
    返回新建的 LoRA 线性层列表(调用方负责保证它们 requires_grad=True)。
    必须在整体冻结之后调用 —— 新建层默认可训练,顺序错了会把 LoRA 一起冻掉。"""
    lora_layers = []
    for blk in blocks:
        qkv = blk.attn.qkv
        dim = qkv.in_features
        a_q, b_q = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
        a_v, b_v = nn.Linear(dim, r, bias=False), nn.Linear(r, dim, bias=False)
        for a in (a_q, a_v):
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
        for b in (b_q, b_v):
            nn.init.zeros_(b.weight)                 # 冷启动 == 冻结版(验收 [3])
        blk.attn.qkv = _LoRA_qkv_timm(
            qkv, a_q, b_q, a_v, b_v,
            nn.Identity(), nn.Identity(),            # k 路不加(同 DrivoR use_qkv=False)
            nn.Identity(), nn.Identity(), nn.Identity(),   # 无额外 LayerNorm
        )
        lora_layers += [a_q, b_q, a_v, b_v]
    return lora_layers
```

依据:VGGT `attn.qkv` 融合布局与 timm 一致(§4),`_LoRA_qkv_timm` 往前 1/3 加 new_q、后 1/3 加 new_v,语义正确;`LinearKMaskedBias` 的 bias mask 行为被原样保留(原 qkv 模块整体包在里面)。

**复用边界(勿整体复用 `LoRA_ViT_timm`)**:DrivoR 的 `LoRA_ViT_timm` 是"包装器 + 遍历器 + safetensors 存取"三合一,不能直接套 VGGT——它写死遍历 `vit_model.blocks`(VGGT 是 `frame_blocks`/`inter_frame_blocks` 两条列表)、会把模型吞进 `self.lora_vit` 改掉模块树与 state_dict 键(破坏 strict=True 加载与 eval 钉死覆写)、且自带我们不用的 safetensors 路径(§3.6:LoRA 参数随 Lightning checkpoint 走)。只复用其数值核心 `_LoRA_qkv_timm`(单模块、与外层结构无关);`apply_lora_to_blocks` 就是为 VGGT 写的薄遍历器,k 路/LN 传 `nn.Identity()` 等价 DrivoR 默认(`use_qkv=False, use_layer_norm=False`,`dinov2_lora.py:318` 用默认值)。

### 6.3 `VggtOmegaImgEncoder` — 满足 DrivoRModel 的 backbone 契约

```python
class VggtOmegaImgEncoder(nn.Module):
    """VGGT-Omega 1B 做图像主干(默认冻结;use_lora=true = LoRA 变体)。
    接口契约与 ImgEncoder 相同:
    forward(img (B,N,3,H,W), scene_tokens (B,N,S,num_features)) -> (B, N*S, tf_d_model);
    num_features=1024(注入维度);neck 输入 2048(读出维度),对外不可见。"""

    VGGT_EMBED_DIM = 1024
    READOUT_DIM = 2048          # cat(frame_attn, global_attn),同并联缓存语义
    LORA_TARGETS = ("frame", "inter_frame")   # trunk 禁用:不在 checkpoint 保护内(§3.6)

    def __init__(self, config):
        super().__init__()
        self.num_features = self.VGGT_EMBED_DIM

        self.aggregator = SceneTokenAggregator(
            num_scene_tokens=config["num_scene_tokens"],
            grad_checkpointing=config.get("grad_checkpointing", True),
        )
        # 解包逻辑与 FrozenVggtGeometryTeacher 一致(vggt_geometry.py:203-206):
        # model/state_dict 双键 + module. 前缀;随后过滤 aggregator.* 前缀。
        # LoRA 手术在加载之后、scene token 参数在模块外 → 键集必须精确匹配,strict=True
        # (missing 与 unexpected 双向为空)。
        state = torch.load(config["checkpoint_path"], map_location="cpu")
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        state = {k[len("aggregator."):]: v for k, v in state.items() if k.startswith("aggregator.")}
        self.aggregator.load_state_dict(state, strict=True)

        # 1) 先整体冻结
        for p in self.aggregator.parameters():
            p.requires_grad_(False)

        # 2) 再做 LoRA 手术(新建层默认可训练;顺序不可颠倒)
        self.use_lora = config.get("use_lora", False)
        if self.use_lora:
            targets = {
                "frame": self.aggregator.frame_blocks,
                "inter_frame": self.aggregator.inter_frame_blocks,
            }
            for name in config.get("lora_targets", list(self.LORA_TARGETS)):
                if name not in targets:
                    # trunk 等:patch_embed 在 checkpoint 循环外,LoRA 激活无保护(§3.6)
                    raise ValueError(f"unsupported lora target: {name} (allowed: {sorted(targets)})")
                apply_lora_to_blocks(targets[name], r=config.get("lora_rank", 32))

        self.neck = nn.Linear(self.READOUT_DIM, config["tf_d_model"])

        # pipeline 保留 GridMask 代码路径;冻结版默认关(教师域外输入),LoRA 版可开
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = config.get("use_grid_mask", False)

    def train(self, mode: bool = True):
        # Lightning 每个 epoch 全模型 .train();主干钉死 eval 语义
        # (VGGT/DINOv3 无 dropout/BN,数值本就一致;LoRA 线性层不受 eval 影响,梯度照常)
        super().train(mode)
        self.aggregator.eval()
        return self

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        B, N = img.shape[:2]
        if self.use_grid_mask and self.training:
            img = self.grid_mask(img.flatten(0, 1)).view_as(img)   # [0,1] 图上打洞(归一化前)
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            geo = self.aggregator(img, scene_tokens)               # (B, N, S, 2048)
        tokens = self.neck(geo.float())                            # (B, N, S, 256), fp32
        return tokens.reshape(B, -1, tokens.shape[-1])             # (B, N*S, 256)
```

要点:

- **梯度路径**:`scene_embeds`(以及 LoRA A/B)→ 穿过冻结的 24 层(48 block)→ 读出;参数冻结不阻断对输入的梯度(验收 [2]);
- neck 在 autocast 外 fp32 计算(与并联 geo_proj 约定一致);
- checkpoint 含冻结 1.2B 权重(≈4.9GB fp32/份)。接受:保证 `initialize()`/断点续训零改动;磁盘敏感时可后续加 `on_save_checkpoint` 钩子剔除,不进本次改动。

## 7. 修改 `navsim/agents/drivoR/drivor_model.py`(唯一的既有文件改动)

**(a) 前置互斥校验**——紧跟 `vggt_geometry_enabled/geo_only` 解析(51-56 行)之后、**任何分支之前**(否则 `geo_only=true` 会跳过 backbone 分支,校验被绕过,配置静默退化成 geo_only):

```python
        # vggt_omega_1b 主干与并联/geo_only 几何分支互斥(含 geo_only:必须先于分支判断)
        _backbone_model_name = cfg_get(cfg_get(config, "image_backbone", None), "model_name", None)
        if _backbone_model_name == "vggt_omega_1b" and self.vggt_geometry_enabled:
            raise ValueError(
                "vggt_omega_1b backbone must not be combined with vggt_geometry.enabled=true"
            )
```

**(b) 构造分发**——`self.image_backbone = ImgEncoder(config_image_backbone)`(**当前 64 行**,在 `if self.num_cams > 0 and not self.vggt_geometry_geo_only:` 分支内)改为:

```python
            if config_image_backbone.get("model_name") == "vggt_omega_1b":
                from navsim.agents.drivoR_vggt_omega.vggt_omega_backbone import VggtOmegaImgEncoder
                self.image_backbone = VggtOmegaImgEncoder(config_image_backbone)
            else:
                self.image_backbone = ImgEncoder(config_image_backbone)
```

- 哨兵是 `model_name`,复现/基线/并联的 timm 名称走原路径,行为零变化;
- `cfg_get` 已在 `drivor_model.py:9` import(并联引入),前置校验直接复用;
- **惰性 import**(放分支内):不用 VGGT 时零导入成本,也天然避免 drivoR ↔ drivoR_vggt_omega 循环导入;
- 上方三行(61-63)已把 `image_size / num_scene_tokens / tf_d_model` 注入 `config_image_backbone`,新 encoder 免费获得;
- 下一行 `scene_embeds`(65 行)**不动**:`num_features=1024` 自动派生正确形状,初始化沿用 `randn*1e-6`;
- `drivor_agent.py` **不动**:`DrivoRModel(config)` 构造经此分发自动得到 VGGT backbone。

### 7.1 对既有实验(复现/基线/并联/geo_only)零影响的论证(硬性要求)

| 途径 | 论证 |
|---|---|
| 代码路径 | 改动 = 前置互斥校验 + 构造分发;原实验的 `model_name` 是 timm 名称 → 校验条件恒 False、恒走 `else` 原路径 |
| import 副作用 | 新包 import 是惰性的(在 vggt 分支内),原实验从不触发 |
| 配置 | `drivoR.yaml` / `default_training.yaml` 零改动;新增 yaml 是独立文件,hydra 不会隐式加载 |
| 特征缓存 | 新 builder 唯一名 `drivor_vggt_omega_feature`,与 `drivor_feature` 缓存目录天然隔离 |
| 并联/geo_only 代码 | 只 import `preprocess_arrays_for_teacher`(纯函数),不修改 `vggt_geometry.py` 任何内容 |
| lidar 分支 | lidar `ImgEncoder` 构造(`drivor_model.py:70-76`)不在分发范围内,原样保留 |

**验证(不止靠论证)**:验收 [4],其中 bitwise 对比依赖**改代码前预先留档的 golden artifact**(§10-[0]);实施顺序第 0 步强制执行,补丁打完后无法再补。

## 8. 新增 features / agent / 配置 / 训练入口

### 8.1 `navsim/agents/drivoR_vggt_omega/vggt_omega_features.py`

```python
from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.vggt_geometry import preprocess_arrays_for_teacher

# 前视必须第 0 帧(VGGT reference frame);与并联缓存的 VGGT_GEOMETRY_CAMERA_ORDER 一致。
CAMERA_ORDER = ("cam_f0", "cam_l0", "cam_r0", "cam_b0")


class VggtOmegaFeatureBuilder(DrivoRFeatureBuilder):
    """真实原图(1920x1080)→ 官方 VGGT 预处理(balanced/512, AR 保持, patch16 对齐)
    → (4, 3, 384, 688), [0,1]。不做 ImageNet 归一化(在 encoder 内部做,官方位置)、
    不做颜色增广。ego_status 逻辑继承不变。"""

    def get_unique_name(self) -> str:
        return "drivor_vggt_omega_feature"   # 必须区别于 drivor_feature,防缓存互相污染

    def _get_camera_feature(self, agent_input):
        cameras = agent_input.cameras[-1]
        raw = [getattr(cameras, name).image for name in CAMERA_ORDER]
        images = preprocess_arrays_for_teacher(
            raw,
            mode=self._config.get("vggt_preprocess_mode", "balanced"),
            image_resolution=self._config.get("vggt_image_resolution", 512),
        )
        return {"image": images}
```

- 键仍叫 `image`,`DrivoRModel.forward`(192-202 行)零改动消费;`cam_K`/`world_2_cam` 不再输出(全仓库无消费,§4);
- 特征体量 688×384×3×4 fp32 ≈12.7MB/样本(< DrivoR 原生 ≈37MB);**首次训练在线构建,不进特征缓存**(§3.8)。

### 8.2 `navsim/agents/drivoR_vggt_omega/vggt_omega_agent.py`

```python
from navsim.agents.drivoR.drivor_agent import DrivoRAgent
from .vggt_omega_features import VggtOmegaFeatureBuilder


class DrivoRVggtOmegaAgent(DrivoRAgent):
    """DrivoRAgent 全部逻辑继承(模型构造经 drivor_model.py 分发自动得到 VGGT backbone;
    loss、优化器、metric cache、callbacks、checkpoint 加载均复用),仅换 feature builder。
    类名必须含 DrivoR:AgentLightningModule.validation_step 按 name() 分发 DrivoR 专用
    验证分支,checkpoint monitor(val/score_epoch)只在该分支产生(§3.1,验收 [7])。"""

    def get_feature_builders(self):
        return [VggtOmegaFeatureBuilder(config=self._config)]
```

### 8.3 `navsim/planning/script/config/common/agent/drivoR_vggt_omega.yaml`

复制 `drivoR.yaml` 后仅改:

```yaml
_target_: navsim.agents.drivoR_vggt_omega.vggt_omega_agent.DrivoRVggtOmegaAgent

config:
  # ...(其余键与 drivoR.yaml 完全相同,略;vggt_geometry 块保留且必须 enabled: false,
  #     enabled=true 会在 DrivoRModel 构造期与本主干互斥抛错)...

  image_size: [688, 384]        # = 官方 balanced/512 对 1920x1080 的确定性输出(记录用)
  vggt_preprocess_mode: balanced
  vggt_image_resolution: 512

  image_backbone:
    model_name: vggt_omega_1b   # 分发哨兵(§7)
    checkpoint_path: weights/vggt_omega_1b_512.pt
    grad_checkpointing: true
    use_grid_mask: false        # 冻结版默认关;LoRA 版可开(消融)
    # ---- LoRA(实验 vggtomega_backbone_lora,仅冻结版意外好时才跑;开关在此)----
    use_lora: false
    lora_rank: 32
    lora_targets: [frame, inter_frame]   # 仅允许这两个;trunk 构造期抛错(§3.6)
    # ImgEncoder 专属键(model_weights/finetune/focus_front_cam/...)全部删除
```

**权重路径注意**:本地开发机是 `weight/`(单数)、训练服务器是 `weights/`(复数)——与并联配置保持同一写法 `weights/vggt_omega_1b_512.pt`,本地跑验收时用 override 或软链接对齐,**不要两处配置写不同路径**。

### 8.4 训练入口 `temp_script/vggtomega_backbone/train_vggtomega_backbone.sh`

不新增 training yaml(避免静默继承 `default_training` 的 cache-only/20ep/batch64);仿 `temp_script/parallel/train_paralle_drivor.sh`,协议全部显式:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$REPO_ROOT/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REPO_ROOT/dataset}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"   # 锁定 4 卡:Lightning 无
# devices 覆盖时用全部可见 GPU;agent.num_gpus 只进 LR/T_max 计算,不控制设备数(§4)
NUM_GPUS="${NUM_GPUS:-4}"
python -c "import torch; n=torch.cuda.device_count(); assert n==${NUM_GPUS}, \
    f'visible GPUs {n} != NUM_GPUS ${NUM_GPUS}: LR/T_max would be computed for the wrong world size'"

EXPERIMENT="${EXPERIMENT:-vggtomega_backbone_10ep}"
python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py" \
    agent=drivoR_vggt_omega \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=10 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size="${BATCH_SIZE:-16}" \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus="$NUM_GPUS" \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed="${SEED:-2}" \
    "$@"
```

- `cache_path=null use_cache_without_dataset=false` = 在线构建特征(§3.8,新 builder 名在既有缓存里不存在,cache-only 必失败);
- **batch 口径**:`dataloader.params.batch_size` 是 **per-GPU**(`drivoR.yaml:142` 注释 + 基线脚本一致)→ 16/卡 × 4 卡 = **全局 64,与基线协议完全相同**(lr 2e-4 对 `base_batch_size: 64` 恰为 1:1)。早前文档写"全局 batch 16"是口径错误,已废弃——协议必须与基线同全局 batch 才可比;
- **GPU 数锁定**:实际 world size 由 `CUDA_VISIBLE_DEVICES` 决定(`trainer.params` 无 `devices` 键,Lightning 用全部可见卡),`agent.num_gpus` 只进 LR/T_max 计算——两者不一致时 LR/scheduler 静默算错,故脚本前置断言 `torch.cuda.device_count() == NUM_GPUS`;
- **OOM 回退**(验收 [6] 实测后决定):`get_optimizers` 的 `global_batchsize = agent.batch_size × num_gpus`,**梯度累积不进该计算**(§4)——只改 dataloader batch 会把 lr 静默降为 `2e-4×sqrt(16/64)=1e-4`、scheduler T_max 按 step 口径拉长 4 倍。正确回退必须三个键一起覆盖:

  ```bash
  BATCH_SIZE=4 bash train_vggtomega_backbone.sh \
      trainer.params.accumulate_grad_batches=4 \
      agent.batch_size=16
  ```

  `agent.batch_size=16` 显式覆盖 `${dataloader.params.batch_size}` 插值(`drivoR.yaml:141`),使 LR/T_max 仍按"有效 batch 16/卡 × 4 卡 = 全局 64"计算(scheduler 按 optimizer step 步进,累积下 optimizer step 数恰与正常配置一致);micro-batch 实际为 4。验收 [8] 断言这两个量;
- LoRA 版复活:追加 `agent.config.image_backbone.use_lora=true`(+ 按需 `use_grid_mask=true`),`EXPERIMENT=vggtomega_backbone_lora_10ep`。

与 `train_paralle_drivor.sh` 的协议对照(实现/审查时逐项核对):

| 类别 | 与基线相同 | vggtomega_backbone 唯一差异 |
|---|---|---|
| 数据/周期 | navtrain、10 epochs、在线 feature、seed 2 | builder 改为官方 VGGT 预处理,输入 688×384、[0,1] |
| 优化 | AdamW、lr 2e-4、batch 16/卡 × 4、prefetch 1 | OOM 时只允许三键联动回退,有效 batch/LR/T_max 不变 |
| DrivoR 模型协议 | one_token_per_traj、refiner heads/层数、tf 维度、loss 权重、`long_trajectory_additional_poses=2` 全部相同 | agent/backbone 类型不同;GridMask 默认关闭 |
| 几何分支 | — | `vggt_geometry.enabled=false`,不允许再并联几何 token |

训练与评测必须同时保留 `long_trajectory_additional_poses=2`;这是基线脚本相对 `drivoR.yaml` 默认值 `-1` 的显式协议覆盖,不得依赖 agent yaml 默认值。

### 8.5 评测入口 `temp_script/vggtomega_backbone/eval_vggtomega_backbone.sh`

仿 `temp_script/parallel/eval_paralle_drivor.sh`(本 agent 在线前向、无 token 缓存段),关键差异:`agent=drivoR_vggt_omega` + 显式 `vggt_geometry.enabled=false`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$REPO_ROOT/dataset/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$REPO_ROOT/exp}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$REPO_ROOT}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$REPO_ROOT/dataset}"
export SUBSCORE_PATH="${SUBSCORE_PATH:-$NAVSIM_EXP_ROOT}"
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CKPT_EXPERIMENT="${CKPT_EXPERIMENT:-vggtomega_backbone_10ep}"
EVAL_SPLIT="${EVAL_SPLIT:-navtest}"
EXPERIMENT="${EXPERIMENT:-vggtomega_backbone_${EVAL_SPLIT}}"
CKPT_PATH="${CKPT_PATH:-}"

if [ -z "$CKPT_PATH" ]; then
    CKPT_PATH=$(ls -t "${NAVSIM_EXP_ROOT}/ke/${CKPT_EXPERIMENT}"/*/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null | head -n 1 || true)
fi
[ -n "$CKPT_PATH" ] || { echo "Error: checkpoint not found; set CKPT_PATH or CKPT_EXPERIMENT."; exit 1; }

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_multi_gpu.py" \
    train_test_split="$EVAL_SPLIT" \
    agent=drivoR_vggt_omega \
    agent.checkpoint_path="$CKPT_PATH" \
    experiment_name=$EXPERIMENT \
    agent.config.proposal_num=64 \
    agent.config.refiner_ls_values=0.0 \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.config.long_trajectory_additional_poses=2 \
    ++trainer.params.logger=false \
    ++trainer.params.enable_checkpointing=false \
    agent.config.vggt_geometry.enabled=false \
    agent.config.noc=1 \
    agent.config.dac=1 \
    agent.config.ddc=0.0 \
    agent.config.ttc=5 \
    agent.config.ep=5 \
    agent.config.comfort=2 \
    "$@"
```

- checkpoint 查找规则与并联评测一致(注意 `${NAVSIM_EXP_ROOT}/ke/` 前缀),或显式 `CKPT_PATH=...`;
- **LoRA checkpoint 评测必须同步开关**:追加 `agent.config.image_backbone.use_lora=true`(建议 `EXPERIMENT=vggtomega_backbone_lora_${EVAL_SPLIT}`)。忘开时 `initialize()` 的 `load_state_dict`(strict)会因缺少 LoRA 键直接报错——响亮失败,不会静默错评,但报错信息不直观,先想到这里;
- `vggt_geometry.enabled=false` 显式写出(agent yaml 默认已 false;显式覆盖防止有人复制并联评测脚本时带进 `++agent.config.vggt_geometry.*` 覆盖,与构造期互斥校验双保险);
- 评测走 `run_pdm_score_multi_gpu.py` → 同一 builder 在线预处理 + VGGT 在线前向(no_grad,checkpoint 门控自动关,§3.7),无任何缓存依赖;
- 验收 [9] 要求 navtest 缩样冒烟(如 `train_test_split=navmini` 或过滤少量 token)先跑通全链路再上全量。

## 9. 训练成本与显存核算(预算依据,启动前实测校准)

- 每帧 token:1+16+16+1032=**1065**;4 帧联合 global attention **4260 token**;aggregator depth=24(48 block)+ 24 层 DINOv3 trunk(冻结、无梯度、不建图);
- **梯度检查点必开**:关闭时激活 >5GB/样本;开启后 block 边界激活 ≈0.4GB/样本 + 单层重算峰值。单卡 batch 16 ≈ 6.4GB 激活 + 4.9GB fp32 权重,A100-80G 预算内;不行则 §8.4 回退方案(batch 4 + 梯度累积 4);
- 时间粗估:bf16 前向 ~0.1–0.15s/样本,checkpointing 反传 ≈2×前向 → 85k×10 epochs ÷ 4 卡 ≈ **20–30h**(估算,验收 [6] 实测校准);
- 可训练参数:冻结版 = decoders+scorer+heads(同基线量级)+ `scene_embeds` 65k + `neck` 0.5M;LoRA 版另加 48 block × (q,v) × (A+B) ≈ **6.3M**。**日志打印 trainable/frozen 计数**(验收 [2] 断言);
- LoRA 版反传新增权重梯度计算,时间 ≈ 冻结版(输入梯度本来就要算),显存增量可忽略。

## 10. 新增 `scripts/vggt_omega_acceptance_checks.py`(训练启动前跑一遍留档)

```python
# [0] golden 留档(--capture-golden 模式,必须在改 drivor_model.py 之前执行!):
#     基线配置、pl.seed_everything 固定种子、固定假 batch,跑 1 个训练 step,保存:
#     loss 值(bit 精度)、forward 输出、模型初始 state_dict 的 sha256、
#     torch/CUDA/cuDNN 版本 → scripts/artifacts/vggt_omega_golden.pt。
#     [4]b 读取它做补丁后对比;补丁打完后无法再补,故为实施顺序第 0 步(§12)。
#     对比必须在同一机器同一环境(bitwise 才有意义);若确需跨环境,
#     改为显式容差(如 rtol=1e-6)并在留档里记录原因与环境差异。

# [1] 前向正确性:num_scene_tokens=0 时,SceneTokenAggregator 与官方 VGGTOmega 输出一致
#     同一输入 (1,4,3,384,688) [0,1],取 forward_full(...)[:, :, :17](camera+register 段;
#     注意不能用 forward() —— 其 scene 切片在 S=0 下为空),
#     与 VGGTOmega(enable_camera=False, enable_depth=False) 的
#     camera_and_register_tokens 逐 token cosine > 0.999
#     —— 验证 forward 复制 + patch_token_start 改写 + 归一化位置都没引入偏差

# [2] 梯度路径(冻结版):DrivoRModel(vggt 配置) 1 次前向+反传后
#     assert model.scene_embeds.grad is not None
#     assert model.image_backbone.neck.weight.grad is not None
#     assert all(p.grad is None for p in model.image_backbone.aggregator.parameters())
#     打印并断言 trainable 参数量 < 20M(防误解冻 1B)
#     附带打印 scene_embeds 的范数与本次梯度范数(§3.2 观察点基线值,留档)

# [3] LoRA 冷启动与梯度:
#     a. 冷启动等价 —— 比较前提必须是同一份权重,不能分别构造两个模型
#        (neck/scene_embeds 随机初始化不同,输出必然不同,测试失去意义):
#        构造一个冻结版 VggtOmegaImgEncoder(use_lora=false),eval()、use_grid_mask=false,
#        固定输入 + 固定 scene token 记录输出 out0;然后在同一实例上原地执行
#        apply_lora_to_blocks(frame + inter_frame)(先冻结已满足),同输入记录 out1;
#        assert torch.equal(out0, out1)   # rtol=0 且 atol=0 的逐元素严格相等;
#        注意 allclose 默认 rtol=1e-5,单给 atol=0 并不是严格相等
#        (B 零初始化 → new_q/new_v 恰为 0 → qkv+0 逐位不变,bf16 下同样成立)
#     b. 反传后所有 LoRA A/B .grad is not None,教师原始参数 .grad is None
#     c. trainable 计数 = 冻结版 + 6.3M(lora_targets 默认值下)
#     d. lora_targets 含 "trunk" → 构造即抛 ValueError(§3.6)
#     e. 精确 target 断言:共 48 个 qkv wrapper,且全部位于 frame_blocks /
#        inter_frame_blocks;patch_embed.blocks 中 LoRA 参数数为 0
#     f. 固定输入比较手术前后原 qkv 的 K 段 [:, :, dim:2*dim],torch.equal 为 True
#        —— 防止实现被标准 PEFT full-qkv 或其他包装替换后静默改变 K

# [4] 分发零回归(§7.1 硬性要求):
#     a. 必须在新起子进程中执行(subprocess 跑 python -c;若在验收脚本主进程里查,
#        脚本自身早已 import 过新包,sys.modules 断言必然假失败):
#        子进程内构造基线配置的 DrivoRModel,断言 image_backbone 类型是 ImgEncoder,
#        且 "navsim.agents.drivoR_vggt_omega" not in sys.modules(查完整模块名)
#     b. 强证据:与 [0] 的 golden artifact 同配置、同种子、同假 batch 跑 1 个训练 step,
#        loss 逐位一致(bitwise equal;跨环境降级规则见 [0])
#     c. 互斥校验(双向):vggt_omega_1b + vggt_geometry.enabled=true 构造即抛 ValueError;
#        vggt_omega_1b + geo_only=true 同样抛错(校验在分支之前,不会被 geo_only 绕过)

# [5] 模式钉死:model.train() 之后 assert not model.image_backbone.aggregator.training

# [6] 显存冒烟 + checkpoint 生效探针:batch=16(§8.4 口径)、grad_checkpointing=true,
#     单卡 1 个 step(前向+反传+step),冻结版与 LoRA 版各跑一次:
#     a. 不 OOM,记录 torch.cuda.max_memory_allocated 留档 —— §9 预算的实测校准;
#        OOM 则记录并改用 §8.4 回退方案(batch 4 + accumulate 4)重测留档
#     b. checkpoint 生效证明:给 run_block 包计数 hook,
#        1 次前向+反传后计数 == 2 × depth = 48(前向 24 + backward 重算 24;
#        注意 depth=24,每次 run_block 跑 frame+inter-frame 两个 block;
#        若门控失效退化为普通前向,计数只有 24,直接 fail)

# [7] 验证分支与 checkpoint monitor(§3.1):
#     a. assert "drivor" in agent.name().lower()(DrivoRVggtOmegaAgent 含 DrivoR → 命中
#        agent_lightning_module.py:66 的分发条件)
#     b. 假 batch 跑一次 AgentLightningModule.validation_step,断言 logged metrics
#        含 val/score(on_epoch → 产生 val/score_epoch)→ ModelCheckpoint
#        (monitor='val/score_epoch') 可正常保存最优模型

# [8] 优化协议断言(正常配置与 OOM 回退配置各跑一次,§8.4):
#     opt_list, sched_list = agent.get_optimizers()
#     assert opt_list[0].param_groups[0]["lr"] == 2e-4          # sqrt(64/64) × 2e-4
#     T_max_expected = ceil(dataset_size / 64) * num_epochs      # 全局 batch 恒 64
#     断言 SequentialLR 的 ramp+cosine 总步数 == T_max_expected
#     —— 回退配置(BATCH_SIZE=4 + accumulate=4 + agent.batch_size=16)必须给出
#     与正常配置完全相同的 lr 与 T_max,否则回退协议与基线不可比(§4 LR/scheduler 行)

# [9] 评测冒烟(实施顺序第 5 步):eval_vggtomega_backbone.sh 对缩样 split
#     (navmini 或少量 token 过滤)跑通全链路 —— agent 构造、checkpoint 加载(冻结版;
#     LoRA 版加 use_lora=true 再跑一次)、在线 builder、no_grad 前向、PDMS 落盘;
#     确认无 vggt_geometry 注入:先在 VggtOmegaImgEncoder 单测断言输出
#     shape == (B, 64, 256),再给 trajectory_decoder 注册临时 pre-hook,
#     断言收到的 scene_features.shape[1] == 64。生产 output 不新增调试字段
#     (`vggt_geometry_memory_len` 在 enabled=false 时本来就不存在)

# 附带:builder 回归 —— VggtOmegaFeatureBuilder 输出与官方 load_and_preprocess_images(jpg 路径)
#     对 8 个样本 cosine > 0.999(桥接函数已过并联三方验收,此处只是回归确认)
```

## 11. 已知坑

1. **agent 包名不要叫裸 `vggt_omega`**:仓库顶层已有官方包 `vggt_omega/`,同名造成模块解析歧义 → 用 `drivoR_vggt_omega`。
2. **agent 类名必须含 `DrivoR`**(`DrivoRVggtOmegaAgent`):`validation_step` 按 `name()`(= 类名)分发,名字不含 drivor/DrivoR → 走通用分支、`val/score_epoch` 不产生、最优 checkpoint 静默失效(验收 [7] 兜底)。
3. **默认训练配置是 cache-only**(`default_training.yaml:18-19`):新 builder 名没有对应缓存,照默认启动必失败 → 训练脚本必须 `cache_path=null use_cache_without_dataset=false`(在线构建);改离线缓存前先核算 ≈1.1TB 磁盘。
4. **分发 import 必须惰性**(放分支内):顶层 import 会形成 drivoR → drivoR_vggt_omega → drivoR 循环导入。
5. **builder 唯一名必须换**(`drivor_vggt_omega_feature`),否则与 DrivoR 特征缓存互相污染——图像尺寸/归一化完全不同,错读缓存静默得到垃圾输入。
6. **归一化恰好一次**:builder 输出 [0,1],归一化在 `SceneTokenAggregator.forward_full` 内部(官方位置)。encoder 外再做 ImageNet 归一化 = 静默双重归一化(验收 [1] 兜底)。
7. **冻结与 LoRA 手术顺序不可颠倒**:先 `requires_grad_(False)`,再 `apply_lora_to_blocks`(验收 [3]b 兜底)。
8. **Lightning 的 `.train()` 波及主干** → `VggtOmegaImgEncoder.train()` 覆写钉死 eval。**连带铁律**:aggregator 内部任何逻辑不得以 `self.training` 做分支(恒 False)——checkpoint 门控只能用 `torch.is_grad_enabled()`,违反即静默关闭 checkpoint、显存预算作废(验收 [6]b 兜底)。
9. **互斥校验必须前置**:放 backbone 分支内会被 `geo_only=true` 绕过(整个分支不执行),配置静默退化成 geo_only → 校验在所有分支之前(§7a,验收 [4]c 兜底);新 agent yaml 里 `vggt_geometry` 块保留但必须 `enabled: false`。
10. **trunk LoRA 本阶段禁用**:`patch_embed` 在 checkpoint 循环外,trunk 打 LoRA 后 24 层激活无保护,显存预算作废 → 构造期抛错(验收 [3]d 兜底)。
11. **checkpoint 解包要覆盖全部合法格式**:`model`/`state_dict` 双键 + `module.` 前缀(复用 `vggt_geometry.py:203-206` 的逻辑),否则合法 checkpoint 在 strict=True 下报"缺少全部 aggregator 参数",难定位。
12. **scene_embeds 优化动力学与 DINO 版不同**(§3.2):冻结主干下梯度只回流 token;训练不动先查 token 学习率/初始化尺度,再解读读数。
13. **golden artifact 必须在改代码前留档**(§10-[0]):补丁打完后无法再得到可信的"补丁前"结果,bitwise 验收就作废了。
14. **DDP**:requires_grad=False 参数不进 reducer;LoRA 参数正常同步;冻结 1.2B 权重初始化广播一次(~5GB),启动慢属正常。
15. **checkpoint 体积** ≈4.9GB/份(含冻结主干),`save_top_k=1 + save_last` 两份,注意配额。
16. **评测路径**:必须用 `eval_vggtomega_backbone.sh`(§8.5),不要复用并联评测脚本(agent 配置不同,且会带进 `vggt_geometry` 覆盖);`run_pdm_score` 走同一 builder + 在线前向,无缓存依赖;延迟预计 ≥并联在线版,本 agent 不是部署方案,仅记录不优化。
17. **LoRA checkpoint 评测必须 `use_lora=true`**:忘开时 `initialize()` 加载因缺 LoRA 键报错(响亮失败但信息不直观);冻结版/LoRA 版的 checkpoint 与配置一一对应,靠实验名区分(§8.5)。
18. **OOM 回退三键联动**:梯度累积不进 `get_optimizers` 的 LR/T_max 计算(`drivor_agent.py:239-251`),只改 dataloader batch 会静默 lr 减半 + scheduler 慢 4 倍 → 必须同时 `dataloader.params.batch_size=4 + trainer.params.accumulate_grad_batches=4 + agent.batch_size=16`(§8.4,验收 [8] 兜底)。
19. **GPU 数三处一致**:`CUDA_VISIBLE_DEVICES`(决定实际 world size)、`agent.num_gpus`(只进 LR/T_max)、脚本 `NUM_GPUS` 断言——三者不一致时 LR/scheduler 静默算错(§8.4)。
20. **对既有代码的唯一依赖**是 `vggt_geometry.preprocess_arrays_for_teacher`(纯函数);`vggt_geometry.py` 若重构,这是本包唯一跟动点。
21. **权重路径**:本地 `weight/`(单数)vs 服务器 `weights/`(复数),配置统一写 `weights/`,本地用 override/软链接(§8.3)。
22. **溯源**(项目约定,不用 git):实验记录写 `weights/vggt_omega_1b_512.pt` 与 `aggregator.py`/`attention.py`/`dinov2_lora.py` 的 sha256;复用 `vggt_geometry.file_sha256`。
23. **脚本必须自包含**:训练/评测入口都要计算 `REPO_ROOT` 并完整定义 NAVSIM/NUPLAN 环境变量;`set -u` 下不得依赖调用者碰巧预先导出变量。
24. **基线非默认覆盖不可漏**:`long_trajectory_additional_poses=2` 必须同时出现在训练和评测命令;agent yaml 默认 `-1` 不是本实验协议。
25. **LoRA 语义锁定为手写 Q/V-only**:标准 PEFT target 融合 qkv 会同时改变 K,属于新的 full-qkv 变体,不能替换当前实现或混用读数(§3.6,验收 [3]e/f)。

## 12. 实施顺序

0. **golden 留档(改任何代码之前)**:`scripts/vggt_omega_acceptance_checks.py --capture-golden` → 验收 [0](loss/输出/state sha256/环境版本存档)。
1. `vggt_omega_backbone.py`(SceneTokenAggregator + VggtOmegaImgEncoder,先不接 LoRA)→ 验收 [1][2]。
2. `apply_lora_to_blocks` 接入 → 验收 [3](同权重原地手术的冷启动等价 + 梯度 + 参数计数 + trunk 拒绝)。
3. `drivor_model.py` 前置互斥校验 + 分发 → 验收 [4](原 model_name 走原路径,golden bitwise,互斥双向)。
4. `vggt_omega_features.py` + `vggt_omega_agent.py` + agent yaml + 训练/评测脚本 → 假 batch 冒烟 1 个 step + 验收 [5][6][7][8](显存留档、验证分支、优化协议双配置断言)。
5. 评测冒烟:`eval_vggtomega_backbone.sh` 缩样 split 全链路(冻结版 + LoRA 开关各一次)→ 验收 [9]。
6. 验收全过、留档(含 max_memory 与 checkpoint sha256)。
7. 冻结版训练:`train_vggtomega_backbone.sh`,navtrain 10 epochs × 3 seeds(消融协议);评测:`eval_vggtomega_backbone.sh`(navtest);读数记入实验台账:
   (vggtomega_backbone − 基线 = ___ ± ___;TLC 子分 = ___;**若意外接近或超过基线 → `use_lora=true` 跑 vggtomega_backbone_lora,训练评测两侧都开**)。

## 13. 判定与读数(锁定)

- 训练协议:navtrain 10 epochs、AdamW lr 2e-4、**batch 16/卡 × 4 卡 = 全局 64**(与基线脚本同口径,`base_batch_size: 64` 下 lr 恰 1:1)、`long_trajectory_additional_poses=2`、3 seeds,同基线消融协议;OOM 回退保持同一优化协议(三键联动,§8.4,验收 [8] 断言 lr 与 T_max 不变);
- 评测协议:`eval_vggtomega_backbone.sh`,navtest,与训练同 agent 配置和 `long_trajectory_additional_poses=2`(LoRA 版两侧同开 `use_lora=true`);
- 无论结果如何都有话可说:显著掉点 → 佐证"混合方案而非替换主干"的项目主线;意外不掉 → 几何主干可直接承载规划,蒸馏系/并联的必要性需重估。

## 14. 方案演化史(否掉的路,勿复走)

| 版本 | 设计差异 | 结局 |
|---|---|---|
| v1 | 修改 drivoR 包 5+ 文件(仿并联落点) | 否:改动过大 |
| v2 | drivoR 包零修改,空相机配置借道父类构造的 hack | 否:隐患多于收益 |
| v3 | 最小哨兵分发 + 独立新包,联合前向,LoRA 一并实现 | 基本定稿 |
| v4 | v3 + 首轮审查修正:checkpoint 门控改梯度开关、拆 `forward_full` 供 S=0 验收、权重加载 strict=True、零回归断言改子进程、与并联互斥、scene_embeds 观察点、depth=24 修正、引用对齐当前代码 | 定稿(后被 v5 取代) |
| v5 | v4 + 首轮 `question.md` 修正:agent 类名 `DrivoRVggtOmegaAgent`(验证分支/monitor)、训练入口改 shell 脚本并显式在线特征 + 锁定协议(全局 batch 64 与基线同口径,修正早前"全局 16"口径错误)、checkpoint 解包对齐 teacher 实现、trunk LoRA 构造期禁用、互斥校验前置(堵 geo_only 绕过)、golden artifact 前置留档、LoRA 实验名 `vggtomega_backbone_lora` | 定稿(后被 v6 取代) |
| v6 | v5 + 复审 `question.md` 修正:OOM 回退三键联动(`agent.batch_size=16` 显式覆盖插值,梯度累积不进 LR/T_max 计算,验收 [8]);新增评测入口 `eval_vggtomega_backbone.sh` + LoRA 开关同步 + navtest 缩样冒烟(验收 [9]);训练脚本锁定 `CUDA_VISIBLE_DEVICES` 并前置断言 device_count == NUM_GPUS;LoRA 冷启动等价改为同权重原地手术 + `torch.equal` 严格相等;并联脚本引用更新为现行命名 | 定稿(后被 v7 取代) |
| v7(本文) | v6 + 第三轮 `question.md` 修正:训练/评测脚本补全 REPO_ROOT 与全部环境变量;两侧显式锁定 `long_trajectory_additional_poses=2` 并加入基线单变量对照表;评测 override 全部进入真实命令且 `$@` 保持最后;memory 64-token 验收改为 encoder shape + decoder pre-hook;锁定手写 Q/V-only LoRA,标准 PEFT full-qkv 另立变体,验收增加 48 target/K 段/trunk 断言 | ✅ 定稿 |

## 状态

☐ golden 留档 → ☐ 代码实现 → ☐ 验收断言留档 → ☐ 冻结版训练中 → ☐ 完成(vggtomega_backbone_lora:☐ 未触发 / ☐ 已触发)
